// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#include "sensor/video_recording_manager.hh"

#include <limits>

#include <gz/common/Console.hh>
#include <gz/common/VideoEncoder.hh>
#include <gz/rendering/Camera.hh>
#include <gz/rendering/Image.hh>
#include <gz/rendering/PixelFormat.hh>
#include <gz/rendering/Scene.hh>
#include <gz/rendering/Visual.hh>

using namespace custom;

namespace
{
  /// Derive the encoder format from the output file extension.
  std::string FormatFromFilename(const std::string &_filename)
  {
    auto dot = _filename.find_last_of('.');
    if (dot == std::string::npos || dot + 1 >= _filename.size())
      return "mp4";
    std::string ext = _filename.substr(dot + 1);
    if (ext == "mp4" || ext == "avi" || ext == "ogv")
      return ext;
    gzwarn << "Unsupported video extension [" << ext
           << "], falling back to mp4" << std::endl;
    return "mp4";
  }
}

/// One active recording. Lives entirely in the render thread.
struct VideoRecordingManager::Recording
{
  gz::rendering::CameraPtr camera;
  gz::rendering::Image image;
  gz::common::VideoEncoder encoder;
  std::string outputName;
  CameraConfig config;
  /// Sim time at which the next frame should be captured.
  std::chrono::steady_clock::duration nextFrameTime{0};
  std::chrono::steady_clock::duration framePeriod{0};
  /// The camera is created mid-frame (after this iteration's
  /// scene->PreRender()), so its scene-graph state is not initialized yet.
  /// Skip capturing until the next iteration's PreRender has run.
  bool warmup{true};
};

//////////////////////////////////////////////////
VideoRecordingManager::VideoRecordingManager() = default;

//////////////////////////////////////////////////
VideoRecordingManager::~VideoRecordingManager() = default;

//////////////////////////////////////////////////
void VideoRecordingManager::SetScene(gz::rendering::ScenePtr _scene)
{
  this->scene = _scene;
}

//////////////////////////////////////////////////
bool VideoRecordingManager::StartRecording(const std::string &_camName,
    const gz::math::Pose3d &_pose, const std::string &_outputName,
    const CameraConfig &_config)
{
  if (_camName.empty() || _outputName.empty())
    return false;

  {
    std::lock_guard<std::mutex> stateLock(this->stateMutex);
    if (this->states.count(_camName))
    {
      gzwarn << "Recording [" << _camName << "] already queued or active"
             << std::endl;
      return false;
    }
    this->states[_camName] = State::kQueued;
  }

  std::lock_guard<std::mutex> lock(this->commandMutex);
  this->commands.push_back(Command{Command::Type::kStart, _camName, _pose,
                                   _outputName, _config});
  this->pendingCommands = this->commands.size();
  return true;
}

//////////////////////////////////////////////////
VideoRecordingManager::State VideoRecordingManager::RecordingState(
    const std::string &_camName) const
{
  std::lock_guard<std::mutex> lock(this->stateMutex);
  auto it = this->states.find(_camName);
  return it == this->states.end() ? State::kNone : it->second;
}

//////////////////////////////////////////////////
bool VideoRecordingManager::UpdateRecordingCamPos(const std::string &_camName,
    const gz::math::Pose3d &_pose)
{
  if (_camName.empty())
    return false;

  std::lock_guard<std::mutex> lock(this->commandMutex);
  this->commands.push_back(Command{Command::Type::kUpdatePose, _camName,
                                   _pose, "", CameraConfig()});
  this->pendingCommands = this->commands.size();
  return true;
}

//////////////////////////////////////////////////
bool VideoRecordingManager::StopRecording(const std::string &_camName)
{
  if (_camName.empty())
    return false;

  std::lock_guard<std::mutex> lock(this->commandMutex);
  this->commands.push_back(Command{Command::Type::kStop, _camName,
                                   gz::math::Pose3d(), "", CameraConfig()});
  this->pendingCommands = this->commands.size();
  return true;
}

//////////////////////////////////////////////////
bool VideoRecordingManager::RequiresRender(
    const std::chrono::steady_clock::duration &_simTime) const
{
  if (this->pendingCommands.load() > 0)
    return true;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      _simTime).count() >= this->nextFrameTimeNs.load();
}

//////////////////////////////////////////////////
void VideoRecordingManager::OnRender(
    const std::chrono::steady_clock::duration &_simTime)
{
  if (!this->scene)
    return;

  this->ProcessCommands();

  for (auto &[name, rec] : this->recordings)
  {
    if (!rec->encoder.IsEncoding())
      continue;
    if (rec->warmup)
    {
      // First iteration after creation: the camera missed this frame's
      // scene->PreRender() pass. Capture starts on the next iteration.
      rec->warmup = false;
      continue;
    }
    if (_simTime < rec->nextFrameTime)
      continue;

    // scene->PreRender() has already been called by SensorsPrivate::RunOnce,
    // but a camera created in this iteration (via ProcessCommands above)
    // missed that pass, so its render target / compositor workspace is not
    // built yet. Camera::PreRender() rebuilds it if needed (no-op otherwise).
    rec->camera->PreRender();
    rec->camera->Render();
    rec->camera->PostRender();
    rec->camera->Copy(rec->image);

    // Timestamp frames with sim time; the encoder paces/duplicates frames
    // internally to match the requested fps.
    std::chrono::steady_clock::time_point t{_simTime};
    if (!rec->encoder.AddFrame(rec->image.Data<unsigned char>(),
        rec->config.width, rec->config.height, t))
    {
      gzwarn << "Failed to add video frame for [" << name << "]" << std::endl;
    }

    rec->nextFrameTime = _simTime + rec->framePeriod;
  }

  this->RefreshNextFrameHint();
}

//////////////////////////////////////////////////
void VideoRecordingManager::ProcessCommands()
{
  std::vector<Command> cmds;
  {
    std::lock_guard<std::mutex> lock(this->commandMutex);
    cmds.swap(this->commands);
    this->pendingCommands = 0;
  }

  for (const auto &cmd : cmds)
  {
    switch (cmd.type)
    {
      case Command::Type::kStart:
        this->DoStart(cmd);
        break;
      case Command::Type::kUpdatePose:
      {
        auto it = this->recordings.find(cmd.name);
        if (it != this->recordings.end())
          it->second->camera->SetLocalPose(cmd.pose);
        else
          gzwarn << "UpdateRecordingCamPos: no recording named [" << cmd.name
                 << "]" << std::endl;
        break;
      }
      case Command::Type::kStop:
        this->DoStop(cmd.name);
        break;
    }
  }
}

//////////////////////////////////////////////////
void VideoRecordingManager::DoStart(const Command &_cmd)
{
  if (this->recordings.count(_cmd.name))
  {
    gzwarn << "Recording [" << _cmd.name << "] already exists" << std::endl;
    return;
  }

  auto rec = std::make_unique<Recording>();
  rec->config = _cmd.config;
  rec->outputName = _cmd.output;

  // Prefix to avoid clashing with real sensor camera names in the scene.
  const std::string camName = "video_recorder::" + _cmd.name;
  rec->camera = this->scene->CreateCamera(camName);
  if (!rec->camera)
  {
    gzerr << "Failed to create camera [" << camName << "]" << std::endl;
    this->ClearState(_cmd.name);
    return;
  }

  const auto &c = rec->config;
  rec->camera->SetImageWidth(c.width);
  rec->camera->SetImageHeight(c.height);
  rec->camera->SetImageFormat(gz::rendering::PF_R8G8B8);
  rec->camera->SetAspectRatio(
      static_cast<double>(c.width) / static_cast<double>(c.height));
  rec->camera->SetHFOV(c.hfovRad);
  rec->camera->SetNearClipPlane(c.nearClip);
  rec->camera->SetFarClipPlane(c.farClip);
  rec->camera->SetAntiAliasing(c.antiAliasing);
  rec->camera->SetLocalPose(_cmd.pose);
  this->scene->RootVisual()->AddChild(rec->camera);

  rec->image = rec->camera->CreateImage();
  rec->framePeriod = std::chrono::nanoseconds(
      static_cast<std::int64_t>(1e9 / c.fps));
  // Capture the first frame on the very next OnRender call.
  rec->nextFrameTime = std::chrono::steady_clock::duration::zero();

  // gz-common's VideoEncoder writes into a working file while encoding;
  // SaveToFile() moves it to the destination on stop. With a blank filename
  // every encoder shares a single "TMP_RECORDING.<format>" in the cwd,
  // which corrupts concurrent recordings — use a per-recording temp file
  // next to the destination instead (same filesystem => atomic move). The
  // temp file must keep a real video extension because ffmpeg selects the
  // muxer from it, so ".tmp" goes before the extension.
  // Note: encoding straight into _cmd.output is NOT an option, because
  // VideoEncoder::Reset() (called from SaveToFile and the destructor)
  // deletes the encoder's working file.
  const std::string format = FormatFromFilename(_cmd.output);
  const auto slash = _cmd.output.find_last_of('/');
  auto dot = _cmd.output.find_last_of('.');
  if (dot == std::string::npos ||
      (slash != std::string::npos && dot < slash))
    dot = _cmd.output.size();
  const std::string tmpName =
      _cmd.output.substr(0, dot) + ".tmp." + format;
  if (!rec->encoder.Start(format, tmpName,
      c.width, c.height, c.fps, c.bitrate))
  {
    gzerr << "Failed to start video encoder for [" << _cmd.name << "]"
          << std::endl;
    this->scene->DestroySensor(rec->camera);
    this->ClearState(_cmd.name);
    return;
  }

  gzmsg << "Started recording [" << _cmd.name << "] -> [" << _cmd.output
        << "] (" << c.width << "x" << c.height << " @ " << c.fps << " fps, "
        << format << ")" << std::endl;
  this->recordings[_cmd.name] = std::move(rec);

  {
    std::lock_guard<std::mutex> lock(this->stateMutex);
    this->states[_cmd.name] = State::kActive;
  }
}

//////////////////////////////////////////////////
void VideoRecordingManager::DoStop(const std::string &_name)
{
  auto it = this->recordings.find(_name);
  if (it == this->recordings.end())
  {
    gzwarn << "StopRecording: no recording named [" << _name << "]"
           << std::endl;
    this->ClearState(_name);
    return;
  }

  auto &rec = it->second;
  if (rec->encoder.IsEncoding())
  {
    // SaveToFile() stops the encoder and moves the temp file into place.
    if (rec->encoder.SaveToFile(rec->outputName))
      gzmsg << "Video saved to [" << rec->outputName << "]" << std::endl;
    else
      gzerr << "Failed to save video to [" << rec->outputName << "]"
            << std::endl;
  }

  this->scene->DestroySensor(rec->camera);
  this->recordings.erase(it);
  this->ClearState(_name);
}

//////////////////////////////////////////////////
void VideoRecordingManager::ClearState(const std::string &_name)
{
  std::lock_guard<std::mutex> lock(this->stateMutex);
  this->states.erase(_name);
}

//////////////////////////////////////////////////
void VideoRecordingManager::Shutdown()
{
  std::vector<std::string> names;
  names.reserve(this->recordings.size());
  for (const auto &[name, rec] : this->recordings)
    names.push_back(name);
  for (const auto &name : names)
    this->DoStop(name);
  this->RefreshNextFrameHint();
}

//////////////////////////////////////////////////
void VideoRecordingManager::RefreshNextFrameHint()
{
  std::int64_t next = std::numeric_limits<std::int64_t>::max();
  for (const auto &[name, rec] : this->recordings)
  {
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        rec->nextFrameTime).count();
    if (ns < next)
      next = ns;
  }
  this->nextFrameTimeNs = next;
}
