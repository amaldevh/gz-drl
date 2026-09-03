// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

/*
 * VideoRecordingManager
 *
 * Records videos from free-floating rendering cameras created directly on
 * the gz-rendering Scene owned by the (modified) gz-sim Sensors system.
 *
 * Threading model:
 *  - StartRecording / UpdateRecordingCamPos / StopRecording are thread-safe
 *    and may be called from anywhere (transport callbacks, PostUpdate, ...).
 *    They only enqueue commands.
 *  - OnRender() must be called from the Sensors render thread, inside
 *    SensorsPrivate::RunOnce(), between scene->PreRender() and
 *    scene->PostRender(). It drains the command queue, renders due frames
 *    and feeds them to the encoders.
 *  - RequiresRender() is lock-free and safe to call from the sim thread
 *    (Sensors::PostUpdate) to decide whether the render thread must wake up.
 */

#ifndef VIDEO_RECORDING_MANAGER_HH_
#define VIDEO_RECORDING_MANAGER_HH_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <gz/math/Pose3.hh>
#include <gz/rendering/RenderTypes.hh>

namespace custom
{
  class VideoRecordingManager
  {
    /// Constructor and destructor are defined in the .cc where Recording is
    /// complete, so unique_ptr<Recording> can be destroyed there.
    public: VideoRecordingManager();
    public: ~VideoRecordingManager();

    /// Optional per-recording camera settings.
    public: struct CameraConfig
    {
      CameraConfig(){};
      unsigned int width{1280};
      unsigned int height{720};
      unsigned int fps{25};
      /// 0 lets gz-common pick a bitrate automatically.
      unsigned int bitrate{0};
      /// Horizontal field of view in radians.
      double hfovRad{1.047};
      double nearClip{0.1};
      double farClip{1000.0};
      unsigned int antiAliasing{2};
    };

    /// Lifecycle state of a recording, queryable from any thread.
    public: enum class State
    {
      /// No recording with this name (never started, failed, or stopped).
      kNone,
      /// Start command queued; camera not created yet.
      kQueued,
      /// Camera created and encoder running.
      kActive,
    };

    /// \brief Thread-safe query of a recording's lifecycle state.
    public: State RecordingState(const std::string &_camName) const;

    /// \brief Set the scene. Call once from the render thread after the
    /// Sensors plugin obtains its scene (renderUtil.Scene()).
    public: void SetScene(gz::rendering::ScenePtr _scene);

    /// \brief Request a new recording. The camera is created on the next
    /// render iteration. Output format is derived from _outputName's
    /// extension (mp4, avi, ogv); defaults to mp4.
    /// \return false if _camName is empty or already queued/recording.
    public: bool StartRecording(const std::string &_camName,
                                const gz::math::Pose3d &_pose,
                                const std::string &_outputName,
                                const CameraConfig &_config = CameraConfig());

    /// \brief Request a camera pose update (world pose).
    public: bool UpdateRecordingCamPos(const std::string &_camName,
                                       const gz::math::Pose3d &_pose);

    /// \brief Finalize the video and destroy the camera on the next render
    /// iteration.
    public: bool StopRecording(const std::string &_camName);

    /// \brief Stop everything and destroy all cameras. Call from the render
    /// thread on shutdown (e.g. Sensors destructor / Stop()).
    public: void Shutdown();

    /// \brief Whether the render thread needs to run for the recorder:
    /// pending commands, or an active recording with a frame due at _simTime.
    /// Lock-free; safe from the sim thread.
    public: bool RequiresRender(
        const std::chrono::steady_clock::duration &_simTime) const;

    /// \brief Process commands and capture due frames. Render thread only,
    /// between scene->PreRender() and scene->PostRender().
    /// \param[in] _simTime Current sim time (updateTimeApplied in RunOnce).
    public: void OnRender(const std::chrono::steady_clock::duration &_simTime);

    private: struct Command
    {
      enum class Type { kStart, kUpdatePose, kStop };
      Type type;
      std::string name;
      gz::math::Pose3d pose;
      std::string output;
      CameraConfig config;
    };

    private: struct Recording;  // defined in .cc (owns the VideoEncoder)

    private: void ProcessCommands();
    private: void DoStart(const Command &_cmd);
    private: void DoStop(const std::string &_name);
    /// Remove a recording's lifecycle state entry (thread-safe).
    private: void ClearState(const std::string &_name);
    /// Recompute the lock-free "earliest next frame" hint.
    private: void RefreshNextFrameHint();

    private: gz::rendering::ScenePtr scene;

    private: std::mutex commandMutex;
    private: std::vector<Command> commands;
    private: std::atomic<std::size_t> pendingCommands{0};

    /// Lifecycle state per recording name. Written by StartRecording (any
    /// thread) and DoStart/DoStop (render thread); read from any thread.
    private: mutable std::mutex stateMutex;
    private: std::unordered_map<std::string, State> states;

    /// Earliest sim time (ns) at which any recording needs a frame.
    /// int64 max when there are no active recordings.
    private: std::atomic<std::int64_t> nextFrameTimeNs{
        std::numeric_limits<std::int64_t>::max()};

    /// Render-thread only.
    private: std::unordered_map<std::string,
        std::unique_ptr<Recording>> recordings;
  };
}

#endif
