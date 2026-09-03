// /*
//  * Copyright 2015 Fadri Furrer, ASL, ETH Zurich, Switzerland
//  * Copyright 2015 Michael Burri, ASL, ETH Zurich, Switzerland
//  * Copyright 2015 Mina Kamel, ASL, ETH Zurich, Switzerland
//  * Copyright 2015 Janosch Nikolic, ASL, ETH Zurich, Switzerland
//  * Copyright 2015 Markus Achtelik, ASL, ETH Zurich, Switzerland
//  * Copyright 2016 Geoffrey Hunter <gbmhunter@gmail.com>
//  * Copyright (C) 2019 Open Source Robotics Foundation
//  * Copyright (C) 2022 Benjamin Perseghetti, Rudis Laboratories
//  * Copyright (C) 2024 Amal Dev Haridevan, SDCNLab, York University, Toronto, ON
//  *
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  *     http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  *
//  */
#include "plugins/rotor_plugin.hh"
#include <mutex>
#include <string>
#include <optional>
#include <cmath>
#include <limits>
#include <sstream>
#include <algorithm>
#include <vector>

#include <gz/msgs/actuators.pb.h>

#include <gz/common/Profiler.hh>
#include <gz/plugin/Register.hh>
#include <gz/transport/Node.hh>

#include <gz/math/Helpers.hh>
#include <gz/math/Pose3.hh>
#include <gz/math/Vector3.hh>
#include <gz/msgs/Utility.hh>

#include <sdf/sdf.hh>

#include "gz/sim/components/Actuators.hh"
#include "gz/sim/components/ExternalWorldWrenchCmd.hh"
#include "gz/sim/components/JointAxis.hh"
#include "gz/sim/components/JointVelocity.hh"
#include "gz/sim/components/JointVelocityCmd.hh"
#include "gz/sim/components/ParentLinkName.hh"
#include "gz/sim/components/Pose.hh"
#include "gz/sim/components/Wind.hh"
#include "gz/sim/components/AngularVelocity.hh"
#include "gz/sim/components/LinearVelocity.hh"
#include "gz/sim/Link.hh"
#include "gz/sim/Joint.hh"
#include "gz/sim/Model.hh"
#include "gz/sim/Util.hh"

using namespace gz;
using namespace sim;
using namespace systems;

#ifndef MULTIROTOR_SIMD
#define MULTIROTOR_SIMD 1
#endif
#ifndef MULTIROTOR_SIMD_FILTER
#define MULTIROTOR_SIMD_FILTER 1
#endif

#if defined(MULTIROTOR_OMP_SIMD)
  #define MR_OMP_SIMD _Pragma("omp simd")
#else
  #if defined(__clang__)
    #define MR_OMP_SIMD _Pragma("clang loop vectorize(enable) interleave(enable)")
  #elif defined(__GNUG__)
    #define MR_OMP_SIMD _Pragma("GCC ivdep")
  #else
    #define MR_OMP_SIMD
  #endif
#endif

// ---------------------- Utility ----------------------
template<typename T>
inline T Square(const T v) { return v * v; }

// Minimal first order filter (kept for non-SIMD fallback)
template <typename T>
class FirstOrderFilter {
 public:
  FirstOrderFilter(double timeConstantUp, double timeConstantDown, T initial)
      : timeConstantUp_(timeConstantUp),
        timeConstantDown_(timeConstantDown),
        previousState_(initial) {}
  T UpdateFilter(T input, double dt) {
    const double tau = (input > previousState_) ? timeConstantUp_ : timeConstantDown_;
    const double alpha = std::exp(-dt / std::max(tau, 1e-9));
    previousState_ = alpha * previousState_ + (1.0 - alpha) * input;
    return previousState_;
  }
 private:
  double timeConstantUp_;
  double timeConstantDown_;
  T previousState_;
};

namespace helpers{
  template <typename T>
  std::vector<T> split_to_vec(const std::string& input) {
    std::vector<T> result;
    std::istringstream iss(input);
    T value;
    while (iss >> value) result.push_back(value);
    return result;
  }

  template<typename T>
  bool  get_vector_from_sdf(const std::shared_ptr<const sdf::Element> &_sdf,
                            std::string name,
                            std::vector<T>& param,
                            std::vector<T>& def){
    std::string val;
    const bool ret =  _sdf->Get(name, val, std::string{});
    if (!ret){ param = def; return false; }
    auto res = split_to_vec<T>(val);
    param.assign(res.begin(), res.end());
    return true;
  }

  struct RotorDataStruct{
    explicit RotorDataStruct(int numRotors){
      realMotorVelocities.resize(numRotors);
      absMotorVels.resize(numRotors);
      thrusts_.resize(numRotors);
      thrusts.resize(numRotors);
      heights.resize(numRotors);
      groundEffects.resize(numRotors);
      thrustWorlds.resize(numRotors);
      airDragRotors.resize(numRotors);
      totalThrusts.resize(numRotors);
      poseDiffs.resize(numRotors);
      dragTorqueRotors.resize(numRotors);
      dragTorqueParents.resize(numRotors);
      rollingMoments.resize(numRotors);
      gyroWorlds.resize(numRotors);
      minusGs.resize(numRotors);
      parentWorldTorques.resize(numRotors);
      // bodyDrags.resize(numRotors);
      refMotorRotVels.resize(numRotors);

      rotorLinks.resize(numRotors);
      rotorWorldPoses.resize(numRotors);
      jointPoseComps.resize(numRotors);
      jointAxisComps.resize(numRotors);
      jointVelComps.resize(numRotors);
      jointWorldPoses.resize(numRotors);
      jointAxisWorlds.resize(numRotors);
      relWindWorlds.resize(numRotors);
      vPerps.resize(numRotors);

      rotorWrenchComps.resize(numRotors);
    }

    // numerics
    std::vector<double> realMotorVelocities;
    std::vector<double> absMotorVels;
    std::vector<double> thrusts_;
    std::vector<double> thrusts;
    std::vector<double> heights;
    std::vector<double> groundEffects;
    std::vector<math::Vector3d> thrustWorlds;
    std::vector<math::Vector3d> airDragRotors;
    std::vector<math::Vector3d> totalThrusts;
    std::vector<math::Pose3d>    poseDiffs;
    std::vector<math::Vector3d> dragTorqueRotors;
    std::vector<math::Vector3d> dragTorqueParents;
    std::vector<math::Vector3d> rollingMoments;
    std::vector<math::Vector3d> gyroWorlds;
    std::vector<math::Vector3d> minusGs;
    std::vector<math::Vector3d> parentWorldTorques;
    // std::vector<math::Vector3d> bodyDrags;
    std::vector<double> refMotorRotVels;

    // cached handles & comps
    std::vector<Link> rotorLinks;
    std::vector<std::optional<math::Pose3d>> rotorWorldPoses;
    std::vector<components::Pose*>          jointPoseComps;
    std::vector<components::JointAxis*>     jointAxisComps;
    std::vector<components::JointVelocity*> jointVelComps;
    std::vector<math::Pose3d>               jointWorldPoses;
    std::vector<math::Vector3d>             jointAxisWorlds;
    std::vector<math::Vector3d>             relWindWorlds;
    std::vector<math::Vector3d>             vPerps;

    // wrench component on each rotor link (cached once)
    std::vector<components::ExternalWorldWrenchCmd*> rotorWrenchComps;
  };
}

class gz::sim::systems::MultiRotorPluginPrivate
{
  public: void UpdateForcesAndMoments(EntityComponentManager &_ecm);
  public: bool BindOnce(EntityComponentManager &_ecm);

  // Entities and names
  public: std::vector<Entity> jointEntities;
  public: std::vector<std::string> jointNames;
  public: std::vector<Entity> linkEntities;
  public: std::vector<std::string> linkNames;
  public: std::vector<std::string> otherUavNames;
  public: Entity parentLinkEntity{kNullEntity};
  public: std::string parentLinkName;
  public: Model model{kNullEntity};
  public: Link parentLinkCache{kNullEntity};

  // Params
  public: double samplingTime{0.001};
  public: std::vector<int> turningDirections;

  public: double maxRotVelocity{838.0};
  public: math::Vector3d momentConstantQuadraticParams;
  public: double motorConstant{8.54858e-06};

  public: std::vector<double> refMotorInput; // SIMD-friendly
  public: math::Vector3d downWashCoefficients;
  public: double rollingMomentCoefficient{1.0e-6};
  public: double rotorDragCoefficient{1.0e-4};
  public: double rotorVelocitySlowdownSim{10.0};
  public: double rotorVelocitySlowdownInv{0.1};
  public: double timeConstantDown{1.0 / 40.0};
  public: double timeConstantUp{1.0 / 80.0};
  public: double rotorInertia{0.0};

#if !MULTIROTOR_SIMD_FILTER
  public: std::vector<std::unique_ptr<FirstOrderFilter<double>>> rotorVelocityFilters; // fallback
#else
  public: std::vector<double> filterPrev;   // SIMD
  public: std::vector<double> cmds;         // SIMD
#endif

  public: math::Vector3d thrustConstantQuadraticParams{0,0,0};
  // public: math::Vector3d airDragCoeffs{0,0,0};
  public: double groundEffectConstant{1.0};

  // Cached/aux
  public: Entity windEntity{kNullEntity};
  public: bool initialized{false};
  public: int numRotors{0};
  public: bool bound{false};
  public: std::unique_ptr<helpers::RotorDataStruct> rotorDatas;
  public: components::ExternalWorldWrenchCmd* parentWrench{nullptr};
};

// ---------------------- MultiRotorPlugin ----------------------
MultiRotorPlugin::MultiRotorPlugin()
  : dataPtr(std::make_unique<MultiRotorPluginPrivate>())
{
}

void MultiRotorPlugin::Configure(const Entity &_entity,
    const std::shared_ptr<const sdf::Element> &_sdf,
    EntityComponentManager &_ecm,
    EventManager &/*_eventMgr*/)
{
  this->dataPtr->model = Model(_entity);
  if (!this->dataPtr->model.Valid(_ecm))
  {
    gzerr << "MultiRotorPlugin must attach to a model entity. Init failed.";
    return;
  }

  auto sdfClone = _sdf->Clone();

  if (sdfClone->HasElement("parentLink"))
  {
    this->dataPtr->parentLinkName  = sdfClone->Get<std::string>("parentLink");
  }
  else
  {
    gzerr << "Missing <linkNames>.";
    return;
  }
  if (sdfClone->HasElement("linkNames")){
    helpers::get_vector_from_sdf(sdfClone, "linkNames",
                                 this->dataPtr->linkNames,
                                 this->dataPtr->linkNames);
  }
  if (this->dataPtr->linkNames.empty())
  {
    gzerr << "Missing <linkNames>.";
    return;
  }
  if (sdfClone->HasElement("otherUavNames")){
    helpers::get_vector_from_sdf(sdfClone, "otherUavNames",
                                 this->dataPtr->otherUavNames,
                                 this->dataPtr->otherUavNames);
  }

  this->dataPtr->numRotors = static_cast<int>(this->dataPtr->linkNames.size());
  this->dataPtr->rotorDatas = std::make_unique<helpers::RotorDataStruct>(this->dataPtr->numRotors);
  this->dataPtr->refMotorInput.resize(this->dataPtr->numRotors, 0.0);
#if MULTIROTOR_SIMD_FILTER
  this->dataPtr->filterPrev.assign(this->dataPtr->numRotors, 0.0);
  this->dataPtr->cmds.assign(this->dataPtr->numRotors, 0.0);
#else
  this->dataPtr->rotorVelocityFilters.clear();
  this->dataPtr->rotorVelocityFilters.reserve(this->dataPtr->numRotors);
  for (int i = 0; i<this->dataPtr->numRotors; ++i)
    this->dataPtr->rotorVelocityFilters.push_back(std::make_unique<FirstOrderFilter<double>>(
          this->dataPtr->timeConstantUp, this->dataPtr->timeConstantDown, 0.0));
#endif

  if (sdfClone->HasElement("turningDirections"))
  {
    helpers::get_vector_from_sdf(sdfClone, "turningDirections",
                                 this->dataPtr->turningDirections,
                                 this->dataPtr->turningDirections);
    if (this->dataPtr->turningDirections.size() != this->dataPtr->linkNames.size()){
      gzerr << "Please specify <turningDirections> for each rotor ('+1' or '-1').";
      return;
    }
  }

  // Optional params
  sdfClone->Get<math::Vector3d>("downWashCoefficients",
      this->dataPtr->downWashCoefficients, this->dataPtr->downWashCoefficients);
  sdfClone->Get<double>("rotorDragCoefficient",
      this->dataPtr->rotorDragCoefficient, this->dataPtr->rotorDragCoefficient);
  sdfClone->Get<double>("rollingMomentCoefficient",
      this->dataPtr->rollingMomentCoefficient, this->dataPtr->rollingMomentCoefficient);
  sdfClone->Get<double>("maxRotVelocity",
      this->dataPtr->maxRotVelocity, this->dataPtr->maxRotVelocity);
  sdfClone->Get<math::Vector3d>("momentConstantQuadraticParams",
      this->dataPtr->momentConstantQuadraticParams, this->dataPtr->momentConstantQuadraticParams);

  sdfClone->Get<double>("timeConstantUp",
      this->dataPtr->timeConstantUp, this->dataPtr->timeConstantUp);
  sdfClone->Get<double>("timeConstantDown",
      this->dataPtr->timeConstantDown, this->dataPtr->timeConstantDown);
  auto fixTau = [](double &tau, double fallback){ if (!(tau>0.0) || !std::isfinite(tau)) tau = fallback; };
  fixTau(this->dataPtr->timeConstantUp,   1e-6);
  fixTau(this->dataPtr->timeConstantDown, 1e-6);
  sdfClone->Get<double>("rotorVelocitySlowdownSim",
      this->dataPtr->rotorVelocitySlowdownSim, this->dataPtr->rotorVelocitySlowdownSim);

  sdfClone->Get<math::Vector3d>("thrustConstantQuadraticParams",
      this->dataPtr->thrustConstantQuadraticParams, this->dataPtr->thrustConstantQuadraticParams);
  // sdfClone->Get<math::Vector3d>("airDragCoeffs",
  //     this->dataPtr->airDragCoeffs, this->dataPtr->airDragCoeffs);
  sdfClone->Get<double>("groundEffectConstant",
      this->dataPtr->groundEffectConstant, this->dataPtr->groundEffectConstant);
  sdfClone->Get<double>("rotorInertia",
      this->dataPtr->rotorInertia, this->dataPtr->rotorInertia);

  // // distribute body air drag across N rotors
  // this->dataPtr->airDragCoeffs *= 1.0/static_cast<double>(this->dataPtr->numRotors);

  // Precompute reciprocal to replace divisions in hot path
  this->dataPtr->rotorVelocitySlowdownInv = 1.0 / this->dataPtr->rotorVelocitySlowdownSim;
  this->dataPtr->initialized = true;
}

void MultiRotorPlugin::PreUpdate(const UpdateInfo &_info,
    EntityComponentManager &_ecm)
{
  if (_info.dt < std::chrono::steady_clock::duration::zero())
    gzwarn << "Detected time jump backward. System stability may be affected.";
  if (_info.paused || ! this->dataPtr->initialized)
    return;

  // Resolve entities once
  if (this->dataPtr->linkEntities.empty())
  {
    this->dataPtr->linkEntities.clear();
    for (const auto &linkName : this->dataPtr->linkNames){
      const auto e = this->dataPtr->model.LinkByName(_ecm, linkName);
      this->dataPtr->linkEntities.push_back(e);
      Link link(e);
      link.EnableVelocityChecks(_ecm, true);
      link.EnableAccelerationChecks(_ecm, true);
    }
    // cache link wrappers once
    for (size_t i=0; i<this->dataPtr->linkEntities.size(); ++i)
      this->dataPtr->rotorDatas->rotorLinks[i] = Link(this->dataPtr->linkEntities[i]);
  }

  if (this->dataPtr->parentLinkEntity == kNullEntity)
  {
    this->dataPtr->parentLinkEntity = this->dataPtr->model.LinkByName(_ecm, this->dataPtr->parentLinkName);
    if (this->dataPtr->parentLinkEntity != kNullEntity)
    {
      this->dataPtr->parentLinkCache = Link(this->dataPtr->parentLinkEntity);
      this->dataPtr->parentLinkCache.EnableVelocityChecks(_ecm, true);
      this->dataPtr->parentLinkCache.EnableAccelerationChecks(_ecm, true);
    }
  }

  if (this->dataPtr->jointEntities.empty())
  {
    this->dataPtr->jointNames.clear();
    auto jointEntitiesModel = this->dataPtr->model.Joints(_ecm);
    std::vector<Joint> jointVec; jointVec.reserve(jointEntitiesModel.size());
    for (const auto &jointEnt: jointEntitiesModel)
    {
      Joint j(jointEnt);
      if (!j.Valid(_ecm)) continue;
      j.EnablePositionCheck(_ecm, true);
      j.EnableVelocityCheck(_ecm, true);
      jointVec.push_back(j);
    }
    if (jointVec.size() != jointEntitiesModel.size()) return;
    for (const auto &linkName : this->dataPtr->linkNames){
      for (const auto &joint: jointVec){
        if (auto linkNameTmp = joint.ChildLinkName(_ecm); linkNameTmp && *linkNameTmp == linkName)
        { this->dataPtr->jointEntities.push_back(joint.Entity()); this->dataPtr->jointNames.push_back(*joint.Name(_ecm)); break; }
        if (auto linkNameTmp = joint.ParentLinkName(_ecm); linkNameTmp && *linkNameTmp == linkName)
        { this->dataPtr->jointEntities.push_back(joint.Entity()); this->dataPtr->jointNames.push_back(*joint.Name(_ecm)); break; }
      }
    }
    if (this->dataPtr->jointNames.size() != this->dataPtr->linkNames.size()) return;
  }

  // Ensure components exist and bind pointers once; if we created any, skip this tick.
  if (!this->dataPtr->bound)
  {
    if (!this->dataPtr->BindOnce(_ecm))
      return; // components were created; bind on next frame
  }

  this->dataPtr->samplingTime = std::chrono::duration<double>(_info.dt).count();
  this->dataPtr->UpdateForcesAndMoments(_ecm);
}

bool MultiRotorPluginPrivate::BindOnce(EntityComponentManager &_ecm)
{
  bool created = false;

  auto ensure = [&](Entity e, auto tag){
    using T = std::decay_t<decltype(tag)>;
    if (!_ecm.Component<T>(e)) { _ecm.CreateComponent(e, T{}); created = true; }
  };

  // Per-joint comps
  for (const auto &jointEnt : this->jointEntities){
    ensure(jointEnt, components::JointVelocity());
    ensure(jointEnt, components::JointAxis());
    ensure(jointEnt, components::Pose());
  }
  // Per-rotor link comps
  for (const auto &linkEnt : this->linkEntities){
    ensure(linkEnt,  components::WorldPose());
    ensure(linkEnt,  components::WorldLinearVelocity());
    ensure(linkEnt,  components::ExternalWorldWrenchCmd());
  }
  // Parent link comps
  ensure(this->parentLinkEntity, components::WorldPose());
  ensure(this->parentLinkEntity, components::WorldAngularVelocity());
  ensure(this->parentLinkEntity, components::WorldLinearVelocity());
  ensure(this->parentLinkEntity, components::ExternalWorldWrenchCmd());

  if (created) return false; // wait a frame so components become visible

  // Cache pointers after ensuring existence
  for (int i=0; i<this->numRotors; ++i){
    const auto jointEnt = this->jointEntities[i];
    this->rotorDatas->jointPoseComps[i]  = _ecm.Component<components::Pose>(jointEnt);
    this->rotorDatas->jointAxisComps[i]  = _ecm.Component<components::JointAxis>(jointEnt);
    this->rotorDatas->jointVelComps[i]   = _ecm.Component<components::JointVelocity>(jointEnt);

    const auto linkEnt = this->linkEntities[i];
    this->rotorDatas->rotorWrenchComps[i] = _ecm.Component<components::ExternalWorldWrenchCmd>(linkEnt);
  }
  this->parentWrench = _ecm.Component<components::ExternalWorldWrenchCmd>(this->parentLinkEntity);

  // Validate pointers
  for (int i=0; i<this->numRotors; ++i){
    if (!this->rotorDatas->jointPoseComps[i] || !this->rotorDatas->jointAxisComps[i] ||
        !this->rotorDatas->jointVelComps[i]  || !this->rotorDatas->rotorWrenchComps[i])
      return false;
  }
  if (!this->parentWrench) return false;

  this->bound = true;
  return true;
}

void MultiRotorPluginPrivate::UpdateForcesAndMoments(
    EntityComponentManager &_ecm)
{
  // Actuators
  msgs::Actuators* msgPtr = nullptr;
  auto actuatorsComp = _ecm.Component<components::Actuators>(this->parentLinkEntity);
  // Disable getting cmd from Model, any control cmd should be attached to the 
  // link component
  // if (!actuatorsComp) actuatorsComp = _ecm.Component<components::Actuators>(this->model.Entity());
  if (actuatorsComp) msgPtr = &actuatorsComp->Data(); else return;

  auto &act = *msgPtr;
  if (act.velocity_size() < this->numRotors)
  { gzerr << "Not enough actuator commands, got " << act.velocity_size() << ", expected " << this->numRotors << "\n"; return; }

  for (int i = 0; i < this->numRotors; ++i)
  {
    const double v = static_cast<double>(act.velocity(i));
    this->refMotorInput[i] = std::isfinite(v) ? std::min(v, this->maxRotVelocity) : 0.0;
    *act.mutable_velocity()->Mutable(i) = 0.0; // consume
  }

  if (this->windEntity == kNullEntity)
    this->windEntity = _ecm.EntityByComponents(components::Wind());

  math::Vector3d windWorld{0,0,0};
  if (this->windEntity != kNullEntity)
    if (auto windLin = _ecm.Component<components::WorldLinearVelocity>(this->windEntity))
      windWorld = windLin->Data();

  // Parent state (shared)
  auto parentWorldPose = this->parentLinkCache.WorldPose(_ecm);
  if (!parentWorldPose) return;

  const auto worldAngVelComp = _ecm.Component<components::WorldAngularVelocity>(this->parentLinkEntity);
  const auto omegaBody = worldAngVelComp ? parentWorldPose->Rot().RotateVectorReverse(worldAngVelComp->Data()) : math::Vector3d();
  const auto parentLinVelComp = _ecm.Component<components::WorldLinearVelocity>(this->parentLinkEntity);

  math::Vector3d parentForceAccum = msgs::Convert(this->parentWrench->Data().force());
  math::Vector3d parentTorqueAccum = msgs::Convert(this->parentWrench->Data().torque());

  constexpr double kMinHeight = 0.05; // 5 cm

  // Pass 1: gather per-rotor state
  for (int i = 0; i < this->numRotors; ++i)
  {
    auto &rd = *this->rotorDatas;

    rd.rotorWorldPoses[i] = rd.rotorLinks[i].WorldPose(_ecm);
    if (!rd.rotorWorldPoses[i]) return;

    auto *poseC  = rd.jointPoseComps[i];
    auto *axisC  = rd.jointAxisComps[i];
    auto *velC   = rd.jointVelComps[i];

    rd.jointWorldPoses[i] = (*rd.rotorWorldPoses[i]) * poseC->Data();
    rd.jointAxisWorlds[i] = rd.jointWorldPoses[i].Rot().RotateVector(axisC->Data().Xyz());
    const double n = rd.jointAxisWorlds[i].Length();
    if (n > 1e-9) rd.jointAxisWorlds[i] = rd.jointAxisWorlds[i].Normalized();

    const auto rotorLinVel = *rd.rotorLinks[i].WorldLinearVelocity(_ecm);
    rd.relWindWorlds[i] = rotorLinVel - windWorld;
    rd.vPerps[i] = rd.relWindWorlds[i] - (rd.relWindWorlds[i].Dot(rd.jointAxisWorlds[i]) * rd.jointAxisWorlds[i]);

    rd.realMotorVelocities[i] = velC->Data()[0] * this->rotorVelocitySlowdownSim;
    rd.absMotorVels[i] = std::abs(rd.realMotorVelocities[i]);
    rd.heights[i] = std::max(kMinHeight, std::abs(rd.rotorWorldPoses[i]->Pos().Z()));
  }

#if MULTIROTOR_SIMD
  // Pass 2 (SIMD): thrust & ground effect
  {
    const double A = this->thrustConstantQuadraticParams.X();
    const double B = this->thrustConstantQuadraticParams.Y();
    const double C = this->thrustConstantQuadraticParams.Z();
    const double K = this->groundEffectConstant;
    MR_OMP_SIMD
    for (int i = 0; i < this->numRotors; ++i){
      const double w = this->rotorDatas->absMotorVels[i];
      const double h = this->rotorDatas->heights[i];
      const double T = ((A * w) * w) + (B * w) + C;
      this->rotorDatas->thrusts[i] = T;
      const double inv_h2 = 1.0 / (h*h);
      this->rotorDatas->groundEffects[i] = K * w* w * inv_h2;
    }
  }
#else
  // Fallback compute inside per-rotor loop (not shown)
#endif

  // Pass 3: apply link forces and accumulate parent torques
  for (int i = 0; i < this->numRotors; ++i)
  {
    auto &rd = *this->rotorDatas;

    rd.thrustWorlds[i] = rd.rotorWorldPoses[i]->Rot().RotateVector({0, 0, rd.thrusts[i] + rd.groundEffects[i]});
    rd.airDragRotors[i] = -rd.absMotorVels[i] * this->rotorDragCoefficient * rd.vPerps[i];
    rd.totalThrusts[i]  = rd.airDragRotors[i] + rd.thrustWorlds[i];

    auto *rotorWrench = rd.rotorWrenchComps[i];
    msgs::Set(rotorWrench->Data().mutable_force(), rd.totalThrusts[i]);

    rd.poseDiffs[i] = parentWorldPose->Inverse() * (*rd.rotorWorldPoses[i]);
    rd.dragTorqueRotors[i] = {0.0, 0.0, -this->turningDirections[i] * (rd.absMotorVels[i]*rd.absMotorVels[i] * this->momentConstantQuadraticParams.X() +
              rd.absMotorVels[i] * this->momentConstantQuadraticParams.Y()  + this->momentConstantQuadraticParams.Z())};
    rd.dragTorqueParents[i] = parentWorldPose->Rot().RotateVector(rd.poseDiffs[i].Rot().RotateVector(rd.dragTorqueRotors[i]));
    rd.rollingMoments[i] = -rd.absMotorVels[i] * this->rollingMomentCoefficient * rd.vPerps[i];

    rd.gyroWorlds[i].Set(0,0,0);
    if (worldAngVelComp)
    {
      rd.minusGs[i] = {-omegaBody.Y(), omegaBody.X(), 0.0};
      rd.gyroWorlds[i] = parentWorldPose->Rot().RotateVector(this->rotorInertia * rd.minusGs[i] * rd.realMotorVelocities[i]);
    }

    rd.parentWorldTorques[i] = rd.dragTorqueParents[i] + rd.rollingMoments[i] + rd.gyroWorlds[i];
    parentTorqueAccum += rd.parentWorldTorques[i];

    // if (parentLinVelComp)
    // {
    //   const auto &v =  parentLinVelComp->Data();
    //   rd.bodyDrags[i] = { std::clamp(v.X()*std::abs(v.X()) * this->airDragCoeffs.X(), -10.0, 10.0),
    //                       std::clamp(v.Y()*std::abs(v.Y()) * this->airDragCoeffs.Y(), -10.0, 10.0),
    //                       std::clamp(v.Z()*std::abs(v.Z()) * this->airDragCoeffs.Z(), -10.0, 10.0) };
    //   parentForceAccum -= rd.bodyDrags[i]/this->numRotors;
    // }
  }

#if MULTIROTOR_SIMD_FILTER
  // Pass 4 (SIMD): filter + velocity cmds
  {
    const double dt  = this->samplingTime;
    const double up  = this->timeConstantUp;
    const double down= this->timeConstantDown;
    const double inv = this->rotorVelocitySlowdownInv;
    MR_OMP_SIMD
    for (int i = 0; i < this->numRotors; ++i){
      const double input = this->refMotorInput[i];
      const double prev  = this->filterPrev[i];
      const double tau   = (input > prev) ? up : down;
      const double alpha = std::exp(-dt / std::max(tau, 1e-9));
      const double state = alpha * prev + (1.0 - alpha) * input;
      this->filterPrev[i] = state;
      double cmd = this->turningDirections[i] * state * inv;
      if (cmd > 10000.0) cmd = 10000.0; else if (cmd < -10000.0) cmd = -10000.0;
      this->cmds[i] = std::isfinite(cmd) ? cmd : 0.0;
    }
  }
  for (int i = 0; i < this->numRotors; ++i)
    _ecm.SetComponentData<components::JointVelocityCmd>(this->jointEntities[i], {this->cmds[i]});
#else
  // Fallback: per-rotor object filters
  for (int i = 0; i < this->numRotors; ++i){
    double cmd = this->rotorVelocityFilters[i]->UpdateFilter(this->refMotorInput[i], this->samplingTime);
    cmd = std::clamp(this->turningDirections[i] * cmd * this->rotorVelocitySlowdownInv, -10000.0, 10000.0);
    if (!std::isfinite(cmd)) cmd = 0.0;
    _ecm.SetComponentData<components::JointVelocityCmd>(this->jointEntities[i], {cmd});
  }
#endif

  // Pass 5: write parent wrench once
  msgs::Set(this->parentWrench->Data().mutable_force(),  parentForceAccum);
  msgs::Set(this->parentWrench->Data().mutable_torque(), parentTorqueAccum);

  // model downwash
  for (const auto &uavNames : this->otherUavNames){
    if (uavNames == this->model.Name(_ecm)) continue;
    auto model_entity_other = _ecm.EntityByName(uavNames);
    if (!model_entity_other) continue;
    const auto entity =  gz::sim::Model(*model_entity_other).CanonicalLink(_ecm);
    const auto worldPoseUav =  _ecm.Component<components::WorldPose>(entity);
    if (! worldPoseUav) continue;
    const auto poseDelta = worldPoseUav->Data().Pos() - parentWorldPose->Pos();
    // check if its below
    if (poseDelta.Z() > 0) continue;
    auto wrenchcmd = _ecm.Component<components::ExternalWorldWrenchCmd>(entity);
    if (!wrenchcmd) continue;
    math::Vector3d currForce = msgs::Convert(wrenchcmd->Data().force());
    currForce.Z() = currForce.Z() - this->downWashCoefficients.X()*std::exp(-0.5*std::pow((poseDelta.X()*poseDelta.X() + poseDelta.Y()*poseDelta.Y())/(-1.0*this->downWashCoefficients.Y()*poseDelta.Z() + this->downWashCoefficients.Z()), 2));
    msgs::Set(wrenchcmd->Data().mutable_force(), currForce);
  }
}

// ---------------------- Registration ----------------------
GZ_ADD_PLUGIN(gz::sim::systems::MultiRotorPlugin,
              System,
              MultiRotorPlugin::ISystemConfigure,
              MultiRotorPlugin::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(gz::sim::systems::MultiRotorPlugin,
                    "gz::sim::systems::MultiRotorPlugin")
