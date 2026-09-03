// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#include "rl_server.hh"
#include "print_utils.hh"
#include <gz/sim/components/ExternalWorldWrenchCmd.hh>
#include <gz/sim/components/JointForceCmd.hh>
#include <gz/sim/components/JointVelocityCmd.hh>
#include <gz/sim/components/JointVelocityReset.hh>
#include <gz/sim/components/JointPositionReset.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/JointVelocity.hh>
// std::atomic<int> DRLServer::instances_count = 0;
using namespace std::chrono_literals;
using namespace gz;
using namespace sim;

// -------------------- DRLHelperSystem --------------------
// ================= Implementation =================

using namespace std::chrono_literals;

// ---- DRLHelperSystem ----
DRLHelperSystem::DRLHelperSystem(DRLServer *server)
{
  this->server_ptr = server;
  this->model_names = server->model_names;

  // Load SDF from file for nested discovery + OU setup
  const auto opt_file = server->server_config->SdfFile();
  if (opt_file.empty()) [[unlikely]]
    throw std::runtime_error("SDF file not found, cannot initialize DRLHelperSystem");

  this->sdf_root.Load(opt_file);

  const auto world_ptr = this->sdf_root.WorldByIndex(0);
  if (!world_ptr) [[unlikely]]
    throw std::runtime_error("World not found, cannot initialize DRLHelperSystem");

  const size_t num_models = this->model_names.size();
  this->model_sdf_objs.reserve(num_models);

  // OPTIMIZED: Use stack allocation and direct initialization for OU parameters
  using P = OUParams<19>;
  using Vec = P::VecType;
  using Mat = P::MatrixType;

  // OPTIMIZED: Use constexpr for compile-time constant values
  constexpr double OU_POS_VAR = 5.98e-6;        // Position noise variance
  constexpr double OU_QUAT_VAR = 5.06e-7;       // Quaternion noise variance
  constexpr double OU_VEL_VAR = 1.6e-4;         // Velocity noise variance
  constexpr double OU_OMEGA_ACCEL_VAR = 9.8e-5; // Angular velocity + accelerations variance
  constexpr double OU_THETA_SLOW = 30.0;        // Mean reversion for position/orientation
  constexpr double OU_THETA_FAST = 30.0;        // Mean reversion for velocities

  Vec mu = Vec::Zero();
  Mat theta = Mat::Zero();
  Mat cov = Mat::Zero();

  // OPTIMIZED: Vectorized initialization via diagonal setters with constexpr values
  // State vector structure (19D, indices 0-18):
  //   [0-2]   Position (3)
  //   [3-6]   Quaternion (4)
  //   [7-9]   Velocity (3)
  //   [10-12] Angular velocity (3) - body frame
  //   [13-15] Acceleration (3)
  //   [16-18] Angular acceleration (3) - body frame
  cov.diagonal().segment<3>(0).setConstant(OU_POS_VAR);
  cov.diagonal().segment<4>(3).setConstant(OU_QUAT_VAR);
  cov.diagonal().segment<3>(7).setConstant(OU_VEL_VAR);
  cov.diagonal().segment<9>(10).setConstant(OU_OMEGA_ACCEL_VAR); // omega[10-12] + accel[13-15] + alpha[16-18]

  theta.diagonal().segment<3>(0).setConstant(OU_THETA_SLOW);
  theta.diagonal().segment<4>(3).setConstant(OU_THETA_SLOW);
  theta.diagonal().segment<3>(7).setConstant(OU_THETA_SLOW);
  theta.diagonal().segment<9>(10).setConstant(OU_THETA_FAST); // omega[10-12] + accel[13-15] + alpha[16-18]

  const P params(mu, theta, cov);
  step_size = world_ptr->PhysicsByIndex(0)->MaxStepSize();

  // OPTIMIZED: collect nested model names from SDF with reserved capacity
  std::vector<std::string> nested_model_names;
  nested_model_names.reserve(32); // Reserve more to avoid reallocations

  std::function<void(const sdf::Model *)> rec = [&](const sdf::Model *m)
  {
    if (m && m->ModelCount() > 0)
    {
      const uint64_t count = m->ModelCount();
      for (uint64_t i = 0; i < count; ++i)
      {
        const auto *n = m->ModelByIndex(i);
        if (!n) [[unlikely]]
          continue;
        nested_model_names.emplace_back(m->Name() + "::" + n->Name());
        rec(n);
      }
    }
  };

  for (const auto &name : this->model_names)
  {
    if (const auto *mp = world_ptr->ModelByName(name))
      rec(mp);
  }

  // OPTIMIZED: Use std::set for faster lookup instead of linear search
  const std::unordered_set<std::string> existing_names(this->model_names.begin(), this->model_names.end());
  for (const auto &nm : nested_model_names)
  {
    if (existing_names.find(nm) == existing_names.end())
    {
      this->model_names.push_back(nm);
    }
  }

  // OPTIMIZED: Batch process all models
  for (const auto &name : this->model_names)
  {
    const auto model_ptr = world_ptr->ModelByName(name);
    if (!model_ptr) [[unlikely]]
      throw std::runtime_error("Model not found in SDF: " + name);

    this->model_sdf_objs.emplace(name, *model_ptr);
    this->op_procs.emplace(name, std::make_unique<OrnsteinUhlenbeckMultivar<19>>(mu, params, step_size));
  }

  this->server_ptr->model_names = this->model_names;
  this->world_sdf_obj = *world_ptr;

  // OPTIMIZED: Reserve all containers at once
  const auto M = this->model_names.size();
  this->models.reserve(M);
  this->links.reserve(M);
  this->pose_datas.reserve(M);
  this->contacts.reserve(M);
  this->fast_link_slots.reserve(M);
}

DRLHelperSystem::~DRLHelperSystem() { this->sdf_creator.reset(); }

void DRLHelperSystem::Configure(
    const Entity &_entity,
    const std::shared_ptr<const sdf::Element> & /*_sdf*/,
    EntityComponentManager &_ecm,
    EventManager &_eventMgr)
{
  this->sdf_creator = std::make_unique<gz::sim::SdfEntityCreator>(_ecm, _eventMgr);
  this->main_parent_entity = _entity;
  this->init_method(_ecm);
  if (!this->has_ecm_init_state)
  {
    this->ecm_init_state = _ecm.State();
    this->has_ecm_init_state = true;
  }
  this->_configure_ecm = &_ecm; // cache for apply-now writes
}

template <typename Comp, typename T>
void SetOrCreate(EntityComponentManager &ecm, Entity e, const T &v)
{
  if (auto *c = ecm.Component<Comp>(e))
  {
    c->Data() = v;
    ecm.SetChanged(e, Comp::typeId, ComponentState::OneTimeChange); // <-- was missing
  }
  else
  {
    ecm.CreateComponent(e, Comp(v)); // CreateComponent already flags OneTimeChange
  }
}
void ResetUavCommands(EntityComponentManager &ecm, Entity modelEnt,
                      const math::Pose3d &pose, const std::unordered_map<std::string, sdf::Model> &model_sdf_objs)
{
  Model model(modelEnt);

  model.SetWorldPoseCmd(ecm, pose); // teleport
  SetOrCreate<components::LinearVelocityCmd>(ecm, modelEnt, math::Vector3d::Zero);
  SetOrCreate<components::AngularVelocityCmd>(ecm, modelEnt, math::Vector3d::Zero);

  auto it = model_sdf_objs.find(model.Name(ecm));
  if (it == model_sdf_objs.end()) [[unlikely]]
  {
    std::cerr << "SDF for model " << model.Name(ecm) << " not found during reset.\n";
  }
  const sdf::Model &model_sdf = it->second;

  for (uint64_t i = 0; i < model_sdf.LinkCount(); ++i)
  {
    const sdf::Link *link_sdf = model_sdf.LinkByIndex(i);
    if (!link_sdf)
      continue;

    // Pose relative to the model frame (handles //pose/@relative_to correctly).
    math::Pose3d link_rel_model;
    sdf::Errors errs = link_sdf->SemanticPose().Resolve(link_rel_model, "__model__");
    if (!errs.empty()) // graph not resolvable -> raw value
      link_rel_model = link_sdf->RawPose();

    Entity link_ent = model.LinkByName(ecm, link_sdf->Name());
    if (link_ent == kNullEntity)
      continue;

    SetOrCreate<components::Pose>(ecm, link_ent, link_rel_model);               // rel-to-model
    if (ecm.Component<components::WorldPose>(link_ent))                         // only if tracked
      SetOrCreate<components::WorldPose>(ecm, link_ent, pose * link_rel_model); // world

    ecm.RemoveComponent<components::ExternalWorldWrenchCmd>(link_ent);
    SetOrCreate<components::ExternalWorldWrenchCmd>(ecm, link_ent, msgs::Wrench());
  }

  // for (Entity link : model.Links(ecm)) {                   // drop residual wrench
  //   ecm.RemoveComponent<components::ExternalWorldWrenchCmd>(link);
  //   SetOrCreate<components::ExternalWorldWrenchCmd >(ecm, link, gz::msgs::Wrench());
  // }
  for (Entity joint : model.Joints(ecm))
  {
    ecm.RemoveComponent<components::JointForceCmd>(joint);
    ecm.RemoveComponent<components::JointVelocityCmd>(joint);

    // Real DOF, straight from the physics-populated state (1 revolute, 2 universal,
    // 3 ball). Avoids the empty-vector no-op and works for any joint type.
    std::size_t dof = 1;
    if (auto *jp = ecm.Component<components::JointPosition>(joint))
      dof = jp->Data().size();
    if (dof == 0)
      continue; // fixed joint, nothing to reset

    const std::vector<double> zeros(dof, 0.0);

    SetOrCreate<components::JointPositionReset>(ecm, joint, zeros);
    SetOrCreate<components::JointVelocityReset>(ecm, joint, zeros);
    // Also write the state components so anything reading before the next physics
    // step (and the engine itself, in some versions) sees the reset configuration.
    SetOrCreate<components::JointPosition>(ecm, joint, zeros);
    SetOrCreate<components::JointVelocity>(ecm, joint, zeros);
  }

  // for (Entity joint : model.Joints(ecm)) {                // stop + reset rotors
  //   ecm.RemoveComponent<components::JointForceCmd>(joint);
  //   ecm.RemoveComponent<components::JointVelocityCmd>(joint);
  //   SetOrCreate<components::JointPositionReset>(ecm, joint, std::vector<double>{0.0});
  //   SetOrCreate<components::JointVelocityReset>(ecm, joint, std::vector<double>{0.0});
  // }
}
void ReleaseVelocityCmds(EntityComponentManager &ecm, Entity modelEnt)
{
  ecm.RemoveComponent<components::LinearVelocityCmd>(modelEnt);
  ecm.RemoveComponent<components::AngularVelocityCmd>(modelEnt);
}

void DRLHelperSystem::PreUpdate(const UpdateInfo &_info, EntityComponentManager &_ecm)
{
  if (_info.paused) [[unlikely]]
    return;

  // --- OPTIMIZED: Batch processing of resets ---
  if (!this->reset_queue.empty()) [[unlikely]]
  {
    // OPTIMIZED: Reserve space in post_reset_queue to avoid reallocations
    this->post_reset_queue.reserve(this->post_reset_queue.size() + this->reset_queue.size());

    for (auto &reset_elem : this->reset_queue)
    {
      this->post_reset_queue.emplace_back(std::move(reset_elem)); // OPTIMIZED: Move instead of copy
      const auto &name = this->post_reset_queue.back().first;

      if (const auto entityOpt = _ecm.EntityByName(name)) [[likely]]
      {
        // this->sdf_creator->RequestRemoveEntity(*entityOpt, true);
        const auto &po = this->post_reset_queue.back().second;
        constexpr size_t X_IDX = 0, Y_IDX = 1, Z_IDX = 2;
        constexpr size_t ROLL_IDX = 3, PITCH_IDX = 4, YAW_IDX = 5;
        // OPTIMIZED: Construct pose directly without temporary
        const gz::math::Pose3d pose(po(X_IDX), po(Y_IDX), po(Z_IDX),
                                    po(ROLL_IDX), po(PITCH_IDX), po(YAW_IDX));
        ResetUavCommands(_ecm, *entityOpt, pose, this->model_sdf_objs);
      }
      else
      {
        std::cerr << "Model " << name << " not moved for reset.\n";
      }
    }
    this->reset_queue.clear();
    return; // process respawn next iteration
  }

  if (!this->post_reset_queue.empty()) [[unlikely]]
  {
    // OPTIMIZED: Use constexpr for pose indices
    constexpr size_t X_IDX = 0, Y_IDX = 1, Z_IDX = 2;
    constexpr size_t ROLL_IDX = 3, PITCH_IDX = 4, YAW_IDX = 5;

    for (auto &reset_elem : this->post_reset_queue)
    {
      const auto &name = reset_elem.first;
      const auto &po = reset_elem.second;
      if (const auto entityOpt = _ecm.EntityByName(name)) [[likely]]
      {
        // this->sdf_creator->RequestRemoveEntity(*entityOpt, true);
        ReleaseVelocityCmds(_ecm, *entityOpt);
      }
      else
      {
        std::cerr << "Model " << name << " not removed for reset.\n";
      }
    }
    this->post_reset_queue.clear();
    return;
  }
  // Respawn helper
  if (!this->respawn_queue.empty()) [[unlikely]]
  {
    // OPTIMIZED: Reserve space in post_respawn_queue to avoid reallocations
    this->post_respawn_queue.reserve(this->post_respawn_queue.size() + this->respawn_queue.size());

    for (auto &reset_elem : this->respawn_queue)
    {
      this->post_respawn_queue.emplace_back(std::move(reset_elem)); // OPTIMIZED: Move instead of copy
      const auto &name = this->post_respawn_queue.back().first;

      if (const auto entityOpt = _ecm.EntityByName(name)) [[likely]]
      {
        this->sdf_creator->RequestRemoveEntity(*entityOpt, true);
      }
      else
      {
        std::cerr << "Model " << name << " not removed for reset.\n";
      }
    }
    this->respawn_queue.clear();
    return; // process respawn next iteration
  }

  if (!this->post_respawn_queue.empty()) [[unlikely]]
  {
    // OPTIMIZED: Use constexpr for pose indices
    constexpr size_t X_IDX = 0, Y_IDX = 1, Z_IDX = 2;
    constexpr size_t ROLL_IDX = 3, PITCH_IDX = 4, YAW_IDX = 5;

    for (auto &reset_elem : this->post_respawn_queue)
    {
      const auto &name = reset_elem.first;
      const auto &po = reset_elem.second;

      // OPTIMIZED: Construct pose directly without temporary
      const gz::math::Pose3d pose(po(X_IDX), po(Y_IDX), po(Z_IDX),
                                  po(ROLL_IDX), po(PITCH_IDX), po(YAW_IDX));

      auto it = this->model_sdf_objs.find(name);
      if (it == this->model_sdf_objs.end()) [[unlikely]]
      {
        std::cerr << "SDF for model " << name << " not found during reset.\n";
        continue;
      }

      it->second.SetRawPose(pose);
      const auto new_model_entity = this->sdf_creator->CreateEntities(&it->second);
      this->sdf_creator->SetParent(new_model_entity, this->main_parent_entity);

      if (const auto entityOpt = _ecm.EntityByName(name)) [[likely]]
      {
        Model(*entityOpt).SetWorldPoseCmd(_ecm, pose);
      }
      else
      {
        std::cerr << "Requested entity " << name << " cannot be found after respawn.\n";
      }
    }
    this->post_respawn_queue.clear();
    this->init_method(_ecm); // re-index caches
    return;
  }
}

void DRLHelperSystem::Update(const UpdateInfo & /*_info*/, EntityComponentManager & /*_ecm*/) {}

void DRLHelperSystem::PostUpdate(const UpdateInfo &_info, const EntityComponentManager &_ecm)
{
  if (_info.paused) [[unlikely]]
    return;

  // OPTIMIZED: Use constexpr for state vector offsets
  constexpr size_t POS_X = 0, POS_Y = 1, POS_Z = 2;
  constexpr size_t QUAT_W = 3, QUAT_X = 4, QUAT_Y = 5, QUAT_Z = 6;
  constexpr size_t VEL_X = 7, VEL_Y = 8, VEL_Z = 9;
  constexpr size_t OMEGA_X = 10, OMEGA_Y = 11, OMEGA_Z = 12;
  constexpr size_t ACCEL_X = 13, ACCEL_Y = 14, ACCEL_Z = 15;
  constexpr size_t ALPHA_X = 16, ALPHA_Y = 17, ALPHA_Z = 18;

  // OPTIMIZED: Process all models in batch with reduced checks
  for (auto &kv : this->fast_link_slots)
  {
    // OPTIMIZED: Calculate noise once per model, not per slot
    const auto noise_vec = this->op_procs.at(kv.first)->step();

    // OPTIMIZED: Use size_t for loop counter, hint likely path
    auto &slots = kv.second;
    const size_t num_slots = slots.size();

    for (size_t i = 0; i < num_slots; ++i)
    {
      auto &slot = slots[i];

      // OPTIMIZED: Fetch all components in one go, use if constexpr for compile-time branches
      const auto pose_opt = slot.link.WorldPose(_ecm);
      const auto vel_opt = slot.link.WorldLinearVelocity(_ecm);
      const auto omega_opt = slot.link.WorldAngularVelocity(_ecm);
      const auto alpha_opt = slot.link.WorldAngularAcceleration(_ecm);
      const auto accel_opt = slot.link.WorldLinearAcceleration(_ecm);

      // OPTIMIZED: Single early exit check - branch prediction friendly
      if (!pose_opt || !vel_opt || !omega_opt || !alpha_opt || !accel_opt) [[unlikely]]
        continue;

      // OPTIMIZED: Direct references to avoid repeated dereferences
      const auto &pose = *pose_opt;
      const auto &pos = pose.Pos();
      const auto &rot = pose.Rot();
      const auto &v = *vel_opt;
      const auto &w = *omega_opt;
      const auto &a = *accel_opt;
      const auto &alp = *alpha_opt;

      // OPTIMIZED: Inline rotation computation - avoid function call overhead
      const auto omega_body = rot.RotateVectorReverse(w);
      const auto alpha_body = rot.RotateVectorReverse(alp);

      // OPTIMIZED: Direct array access for state vector population with constexpr indices
      auto &dst = *slot.state;
      dst.coeffRef(POS_X) = pos.X();
      dst.coeffRef(POS_Y) = pos.Y();
      dst.coeffRef(POS_Z) = pos.Z();
      dst.coeffRef(QUAT_W) = rot.W();
      dst.coeffRef(QUAT_X) = rot.X();
      dst.coeffRef(QUAT_Y) = rot.Y();
      dst.coeffRef(QUAT_Z) = rot.Z();
      dst.coeffRef(VEL_X) = v.X();
      dst.coeffRef(VEL_Y) = v.Y();
      dst.coeffRef(VEL_Z) = v.Z();
      dst.coeffRef(OMEGA_X) = omega_body.X();
      dst.coeffRef(OMEGA_Y) = omega_body.Y();
      dst.coeffRef(OMEGA_Z) = omega_body.Z();
      dst.coeffRef(ACCEL_X) = a.X();
      dst.coeffRef(ACCEL_Y) = a.Y();
      dst.coeffRef(ACCEL_Z) = a.Z();
      dst.coeffRef(ALPHA_X) = alpha_body.X();
      dst.coeffRef(ALPHA_Y) = alpha_body.Y();
      dst.coeffRef(ALPHA_Z) = alpha_body.Z();
      dst.noalias() += noise_vec; // OPTIMIZED: Use noalias for faster addition

      // OPTIMIZED: Only process contacts if we have any collisions
      auto &cdst = *slot.contacts;
      cdst.clear();
      const auto collisions = slot.link.Collisions(_ecm);
      if (!collisions.empty()) [[likely]]
      {
        cdst.reserve(collisions.size()); // OPTIMIZED: Pre-allocate to avoid reallocs
        for (const auto &collision : collisions)
        {
          if (const auto contact_msg_comp = _ecm.Component<components::ContactSensorData>(collision)) [[likely]]
          {
            cdst.push_back(contact_msg_comp->Data());
          }
        }
      }
    }
  }
}

void DRLHelperSystem::request_contact_data(std::string_view model_name)
{
  const auto it = this->fast_link_slots.find(std::string(model_name));
  if (it == this->fast_link_slots.end()) [[unlikely]]
    return;

  const auto &slot_vec = it->second;
  auto &_ecm = *this->_configure_ecm;

  for (const auto &slot : slot_vec)
  {
    auto &cdst = *slot.contacts;
    cdst.clear();
    const auto collisions = slot.link.Collisions(_ecm);
    for (const auto &collision : collisions)
    {
      _ecm.CreateComponent(collision, components::ContactSensorData());
    }
  }
}

std::unordered_map<std::string, GZ_state> DRLHelperSystem::state_info(std::string_view model_name)
{
  std::unordered_map<std::string, GZ_state> out;
  const auto it = this->pose_datas.find(std::string(model_name));
  if (it == this->pose_datas.end()) [[unlikely]]
    return out;
  out.insert(it->second.begin(), it->second.end());
  return out;
}

bool DRLHelperSystem::copy_state_info_fast(std::string_view model_name, StateMap &out) const
{
  const auto it = this->pose_datas.find(std::string(model_name));
  if (it == this->pose_datas.end()) [[unlikely]]
  {
    out.clear();
    return false;
  }
  out = it->second; // copy
  return true;
}

bool DRLHelperSystem::for_each_state_fast(std::string_view model_name,
                                          const std::function<void(const std::string &, const GZ_state &)> &fn) const
{
  const auto it = this->pose_datas.find(std::string(model_name));
  if (it == this->pose_datas.end()) [[unlikely]]
    return false;
  for (const auto &p : it->second)
    fn(p.first, p.second);
  return true;
}

void DRLHelperSystem::reset_pos(std::string model_name, Eigen::Vector3d &position, Eigen::Vector3d &orientation)
{
  // OPTIMIZED: Emplace directly to avoid extra allocations
  this->reset_queue.emplace_back(
      std::move(model_name),
      (Eigen::Matrix<double, 6, 1>() << position(0), position(1), position(2),
       orientation(0), orientation(1), orientation(2))
          .finished());
}
void DRLHelperSystem::respawn_model(std::string model_name, Eigen::Vector3d &position, Eigen::Vector3d &orientation)
{
  // OPTIMIZED: Emplace directly to avoid extra allocations
  this->respawn_queue.emplace_back(
      std::move(model_name),
      (Eigen::Matrix<double, 6, 1>() << position(0), position(1), position(2),
       orientation(0), orientation(1), orientation(2))
          .finished());
}

// ---- Apply-now helpers ----

void DRLHelperSystem::set_rotor_velocity_cmd(std::string model_name, std::string link_name, Eigen::VectorXd &cmd)
{
  if (!this->_configure_ecm) [[unlikely]]
    return;

  const auto mit = this->models.find(model_name);
  if (mit == this->models.end()) [[unlikely]]
    return;

  const auto lit = this->links.at(model_name).find(link_name);
  if (lit == this->links.at(model_name).end()) [[unlikely]]
    return;

  const auto &link = lit->second;
  const auto link_entity = link.Entity();

  // OPTIMIZED: Resize once and use direct data() pointer access
  const int cmd_size = static_cast<int>(cmd.size());
  this->actuator_cmd_msg.mutable_velocity()->Resize(cmd_size, 0.0);
  double *vel_data = this->actuator_cmd_msg.mutable_velocity()->mutable_data();

  // OPTIMIZED: Memcpy-like loop - compiler can vectorize this
  for (int i = 0; i < cmd_size; ++i)
  {
    vel_data[i] = cmd(i);
  }

  // OPTIMIZED: Check component existence first to avoid redundant SetData
  auto comp = this->_configure_ecm->Component<components::Actuators>(link_entity);
  if (comp) [[likely]]
  {
    const auto state = comp->SetData(this->actuator_cmd_msg, EqualActuatorMsgs)
                           ? ComponentState::PeriodicChange
                           : ComponentState::NoChange;
    this->_configure_ecm->SetChanged(link_entity, components::Actuators::typeId, state);
  }
  else
  {
    this->_configure_ecm->CreateComponent(link_entity, components::Actuators(this->actuator_cmd_msg));
  }
}

void DRLHelperSystem::set_rotor_velocity_cmd(std::string model_name, std::string link_name, Eigen::VectorXd &&cmd)
{
  this->set_rotor_velocity_cmd(std::move(model_name), std::move(link_name), cmd);
}

void DRLHelperSystem::set_srt_cmd(std::string model_name, std::string base_link,
                                  const std::vector<std::string> &link_names, const std::vector<int> &turning_dir, Eigen::VectorXd &cmd, double ktau)
{
  if (!this->_configure_ecm) [[unlikely]]
    return;

  const auto mit = this->models.find(model_name);
  if (mit == this->models.end()) [[unlikely]]
    return;
  math::Vector3d moments;
  const auto blit = this->links.at(model_name).find(base_link);
  if (blit == this->links.at(model_name).end()) [[unlikely]]
    return;

  for (size_t i = 0; i < link_names.size(); ++i)
  {
    const auto &link_name = link_names[i];
    const auto lit = this->links.at(model_name).find(link_name);
    if (lit == this->links.at(model_name).end()) [[unlikely]]
      return;
    const auto pose = lit->second.WorldPose(*this->_configure_ecm);
    if (!pose) [[unlikely]]
      return;
    moments += pose->Rot().RotateVector({0.0, 0.0, -turning_dir[i] * ktau * cmd(i)});
    // OPTIMIZED: Direct construction without intermediate variables
    lit->second.AddWorldWrench(*this->_configure_ecm,
                               pose->Rot().RotateVector({0.0, 0.0, cmd(i)}),
                               {0.0, 0.0, 0.0});
  }
  blit->second.AddWorldWrench(*this->_configure_ecm, {0.0, 0.0, 0.0}, moments);
}
void DRLHelperSystem::set_srt_cmd(std::string model_name, std::string base_link, const std::vector<std::string> &link_names,
                                  const std::vector<int> &turning_dir, Eigen::VectorXd &&cmd, double ktau)
{
  set_srt_cmd(std::move(model_name), std::move(base_link), link_names, turning_dir, cmd, ktau);
}

void DRLHelperSystem::set_velocity_cmd(std::string model_name, std::string link_name, Eigen::Vector3d &cmd)
{
  if (!this->_configure_ecm) [[unlikely]]
    return;

  const auto mit = this->models.find(model_name);
  if (mit == this->models.end()) [[unlikely]]
    return;

  const auto lit = this->links.at(model_name).find(link_name);
  if (lit == this->links.at(model_name).end()) [[unlikely]]
    return;

  const auto &link = lit->second;
  const auto link_entity = link.Entity();
  const math::Vector3d v{cmd(0), cmd(1), cmd(2)};

  auto comp = this->_configure_ecm->Component<components::LinearVelocityCmd>(link_entity);
  if (comp) [[likely]]
  {
    const auto state = comp->SetData(v, EqualVectors3)
                           ? ComponentState::PeriodicChange
                           : ComponentState::NoChange;
    this->_configure_ecm->SetChanged(link_entity, components::LinearVelocityCmd::typeId, state);
  }
  else
  {
    this->_configure_ecm->CreateComponent(link_entity, components::LinearVelocityCmd(v));
  }
}

void DRLHelperSystem::set_velocity_cmd(std::string model_name, std::string link_name, Eigen::Vector3d &&cmd)
{
  this->set_velocity_cmd(std::move(model_name), std::move(link_name), cmd);
}

void DRLHelperSystem::set_angular_velocity_cmd(std::string model_name, std::string link_name, Eigen::Vector3d &cmd)
{
  if (!this->_configure_ecm) [[unlikely]]
    return;

  const auto mit = this->models.find(model_name);
  if (mit == this->models.end()) [[unlikely]]
    return;

  const auto lit = this->links.at(model_name).find(link_name);
  if (lit == this->links.at(model_name).end()) [[unlikely]]
    return;

  const auto &link = lit->second;
  const auto link_entity = link.Entity();
  const math::Vector3d w{cmd(0), cmd(1), cmd(2)};

  auto comp = this->_configure_ecm->Component<components::AngularVelocityCmd>(link_entity);
  if (comp) [[likely]]
  {
    const auto state = comp->SetData(w, EqualVectors3)
                           ? ComponentState::PeriodicChange
                           : ComponentState::NoChange;
    this->_configure_ecm->SetChanged(link_entity, components::AngularVelocityCmd::typeId, state);
  }
  else
  {
    this->_configure_ecm->CreateComponent(link_entity, components::AngularVelocityCmd(w));
  }
}

void DRLHelperSystem::set_angular_velocity_cmd(std::string model_name, std::string link_name, Eigen::Vector3d &&cmd)
{
  this->set_angular_velocity_cmd(std::move(model_name), std::move(link_name), cmd);
}

void DRLHelperSystem::set_ackermann_velocity_cmd(std::string model_name, std::string link_name, Eigen::Vector2d &cmd)
{
  if (!this->_configure_ecm) [[unlikely]]
    return;

  const auto mit = this->models.find(model_name);
  if (mit == this->models.end()) [[unlikely]]
    return;

  const auto lit = this->links.at(model_name).find(link_name);
  if (lit == this->links.at(model_name).end()) [[unlikely]]
    return;

  const auto &link = lit->second;
  const auto link_entity = link.Entity();
  const math::Vector2d v{cmd(0), cmd(1)};

  auto comp = this->_configure_ecm->Component<AckerMannVelocityCmdComp>(link_entity);
  if (comp) [[likely]]
  {
    const auto state = comp->SetData(v, EqualVectors2)
                           ? ComponentState::PeriodicChange
                           : ComponentState::NoChange;
    this->_configure_ecm->SetChanged(link_entity, AckerMannVelocityCmdComp::typeId, state);
  }
  else
  {
    this->_configure_ecm->CreateComponent(link_entity, AckerMannVelocityCmdComp(v));
  }
}

void DRLHelperSystem::set_ackermann_velocity_cmd(std::string model_name, std::string link_name, Eigen::Vector2d &&cmd)
{
  this->set_ackermann_velocity_cmd(std::move(model_name), std::move(link_name), cmd);
}

void DRLHelperSystem::set_joint_position_cmd(std::string model_name, std::string joint_name, Eigen::Vector3d &&cmd)
{
  if (!this->_configure_ecm) [[unlikely]]
    return;

  const auto mit = this->models.find(model_name);
  if (mit == this->models.end()) [[unlikely]]
    return;

  const auto joint = mit->second.JointByName(*this->_configure_ecm, joint_name);
  if (!joint) [[unlikely]]
    return;

  // OPTIMIZED: Use initializer list construction
  const std::vector<double> v{cmd(0), cmd(1), cmd(2)};

  auto comp = this->_configure_ecm->Component<JointPositionCmdComp>(joint);
  if (comp) [[likely]]
  {
    const auto state = comp->SetData(v, EqualStdVec)
                           ? ComponentState::PeriodicChange
                           : ComponentState::NoChange;
    this->_configure_ecm->SetChanged(joint, JointPositionCmdComp::typeId, state);
  }
  else
  {
    this->_configure_ecm->CreateComponent(joint, JointPositionCmdComp(v));
  }
}

void DRLHelperSystem::set_joint_position_cmd(std::string model_name, std::string joint_name, Eigen::Vector3d &cmd)
{
  this->set_joint_position_cmd(std::move(model_name), std::move(joint_name), std::move(cmd));
}

void DRLHelperSystem::set_mass(std::string model_name, std::string link_name, double mass)
{
  auto it = this->model_sdf_objs.find(model_name);
  if (it == this->model_sdf_objs.end())
    return;
  auto link = it->second.LinkByName(link_name);
  if (!link)
    return;
  const gz::math::Inertiald &orig = link->Inertial();
  gz::math::MassMatrix3 new_mass_mat(mass, orig.MassMatrix().DiagonalMoments(), orig.MassMatrix().OffDiagonalMoments());
  gz::math::Inertiald new_inertial(new_mass_mat, orig.Pose());
  link->SetInertial(new_inertial);
}
void DRLHelperSystem::set_inertia(std::string model_name, std::string link_name, Eigen::Matrix3d &inertia)
{
  auto it = this->model_sdf_objs.find(model_name);
  if (it == this->model_sdf_objs.end())
    return;
  auto link = it->second.LinkByName(link_name);
  if (!link)
    return;
  const gz::math::Inertiald &orig = link->Inertial();
  // Inertia matrix is expressed in ineertial frame,
  // however we expect the input to be in link frame
  // therefore we use the tensor transformation formula
  // I_b_d = R * I_i_d * R^T
  // Therefore, I_id = R^T * I_b_d * R
  const auto &rot = gz::math::Matrix3d(orig.Pose().Rot());
  gz::math::Matrix3d gz_inertia;
  gz_inertia(0, 0) = inertia(0, 0);
  gz_inertia(0, 1) = inertia(0, 1);
  gz_inertia(0, 2) = inertia(0, 2);
  gz_inertia(1, 0) = inertia(1, 0);
  gz_inertia(1, 1) = inertia(1, 1);
  gz_inertia(1, 2) = inertia(1, 2);
  gz_inertia(2, 0) = inertia(2, 0);
  gz_inertia(2, 1) = inertia(2, 1);
  gz_inertia(2, 2) = inertia(2, 2);
  gz_inertia = rot.Transposed() * gz_inertia * rot;
  gz::math::Vector3d diagonal{gz_inertia(0, 0), gz_inertia(1, 1), gz_inertia(2, 2)};
  gz::math::Vector3d off_diag{gz_inertia(0, 1), gz_inertia(0, 2), gz_inertia(1, 2)};
  gz::math::MassMatrix3 new_mass_mat(orig.MassMatrix().Mass(), diagonal, off_diag);
  gz::math::Inertiald new_inertial(new_mass_mat, orig.Pose());
  link->SetInertial(new_inertial);
}
void DRLHelperSystem::set_inertia(std::string model_name, std::string link_name, Eigen::Matrix3d &&inertia)
{
  this->set_inertia(std::move(model_name), std::move(link_name), inertia);
}

Eigen::Matrix3d DRLHelperSystem::get_inertia(std::string model_name, std::string link_name)
{
  auto it = this->model_sdf_objs.find(model_name);
  if (it == this->model_sdf_objs.end())
    throw std::runtime_error("Model sdf object not found.");
  auto link = it->second.LinkByName(link_name);
  if (!link)
    throw std::runtime_error("Model sdf object not found.");
  const gz::math::Inertiald &orig = link->Inertial();
  Eigen::Matrix3d inertia;
  auto moi = orig.Moi();
  inertia << moi(0, 0), moi(0, 1), moi(0, 2),
      moi(1, 0), moi(1, 1), moi(1, 2),
      moi(2, 0), moi(2, 1), moi(2, 2);
  return inertia;
}

double DRLHelperSystem::get_mass(std::string model_name, std::string link_name)
{
  auto it = this->model_sdf_objs.find(model_name);
  if (it == this->model_sdf_objs.end())
    throw std::runtime_error("Model sdf object not found.");
  auto link = it->second.LinkByName(link_name);
  if (!link)
    throw std::runtime_error("Model sdf object not found.");
  const gz::math::Inertiald &orig = link->Inertial();
  return orig.MassMatrix().Mass();
}
void DRLHelperSystem::set_rotor_parameters(std::string model_name, const RotorParameters &params)
{
  auto it = this->model_sdf_objs.find(model_name);
  if (it == this->model_sdf_objs.end())
    return;
  // The rotor plugin could be attached to the link or
  // the model, however, all of them will be aggregated under
  // model sdf object
  // Go through model plugins
  sdf::Plugins &plugins = it->second.Plugins();
  sdf::ElementPtr rotor_plugin_elem = nullptr;
  sdf::Plugin plugin;
  size_t idx = 0;
  for (size_t i = 0; i < plugins.size(); ++i)
  {

    if (plugins[i].Name() == "gz::sim::systems::MultiRotorPlugin")
    {
      plugin = plugins[i];
      rotor_plugin_elem = plugin.Element();
      idx = i;
      break;
    }
  }
  if (!rotor_plugin_elem)
  {
    print_err("Rotor plugin not found in model plugins");
    return;
  }
  // Set parameters
  plugin.ClearContents();
  rotor_plugin_elem->GetElement("maxRotVelocity")->Set(params.max_rot_velocity);
  plugin.InsertContent(rotor_plugin_elem->GetElement("maxRotVelocity"));
  // rotor_plugin_elem->GetElement("airDragCoeffs")->Set(
  //   gz::math::Vector3d({params.air_drag_coeffs(0), params.air_drag_coeffs(1), params.air_drag_coeffs(2)}));
  rotor_plugin_elem->GetElement("thrustConstantQuadraticParams")->Set(gz::math::Vector3d({params.thrust_constant_quadratic_params(0), params.thrust_constant_quadratic_params(1), params.thrust_constant_quadratic_params(2)}));
  plugin.InsertContent(rotor_plugin_elem->GetElement("thrustConstantQuadraticParams"));
  rotor_plugin_elem->GetElement("momentConstantQuadraticParams")->Set(gz::math::Vector3d({params.torque_constant_quadratic_params(0), params.torque_constant_quadratic_params(1), params.torque_constant_quadratic_params(2)}));
  plugin.InsertContent(rotor_plugin_elem->GetElement("momentConstantQuadraticParams"));
  rotor_plugin_elem->GetElement("groundEffectConstant")->Set(params.ground_effect_constant);
  plugin.InsertContent(rotor_plugin_elem->GetElement("groundEffectConstant"));
  rotor_plugin_elem->GetElement("timeConstantUp")->Set(params.time_constant_up);
  plugin.InsertContent(rotor_plugin_elem->GetElement("timeConstantUp"));
  rotor_plugin_elem->GetElement("timeConstantDown")->Set(params.time_constant_down);
  plugin.InsertContent(rotor_plugin_elem->GetElement("timeConstantDown"));
  rotor_plugin_elem->GetElement("rotorDragCoefficient")->Set(params.rotor_drag_coefficient);
  plugin.InsertContent(rotor_plugin_elem->GetElement("rotorDragCoefficient"));
  rotor_plugin_elem->GetElement("rotorInertia")->Set(params.rotor_inertia);
  plugin.InsertContent(rotor_plugin_elem->GetElement("rotorInertia"));
  rotor_plugin_elem->GetElement("rollingMomentCoefficient")->Set(params.rolling_moment_coefficient);
  plugin.InsertContent(rotor_plugin_elem->GetElement("rollingMomentCoefficient"));
  // insert everything back again, but since we already overrode rotor params, re-writing them
  // wont affect, since elements are read first-come-first-go
  for (const auto elem : plugins[idx].Contents())
  {
    plugin.InsertContent(elem);
  }
  // Update the plugin back to the model sdf object
  // plugin.ClearContents();

  plugins[idx] = plugin;
  print_info("Rotor parameter updates will take place after next respawn. Ensure to call DRLServer.respawn(model_name, pos, ori).");
}

RotorParameters DRLHelperSystem::get_rotor_parameters(std::string model_name)
{
  auto it = this->model_sdf_objs.find(model_name);
  if (it == this->model_sdf_objs.end())
    throw std::runtime_error("Model sdf object not found.");
  // The rotor plugin could be attached to the link or
  // the model, however, all of them will be aggregated under
  // model sdf object
  // Go through model plugins
  sdf::Plugins &plugins = it->second.Plugins();
  sdf::ElementPtr rotor_plugin_elem = nullptr;
  size_t idx = 0;
  for (size_t i = 0; i < plugins.size(); ++i)
  {
    sdf::Plugin plugin = plugins[i];
    if (plugin.Name() == "gz::sim::systems::MultiRotorPlugin")
    {
      rotor_plugin_elem = plugin.Element();
      idx = i;
      break;
    }
  }
  if (!rotor_plugin_elem)
  {
    print_err("Rotor plugin not found in model plugins");
    throw std::runtime_error("");
  }
  RotorParameters params;
  // Get parameters
  params.max_rot_velocity = rotor_plugin_elem->GetElement("maxRotVelocity")->Get<double>();
  // auto air_drag_coeffs = rotor_plugin_elem->GetElement("airDragCoeffs")->Get<gz::math::Vector3d>();
  // params.air_drag_coeffs = {air_drag_coeffs.X(), air_drag_coeffs.Y(), air_drag_coeffs.Z()};
  auto thrust_constant_quadratic_params = rotor_plugin_elem->GetElement("thrustConstantQuadraticParams")->Get<gz::math::Vector3d>();
  params.thrust_constant_quadratic_params = {thrust_constant_quadratic_params.X(), thrust_constant_quadratic_params.Y(), thrust_constant_quadratic_params.Z()};
  auto torque_constant_quadratic_params = rotor_plugin_elem->GetElement("momentConstantQuadraticParams")->Get<gz::math::Vector3d>();
  params.torque_constant_quadratic_params = {torque_constant_quadratic_params.X(), torque_constant_quadratic_params.Y(), torque_constant_quadratic_params.Z()};
  params.ground_effect_constant = rotor_plugin_elem->GetElement("groundEffectConstant")->Get<double>();
  params.time_constant_up = rotor_plugin_elem->GetElement("timeConstantUp")->Get<double>();
  params.time_constant_down = rotor_plugin_elem->GetElement("timeConstantDown")->Get<double>();
  params.rotor_drag_coefficient = rotor_plugin_elem->GetElement("rotorDragCoefficient")->Get<double>();
  params.rotor_inertia = rotor_plugin_elem->GetElement("rotorInertia")->Get<double>();
  params.rolling_moment_coefficient = rotor_plugin_elem->GetElement("rollingMomentCoefficient")->Get<double>();

  return params;
}

Eigen::MatrixXd DRLHelperSystem::get_rotor_thrust_allocation_matrix(std::string model_name)
{
  // OPTIMIZED: Use constexpr for known dimensions
  constexpr size_t CONTROL_INPUTS = 4; // thrust, roll_moment, pitch_moment, yaw_moment

  auto it = this->model_sdf_objs.find(model_name);
  if (it == this->model_sdf_objs.end()) [[unlikely]]
    throw std::runtime_error("Model sdf object not found.");

  // The rotor plugin could be attached to the link or
  // the model, however, all of them will be aggregated under
  // model sdf object
  // Go through model plugins
  sdf::Plugins &plugins = it->second.Plugins();
  sdf::ElementPtr rotor_plugin_elem = nullptr;
  size_t idx = 0;

  for (size_t i = 0; i < plugins.size(); ++i)
  {
    sdf::Plugin plugin = plugins[i];
    if (plugin.Name() == "gz::sim::systems::MultiRotorPlugin")
    {
      rotor_plugin_elem = plugin.Element();
      idx = i;
      break;
    }
  }

  if (!rotor_plugin_elem) [[unlikely]]
  {
    print_err("Rotor plugin not found in model plugins");
    throw std::runtime_error("");
  }

  // Get parameters, we need thrust and torque constants
  // to compute the allocation matrix
  // The number of rotors as well
  std::vector<int> turning_directions;
  this->get_vector_from_sdf(rotor_plugin_elem, "turningDirections",
                            turning_directions,
                            turning_directions);
  const size_t num_rotors = turning_directions.size();

  // OPTIMIZED: Use constexpr dimension
  Eigen::MatrixXd allocation_matrix(CONTROL_INPUTS, num_rotors);

  auto thrust_constant_quadratic_params = rotor_plugin_elem->GetElement("thrustConstantQuadraticParams")->Get<gz::math::Vector3d>();
  const double thrust_const = thrust_constant_quadratic_params.X(); // assuming all rotors have same thrust constant

  auto torque_constant_quadratic_params = rotor_plugin_elem->GetElement("momentConstantQuadraticParams")->Get<gz::math::Vector3d>();
  const double torque_const = torque_constant_quadratic_params.X();  // assuming all rotors have same torque constant
  const double torque_to_thrust_ratio = torque_const / thrust_const; // Yaw torque per unit thrust

  std::vector<std::string> rotor_link_names;
  // Get rotor link names
  this->get_vector_from_sdf(rotor_plugin_elem, "linkNames",
                            rotor_link_names,
                            rotor_link_names);

  if (rotor_link_names.size() != num_rotors) [[unlikely]]
  {
    print_err("Rotor link names size does not match number of rotors");
    throw std::runtime_error("");
  }

  // Now compute allocation matrix
  // The first row is sum of all thrusts
  // The second row is roll moments
  // The third row is pitch moments
  // The fourth row is yaw moments
  allocation_matrix.setZero();
  allocation_matrix.row(0).setConstant(1.0); // Thrust row

  for (size_t i = 0; i < num_rotors; ++i)
  {
    auto link = it->second.LinkByName(rotor_link_names[i]);
    if (!link) [[unlikely]]
    {
      print_err("Rotor link " + rotor_link_names[i] + " not found in model sdf");
      throw std::runtime_error("");
    }
    auto sematic_pose = link->SemanticPose();
    gz::math::Pose3d pose_in_model;
    sematic_pose.Resolve(pose_in_model);
    const int turning_dir = turning_directions[i];

    // Fill in allocation matrix
    allocation_matrix(1, i) = pose_in_model.Pos().Y();               // MomentX arm
    allocation_matrix(2, i) = -pose_in_model.Pos().X();              // MomentY arm
    allocation_matrix(3, i) = -turning_dir * torque_to_thrust_ratio; // MomentZ contribution
  }

  return allocation_matrix;
}

Eigen::MatrixXd DRLHelperSystem::get_rotor_thrust_allocation_matrix(std::string model_name,
                                                                    const std::vector<std::string> &rotor_links, const std::vector<int> &turning_dir, double ktau)
{
  // OPTIMIZED: Use constexpr for known dimensions
  constexpr size_t CONTROL_INPUTS = 4; // thrust, roll_moment, pitch_moment, yaw_moment

  auto it = this->model_sdf_objs.find(model_name);
  if (it == this->model_sdf_objs.end()) [[unlikely]]
    throw std::runtime_error("Model sdf object not found.");

  const size_t num_rotors = turning_dir.size();

  // OPTIMIZED: Use constexpr dimension
  Eigen::MatrixXd allocation_matrix(CONTROL_INPUTS, num_rotors);

  if (rotor_links.size() != num_rotors) [[unlikely]]
  {
    print_err("Rotor link names size does not match number of rotors");
    throw std::runtime_error("");
  }

  // Now compute allocation matrix
  // The first row is sum of all thrusts
  // The second row is roll moments
  // The third row is pitch moments
  // The fourth row is yaw moments
  allocation_matrix.setZero();
  allocation_matrix.row(0).setConstant(1.0); // Thrust row

  for (size_t i = 0; i < num_rotors; ++i)
  {
    auto link = it->second.LinkByName(rotor_links[i]);
    if (!link) [[unlikely]]
    {
      print_err("Rotor link " + rotor_links[i] + " not found in model sdf");
      throw std::runtime_error("");
    }
    auto sematic_pose = link->SemanticPose();
    gz::math::Pose3d pose_in_model;
    sematic_pose.Resolve(pose_in_model);
    const int turning_diri = turning_dir[i];

    // Fill in allocation matrix
    allocation_matrix(1, i) = pose_in_model.Pos().Y();  // MomentX arm
    allocation_matrix(2, i) = -pose_in_model.Pos().X(); // MomentY arm
    allocation_matrix(3, i) = -turning_diri * ktau;     // MomentZ contribution
  }

  return allocation_matrix;
}

std::unordered_map<std::string, std::vector<gz::msgs::Contacts>> DRLHelperSystem::get_contacts(std::string_view model_name)
{
  std::unordered_map<std::string, std::vector<gz::msgs::Contacts>> out;

  const auto it = this->fast_link_slots.find(std::string(model_name));
  if (it == this->fast_link_slots.end()) [[unlikely]]
    return out;

  // OPTIMIZED: Reserve space in map to minimize rehashing
  out.reserve(it->second.size());

  // OPTIMIZED: Cache the slot vector reference and ECM pointer
  const auto &slots = it->second;
  const auto *ecm_ptr = this->_configure_ecm;

  // OPTIMIZED: Use index-based loop for better cache locality
  const size_t num_slots = slots.size();
  for (size_t i = 0; i < num_slots; ++i)
  {
    const auto &link_slot = slots[i];

    // Skip slots without contact data
    if (!link_slot.contacts || link_slot.contacts->empty()) [[unlikely]]
      continue;

    // OPTIMIZED: Get link name once and use it directly
    const auto name_opt = link_slot.link.Name(*ecm_ptr);
    if (!name_opt) [[unlikely]]
      continue;

    // OPTIMIZED: Move contact data instead of copy (if contacts are temporary)
    out[*name_opt] = *link_slot.contacts;
  }
  return out;
}

// internals
gz::sim::Model DRLHelperSystem::recursive_model_finder(const std::vector<std::string> &nested_names, const gz::sim::Model &parent_model, int idx, EntityComponentManager &_ecm)
{
  if (nested_names.size() == static_cast<size_t>(idx)) [[likely]]
    return parent_model;
  const gz::sim::Model nested_model(parent_model.ModelByName(_ecm, nested_names[idx]));
  return recursive_model_finder(nested_names, nested_model, idx + 1, _ecm);
}

std::vector<std::string> DRLHelperSystem::model_name_splitter(const std::string &s)
{
  // OPTIMIZED: Use constexpr for delimiter and reserve size
  constexpr std::string_view delimiter = "::";
  constexpr size_t delimiter_len = 2; // Length of "::"
  constexpr size_t expected_depth = 4;

  std::vector<std::string> out;
  out.reserve(expected_depth);

  std::size_t pos = 0, next;
  while ((next = s.find(delimiter, pos)) != std::string::npos)
  {
    out.emplace_back(s.substr(pos, next - pos));
    pos = next + delimiter_len;
  }
  out.emplace_back(s.substr(pos));
  return out;
}

void DRLHelperSystem::init_method(gz::sim::EntityComponentManager &_ecm)
{
  // OPTIMIZED: Pre-size vectors to avoid reallocations
  const size_t num_models = this->model_names.size();

  for (const auto &model_name : this->model_names)
  {
    const std::vector<std::string> split_name = model_name_splitter(model_name);
    const auto entityOpt = _ecm.EntityByName(split_name[0]);

    if (!entityOpt) [[unlikely]]
      throw std::runtime_error("Requested entity " + split_name[0] + " cannot be found in the simulation");

    const gz::sim::Model base_model(*entityOpt);
    this->models[model_name] = recursive_model_finder(split_name, base_model, 1, _ecm);

    // OPTIMIZED: Get reference once and reuse
    auto &model = this->models.at(model_name);
    const auto link_entities = model.Links(_ecm);
    const size_t num_links = link_entities.size();

    // OPTIMIZED: Clear and reserve all maps at once
    auto &link_map = this->links[model_name];
    link_map.clear();
    link_map.reserve(num_links);

    auto &state_map = this->pose_datas[model_name];
    state_map.clear();
    state_map.reserve(num_links);

    auto &contact_map = this->contacts[model_name];
    contact_map.clear();
    contact_map.reserve(num_links);

    auto &slots = this->fast_link_slots[model_name];
    slots.clear();
    slots.reserve(num_links);

    // OPTIMIZED: Use index-based loop and batch enable operations
    for (const auto &link_ent : link_entities)
    {
      gz::sim::Link link(link_ent);

      // OPTIMIZED: Enable both checks in one go (fewer ECM transactions)
      link.EnableVelocityChecks(_ecm, true);
      link.EnableAccelerationChecks(_ecm, true);

      const auto link_name_opt = link.Name(_ecm);
      if (!link_name_opt) [[unlikely]]
        continue;

      const std::string &lname = *link_name_opt;

      // OPTIMIZED: Emplace link directly
      link_map.emplace(lname, std::move(link));

      // OPTIMIZED: Use try_emplace to avoid redundant lookups
      auto it_state = state_map.try_emplace(lname, GZ_state::Zero()).first;
      auto it_cont = contact_map.try_emplace(lname, std::vector<gz::msgs::Contacts>{}).first;

      // OPTIMIZED: Build slot with direct references
      slots.push_back({link_map.at(lname), &it_state->second, &it_cont->second});
    }
  }

  this->world_entity = gz::sim::worldEntity(_ecm);
}

void DRLHelperSystem::set_wrench(std::string model_name, std::string link_name,
                                 Eigen::Vector3d &force, Eigen::Vector3d &moments)
{
  if (!this->_configure_ecm) [[unlikely]]
    return;

  const auto mit = this->models.find(model_name);
  if (mit == this->models.end()) [[unlikely]]
    return;

  const auto lit = this->links.at(model_name).find(link_name);
  if (lit == this->links.at(model_name).end()) [[unlikely]]
    return;

  // OPTIMIZED: Direct construction without intermediate variables
  lit->second.AddWorldWrench(*this->_configure_ecm,
                             {force(0), force(1), force(2)},
                             {moments(0), moments(1), moments(2)});
}

void DRLHelperSystem::set_wrench(std::string model_name, std::string link_name,
                                 Eigen::Vector3d &&force, Eigen::Vector3d &&moments)
{
  this->set_wrench(std::move(model_name), std::move(link_name), force, moments);
}

// ---- DRLServer ----
DRLServer::DRLServer(const std::string &partition, const std::string &sdf_file,
                     const std::vector<std::string> &model_names,
                     bool enable_sensors, const DRLServerConfig &config)
    : sensors_enabled_(enable_sensors), _partition(partition), _sdf_file(sdf_file), _model_names(model_names), model_names(model_names), drl_server_config(config)
{
  this->init_method();
}

void DRLServer::run_once() { this->run_N(1); }
void DRLServer::run_N(int N)
{
  if (N <= 0)
    throw std::invalid_argument("N must be greater than zero");
  this->server->Run(true, static_cast<std::uint64_t>(N), false);
}

std::unordered_map<std::string, GZ_state> DRLServer::state_info(std::string model_name)
{
  return this->internal_sys->state_info(std::move(model_name));
}

void DRLServer::reset_pos(std::string model_name, Eigen::Vector3d &position, Eigen::Vector3d &orientation)
{
  // OPTIMIZED: Use constexpr for stabilization steps
  constexpr int STABILIZATION_STEPS = 3;

  // OPTIMIZED: Pass by const reference to avoid copy, then move when needed
  this->internal_sys->reset_pos(model_name, position, orientation);
  this->run_N(STABILIZATION_STEPS);

  // OPTIMIZED: Use iterator instead of range-based for for potential optimization
  for (auto &elem : this->marker_managers)
    elem.second->reset();
}

void DRLServer::reset_pos(std::string model_name, Eigen::Vector3d &&position, Eigen::Vector3d &&orientation)
{
  this->reset_pos(model_name, position, orientation);
}

void DRLServer::respawn_model(std::string model_name, Eigen::Vector3d &position, Eigen::Vector3d &orientation)
{
  // OPTIMIZED: Use constexpr for stabilization steps
  constexpr int STABILIZATION_STEPS = 3;

  // OPTIMIZED: Pass by const reference to avoid copy, then move when needed
  this->internal_sys->respawn_model(model_name, position, orientation);
  this->run_N(STABILIZATION_STEPS);

  // OPTIMIZED: Use iterator instead of range-based for for potential optimization
  for (auto &elem : this->marker_managers)
    elem.second->reset();
}

void DRLServer::respawn_model(std::string model_name, Eigen::Vector3d &&position, Eigen::Vector3d &&orientation)
{
  this->respawn_model(model_name, position, orientation);
}

void DRLServer::set_rotor_velocity_cmd(std::string model_name, std::string link_name, Eigen::VectorXd &cmd)
{
  this->internal_sys->set_rotor_velocity_cmd(std::move(model_name), std::move(link_name), cmd);
}
void DRLServer::set_rotor_velocity_cmd(std::string model_name, std::string link_name, Eigen::VectorXd &&cmd)
{
  this->internal_sys->set_rotor_velocity_cmd(std::move(model_name), std::move(link_name), std::move(cmd));
}

void DRLServer::set_ctbr_cmd(std::string model_name, std::string link_name, Eigen::VectorXd &cmd)
{
  this->update_control_states();
  const auto &control_states_ = this->control_states.at(model_name).at(link_name);
  const auto &state = std::get<0>(control_states_);
  Eigen::Quaterniond quat(state(6), state(7), state(8), state(9));
  Eigen::VectorXd cmd_ = cmd;
  auto it = ctbr_rate_limiters.find({model_name, link_name});
  if (it != ctbr_rate_limiters.end())
  {
    cmd_ = it->second->UpdateFilter(cmd, this->internal_sys->step_size_());
  }
  Eigen::Vector3d F_w = quat * Eigen::Vector3d(0.0, 0.0, cmd_(0));
  Eigen::Vector3d M_w = Eigen::Vector3d::Zero();
  Eigen::Vector3d W_b = cmd_.segment<3>(1);
  this->set_wrench(model_name, link_name, F_w, M_w);
  this->set_angular_velocity_cmd(model_name, link_name, W_b);
}
void DRLServer::set_ctbr_cmd(std::string model_name, std::string link_name, Eigen::VectorXd &&cmd)
{
  this->set_ctbr_cmd(std::move(model_name), std::move(link_name), cmd);
}

void DRLServer::set_ctbr_cmd(std::string model_name, std::string link_name, Eigen::VectorXd &cmd, const Eigen::VectorXd &kp, const Eigen::VectorXd &kd)
{
  this->update_control_states();
  const auto &control_states_ = this->control_states.at(model_name).at(link_name);
  const auto &state = std::get<0>(control_states_);
  const auto &state_dot = std::get<1>(control_states_);
  Eigen::Quaterniond quat(state(6), state(7), state(8), state(9));
  Eigen::Vector3d omega = state.segment<3>(10);
  Eigen::Vector3d alpha = state_dot.segment<3>(10);

  Eigen::VectorXd cmd_ = cmd;
  auto it = ctbr_rate_limiters.find({model_name, link_name});
  if (it != ctbr_rate_limiters.end())
  {
    cmd_ = it->second->UpdateFilter(cmd, this->internal_sys->step_size_());
  }
  Eigen::Vector3d F_w = quat * Eigen::Vector3d(0.0, 0.0, cmd_(0));
  Eigen::Vector3d M_b = (cmd_.segment<3>(1) - omega).cwiseProduct(kp) - kd.cwiseProduct(alpha);
  Eigen::Vector3d M_w = quat * M_b;
  this->set_wrench(model_name, link_name, F_w, M_w);
}
void DRLServer::set_ctbr_cmd(std::string model_name, std::string link_name, Eigen::VectorXd &&cmd, const Eigen::VectorXd &kp, const Eigen::VectorXd &kd)
{
  this->set_ctbr_cmd(std::move(model_name), std::move(link_name), cmd, kp, kd);
}

void DRLServer::set_ctbt_cmd(std::string model_name, std::string link_name, Eigen::VectorXd &cmd)
{
  this->update_control_states();
  const auto &control_states_ = this->control_states.at(model_name).at(link_name);
  const auto &state = std::get<0>(control_states_);
  Eigen::Quaterniond quat(state(6), state(7), state(8), state(9));
  Eigen::VectorXd cmd_ = cmd;
  auto it = ctbt_rate_limiters.find({model_name, link_name});
  if (it != ctbt_rate_limiters.end())
  {
    cmd_ = it->second->UpdateFilter(cmd_, this->internal_sys->step_size_());
  }
  Eigen::Vector3d F_w = quat * Eigen::Vector3d(0.0, 0.0, cmd_(0));
  Eigen::Vector3d M_w = quat * cmd_.segment<3>(1);
  this->set_wrench(model_name, link_name, F_w, M_w);
}
void DRLServer::set_ctbt_cmd(std::string model_name, std::string link_name, Eigen::VectorXd &&cmd)
{
  this->set_ctbt_cmd(std::move(model_name), std::move(link_name), cmd);
}

void DRLServer::set_srt_cmd(std::string model_name, std::string base_link, const std::vector<std::string> &link_names,
                            const std::vector<int> &turning_dir, Eigen::VectorXd &cmd, double ktau)
{
  auto it = srt_rate_limiters.find({model_name, base_link});
  if (it != srt_rate_limiters.end())
  {
    Eigen::VectorXd cmd_ = it->second->UpdateFilter(cmd, this->internal_sys->step_size_());
    this->internal_sys->set_srt_cmd(std::move(model_name), std::move(base_link), link_names, turning_dir, cmd_, ktau);
  }
  else
  {
    this->internal_sys->set_srt_cmd(std::move(model_name), std::move(base_link), link_names, turning_dir, cmd, ktau);
  }
}
void DRLServer::set_srt_cmd(std::string model_name, std::string base_link, const std::vector<std::string> &link_names,
                            const std::vector<int> &turning_dir, Eigen::VectorXd &&cmd, double ktau)
{
  auto it = srt_rate_limiters.find({model_name, base_link});
  if (it != srt_rate_limiters.end())
  {
    Eigen::VectorXd cmd_ = it->second->UpdateFilter(cmd, this->internal_sys->step_size_());
    this->internal_sys->set_srt_cmd(std::move(model_name), std::move(base_link), link_names, turning_dir, std::move(cmd_), ktau);
  }
  else
  {
    this->internal_sys->set_srt_cmd(std::move(model_name), std::move(base_link), link_names, turning_dir, std::move(cmd), ktau);
  }
}

void DRLServer::set_velocity_cmd(std::string model_name, std::string link_name, Eigen::Vector3d &cmd)
{
  this->internal_sys->set_velocity_cmd(std::move(model_name), std::move(link_name), cmd);
}
void DRLServer::set_velocity_cmd(std::string model_name, std::string link_name, Eigen::Vector3d &&cmd)
{
  this->internal_sys->set_velocity_cmd(std::move(model_name), std::move(link_name), std::move(cmd));
}

void DRLServer::set_angular_velocity_cmd(std::string model_name, std::string link_name, Eigen::Vector3d &cmd)
{
  this->internal_sys->set_angular_velocity_cmd(std::move(model_name), std::move(link_name), cmd);
}
void DRLServer::set_angular_velocity_cmd(std::string model_name, std::string link_name, Eigen::Vector3d &&cmd)
{
  this->internal_sys->set_angular_velocity_cmd(std::move(model_name), std::move(link_name), std::move(cmd));
}

void DRLServer::set_ackermann_velocity_cmd(std::string model_name, std::string link_name, Eigen::Vector2d &cmd)
{
  this->internal_sys->set_ackermann_velocity_cmd(std::move(model_name), std::move(link_name), cmd);
}
void DRLServer::set_ackermann_velocity_cmd(std::string model_name, std::string link_name, Eigen::Vector2d &&cmd)
{
  this->internal_sys->set_ackermann_velocity_cmd(std::move(model_name), std::move(link_name), std::move(cmd));
}

void DRLServer::set_mass(std::string model_name, std::string link_name, double mass)
{
  this->internal_sys->set_mass(model_name, link_name, mass);
}
void DRLServer::set_inertia(std::string model_name, std::string link_name, Eigen::Matrix3d &inertia)
{
  this->internal_sys->set_inertia(model_name, link_name, inertia);
}
void DRLServer::set_inertia(std::string model_name, std::string link_name, Eigen::Matrix3d &&inertia)
{
  this->internal_sys->set_inertia(model_name, link_name, std::move(inertia));
}

Eigen::Matrix3d DRLServer::get_inertia(std::string model_name, std::string link_name)
{
  return this->internal_sys->get_inertia(model_name, link_name);
}
double DRLServer::get_mass(std::string model_name, std::string link_name)
{
  return this->internal_sys->get_mass(model_name, link_name);
}
void DRLServer::set_rotor_parameters(std::string model_name, const RotorParameters &params)
{
  this->internal_sys->set_rotor_parameters(model_name, params);
}

RotorParameters DRLServer::get_rotor_parameters(std::string model_name)
{
  return this->internal_sys->get_rotor_parameters(model_name);
}

Eigen::MatrixXd DRLServer::get_rotor_thrust_allocation_matrix(std::string model_name)
{
  return this->internal_sys->get_rotor_thrust_allocation_matrix(model_name);
}
Eigen::MatrixXd DRLServer::get_inverse_rotor_thrust_allocation_matrix(std::string model_name)
{
  auto mat = this->internal_sys->get_rotor_thrust_allocation_matrix(model_name);
  return mat.completeOrthogonalDecomposition().pseudoInverse();
}

Eigen::MatrixXd DRLServer::get_rotor_thrust_allocation_matrix(std::string model_name,
                                                              const std::vector<std::string> &rotor_links, const std::vector<int> &turning_dir, double ktau)
{
  return this->internal_sys->get_rotor_thrust_allocation_matrix(model_name, rotor_links, turning_dir, ktau);
}
Eigen::MatrixXd DRLServer::get_inverse_rotor_thrust_allocation_matrix(std::string model_name,
                                                                      const std::vector<std::string> &rotor_links, const std::vector<int> &turning_dir, double ktau)
{
  auto mat = this->internal_sys->get_rotor_thrust_allocation_matrix(model_name, rotor_links, turning_dir, ktau);
  return mat.completeOrthogonalDecomposition().pseudoInverse();
}

void DRLServer::set_joint_position_cmd(std::string model_name, std::string joint_name, Eigen::Vector3d &&cmd)
{
  this->internal_sys->set_joint_position_cmd(model_name, joint_name, std::move(cmd));
}
void DRLServer::set_joint_position_cmd(std::string model_name, std::string joint_name, Eigen::Vector3d &cmd)
{
  this->internal_sys->set_joint_position_cmd(model_name, joint_name, cmd);
}

std::unordered_map<std::string, std::vector<gz::msgs::Contacts>> DRLServer::get_contacts(std::string model_name)
{
  return this->internal_sys->get_contacts(std::move(model_name));
}

void DRLServer::set_env(std::string partition)
{
  setenv("GZ_PARTITION", partition.c_str(), 1);
  if (!getenv("GZ_PARTITION"))
  {
    std::cerr << "Failed to isolate server. Construction failed." << std::endl;
    return;
  }
}

void DRLServer::init_method()
{
  // OPTIMIZED: Use constexpr for magic numbers
  constexpr int VERBOSITY_LEVEL = 1;
  constexpr double MAX_UPDATE_RATE = 1e9;
  constexpr int POST_INIT_STEPS = 2;

  std::lock_guard<std::mutex> lock(DRLServer::construction_mutex);
  identity = this->_partition;
  DRLServer::set_env(this->_partition);
  this->server.reset();
  this->server_config.reset();
  gz::common::Console::SetVerbosity(VERBOSITY_LEVEL);

  server_config = std::make_unique<gz::sim::ServerConfig>();
  server_config->SetSdfFile(this->_sdf_file);
  server_config->SetUpdateRate(MAX_UPDATE_RATE);
  server_config->SetHeadlessRendering(this->headless_render);
  server_config->SetUseLevels(false);

  this->server = std::make_unique<DRL_SERVER_TYPE>(*server_config);
  // this->server->SetUpdatePeriod(1ns);
  this->internal_sys = std::make_shared<DRLHelperSystem>(this);
  this->server->AddSystem(this->internal_sys);
  this->run_N(POST_INIT_STEPS);

  if (this->sensors_enabled_) [[unlikely]]
  {
    this->sensor_plugin = std::make_shared<systems::custom_plugins::Sensors>();
    this->server->AddSystem(this->sensor_plugin);
  }
  this->run_N(POST_INIT_STEPS);
}

void DRLServer::set_wrench(std::string model_name, std::string link_name, Eigen::Vector3d &force, Eigen::Vector3d &moments)
{
  this->internal_sys->set_wrench(std::move(model_name), std::move(link_name), force, moments);
}
void DRLServer::set_wrench(std::string model_name, std::string link_name, Eigen::Vector3d &&force, Eigen::Vector3d &&moments)
{
  this->internal_sys->set_wrench(std::move(model_name), std::move(link_name), std::move(force), std::move(moments));
}

void DRLServer::control_with_rotor_velocity(std::string model_name, std::string link_name, const Stated &desired_state, int N)
{
  auto &controller_ = this->link_controllers.at({model_name, link_name});
  auto &actuator_map = this->thrust_moments_to_rotor_velocity_map.at({model_name, link_name});
  Eigen::Vector3d force;
  Eigen::Vector3d moments;
  Eigen::Vector4d cmd;
  this->update_control_states();
  const auto &control_states_ = this->control_states.at(model_name).at(link_name);
  const auto &state = std::get<0>(control_states_);
  const auto &state_dot = std::get<1>(control_states_);

  for (int i = 0; i < N; ++i)
  {
    this->update_control_states();
    cmd = controller_->calculate_thrust_moments(state,
                                                state_dot,
                                                desired_state);
    this->set_rotor_velocity_cmd(model_name, link_name, actuator_map(cmd));
    this->run_once();
  }
  if (auto it = this->marker_managers.find({model_name, link_name}); it != this->marker_managers.end())
    it->second->set_marker(std::get<0>(this->control_states.at(model_name).at(link_name)).segment<3>(0));
}

void DRLServer::control_with_wrench(std::string model_name, std::string link_name, const Stated &desired_state, int N)
{
  auto &controller_ = this->link_controllers.at({model_name, link_name});
  // auto &actuator_map = this->thrust_moments_to_rotor_velocity_map.at({model_name, link_name});
  Eigen::Vector3d force;
  Eigen::Vector3d moments;
  Eigen::Vector4d cmd;
  const auto &control_states_ = this->control_states.at(model_name).at(link_name);
  const auto &state = std::get<0>(control_states_);
  const auto &state_dot = std::get<1>(control_states_);
  Eigen::Quaterniond q;
  for (int i = 0; i < N; ++i)
  {
    this->update_control_states();
    force = controller_->calculate_force(state,
                                         state_dot,
                                         desired_state);
    moments = controller_->calculate_moments(state,
                                             state_dot,
                                             desired_state,
                                             force);
    q = Eigen::Quaterniond(state(6), state(7), state(8), state(9));
    force << 0.0, 0.0, (q.inverse() * force)(2);
    force = q * force; // body to world
    moments = q * moments;
    this->set_wrench(model_name, link_name, force, moments);
    // this->set_rotor_velocity_cmd(model_name, link_name, actuator_map(cmd));
    this->run_once();
  }
  if (auto it = this->marker_managers.find({model_name, link_name}); it != this->marker_managers.end())
    it->second->set_marker(std::get<0>(this->control_states.at(model_name).at(link_name)).segment<3>(0));
}

std::function<Eigen::VectorXd(const Eigen::Vector4d &)> DRLServer::get_thrust_moment_to_rotor_velocity_mapping_function(std::string model_name)
{

  auto allocation_matrix = this->get_inverse_rotor_thrust_allocation_matrix(model_name);
  // get rotor parameters
  auto rotor_params = this->get_rotor_parameters(model_name);
  double thrust_const = rotor_params.thrust_constant_quadratic_params[0];
  double max_rot_velocity = rotor_params.max_rot_velocity;
  allocation_matrix /= thrust_const;
  auto actuator_map = [allocation_matrix, max_rot_velocity](const Eigen::Vector4d &u)
  {
    Eigen::VectorXd cmd = (allocation_matrix * u).cwiseMax(0.0).cwiseSqrt().cwiseMin(max_rot_velocity);
    return cmd;
  };
  return actuator_map;
}

std::function<Eigen::VectorXd(const Eigen::Vector4d &)> DRLServer::get_thrust_moment_to_rotor_thrust_mapping_function(std::string model_name)
{

  auto allocation_matrix = this->get_inverse_rotor_thrust_allocation_matrix(model_name);
  auto actuator_map = [allocation_matrix](const Eigen::Vector4d &u)
  {
    Eigen::VectorXd cmd = (allocation_matrix * u);
    return cmd;
  };
  return actuator_map;
}

std::function<Eigen::VectorXd(const Eigen::Vector4d &)> DRLServer::get_thrust_moment_to_rotor_thrust_mapping_function(std::string model_name,
                                                                                                                      const std::vector<std::string> &link_names, std::vector<int> &turning_dir, double ktau)
{

  auto allocation_matrix = this->get_inverse_rotor_thrust_allocation_matrix(model_name, link_names, turning_dir, ktau);
  auto actuator_map = [allocation_matrix](const Eigen::Vector4d &u)
  {
    Eigen::VectorXd cmd = (allocation_matrix * u);
    return cmd;
  };
  return actuator_map;
}

void DRLServer::set_controller(std::string model_name, std::string link_name, std::shared_ptr<UAVController> controller)
{
  thrust_moments_to_rotor_velocity_map[{model_name, link_name}] = this->get_thrust_moment_to_rotor_velocity_mapping_function(model_name);
  this->link_controllers[{std::move(model_name), std::move(link_name)}] = std::move(controller);
  // Update the thrust_moments_to_rotor_velocity_map
}

void DRLServer::update_control_states()
{
  for (auto &name : this->model_names)
  {
    this->internal_sys->for_each_state_fast(name, [&](const std::string &link_name, const GZ_state &pose_data)
                                            {
      auto &entry = this->control_states[name][link_name];

      auto &cs = std::get<0>(entry);
      cs = { pose_data(0), pose_data(1), pose_data(2),
             pose_data(7), pose_data(8), pose_data(9),
             pose_data(3), pose_data(4), pose_data(5), pose_data(6),
             pose_data(10), pose_data(11), pose_data(12) };

      auto &csdot = std::get<1>(entry);
      Eigen::Quaterniond quat{ cs[6], cs[7], cs[8], cs[9] };
      Eigen::Vector3d omega{ cs[10], cs[11], cs[12] };
      auto quat_deriv = uav_controllers::quaternion_derivative(quat, omega);

      csdot = { pose_data(7), pose_data(8), pose_data(9),
                pose_data(13), pose_data(14), pose_data(15),
                quat_deriv(0), quat_deriv(1), quat_deriv(2), quat_deriv(3),
                pose_data(16), pose_data(17), pose_data(18) }; });
  }
}

void DRLServer::set_trajectory_trace(std::string model_name, std::string link_name, DRLServerConfig *config_)
{
  if (!config_)
    config_ = &this->drl_server_config;
  auto key = std::make_pair(model_name, link_name);
  if (this->marker_managers.find(key) == this->marker_managers.end())
    this->marker_managers.emplace(std::move(key), std::make_unique<MarkerManagerDRL>(*config_, this->_partition));
}
void DRLServer::set_headless_render_mode(bool mode) { this->headless_render = mode; }

gz::msgs::Image DRLServer::get_sensor_image(std::string name)
{
  if (this->sensors_enabled_)
    return sensor_plugin->GetSensorImg(name);
  else
    throw std::runtime_error("Sensors not enabled in this DRLServer instance.");
}

gz::msgs::CameraInfo DRLServer::get_camera_info(std::string name)
{
  if (this->sensors_enabled_)
    return sensor_plugin->GetCameraInfo(name);
  else
    throw std::runtime_error("Sensors not enabled in this DRLServer instance.");
}

gz::math::Pose3d DRLServer::get_camera_pose(std::string name)
{
  if (this->sensors_enabled_)
    return sensor_plugin->GetCameraPose(name);
  else
    throw std::runtime_error("Sensors not enabled in this DRLServer instance.");
}

gz::math::Pose3d DRLServer::get_lidar_pose(std::string name)
{
  if (this->sensors_enabled_)
    return sensor_plugin->GetLidarPose(name);
  else
    throw std::runtime_error("Sensors not enabled in this DRLServer instance.");
}

systems::custom_plugins::Sensors::LidarFrameView DRLServer::get_sensor_gpu_lidar(std::string name)
{
  if (this->sensors_enabled_)
    return sensor_plugin->GetSensorGpuLidar(name);
  else
    throw std::runtime_error("Sensors not enabled in this DRLServer instance.");
}
void DRLServer::print_sensor_names()
{
  if (this->sensors_enabled_)
    sensor_plugin->PrintSensorNames();
  else
    throw std::runtime_error("Sensors not enabled in this DRLServer instance.");
}
std::vector<std::string> DRLServer::camera_sensor_names()
{
  if (this->sensors_enabled_)
    return sensor_plugin->CameraSensorNames();
  else
    throw std::runtime_error("Sensors not enabled in this DRLServer instance.");
};
std::vector<std::string> DRLServer::lidar_sensor_names()
{
  if (this->sensors_enabled_)
    return sensor_plugin->LidarSensorNames();
  else
    throw std::runtime_error("Sensors not enabled in this DRLServer instance.");
}

void DRLServer::bind_img_cb(std::function<void(const gz::msgs::Image &)> cb, std::string_view name)
{
  if (this->sensors_enabled_)
    sensor_plugin->BindImgCB(cb, name);
  else
    throw std::runtime_error("Sensors not enabled in this DRLServer instance.");
};
void DRLServer::bind_lidar_cb(std::function<void(const systems::custom_plugins::Sensors::LidarFrameView &)> cb,
                              std::string_view name)
{
  if (this->sensors_enabled_)
    sensor_plugin->BindLidarCB(cb, name);
  else
    throw std::runtime_error("Sensors not enabled in this DRLServer instance.");
}

void DRLServer::start_camera_recording(std::string cam_name, int height, int width, int fps,
                        const Eigen::Vector3d& pos, const Eigen::Vector3d& ori,
                        std::string output_file){
    if (this->sensors_enabled_){
      gz::math::Pose3d pose(pos(0), pos(1), pos(2), ori(0), ori(1), ori(2));
      sensor_plugin->StartCameraRecording(cam_name, height, width, fps, pose, output_file);
      // The recorder camera is created on the render thread during a
      // simulation step. In a world with no other rendering sensors this
      // also triggers render-scene initialization, which can take a while.
      // Step until the recording is live so footage reliably starts here.
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(2);
      while (sensor_plugin->CameraRecordingQueued(cam_name))
      {
        this->run_once();
        if (std::chrono::steady_clock::now() > deadline)
          throw std::runtime_error("Timed out waiting for camera recording to start: " + cam_name);
      }
      // Queued -> Active on success; the state entry is removed on failure
      // (camera or encoder could not be created).
      if (!sensor_plugin->CameraRecordingBusy(cam_name))
        throw std::runtime_error("Failed to start camera recording (see console errors): " + cam_name);
    }
    else{
    throw std::runtime_error("Sensors not enabled in this DRLServer instance.");
    }


    }
void DRLServer::update_camera_recording_pose(std::string cam_name,const Eigen::Vector3d& pos, const Eigen::Vector3d& ori){
  if (this->sensors_enabled_){
      gz::math::Pose3d pose(pos(0), pos(1), pos(2), ori(0), ori(1), ori(2));
      sensor_plugin->UpdateCameraRecordingPose(cam_name, pose);
    }
    else{
    throw std::runtime_error("Sensors not enabled in this DRLServer instance.");
    }
}

void DRLServer::stop_camera_recording(std::string cam_name){
  if (this->sensors_enabled_){
      sensor_plugin->StopCameraRecording(cam_name);
      // The stop is processed on the render thread during a simulation
      // step. Step until the encoder has finalized the video file.
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(2);
      while (sensor_plugin->CameraRecordingBusy(cam_name))
      {
        this->run_once();
        if (std::chrono::steady_clock::now() > deadline)
          throw std::runtime_error("Timed out waiting for camera recording to stop: " + cam_name);
      }
    }
    else{
    throw std::runtime_error("Sensors not enabled in this DRLServer instance.");
    }
}

void DRLServer::reset_world(const std::unordered_map<std::string, std::pair<Eigen::Vector3d, Eigen::Vector3d>> &model_poses)
{
  auto world_ptr = this->internal_sys->sdf_root_().WorldByIndex(0);
  for (const auto &iter : model_poses)
  {
    const auto &name = iter.first;
    const auto &po = iter.second.first;
    const auto &ori = iter.second.second;
    gz::math::Pose3d pose(po(0), po(1), po(2), ori(0), ori(1), ori(2));
    auto model_ptr = world_ptr->ModelByName(name);
    if (!model_ptr)
      throw std::runtime_error("Model not found in SDF: " + name);
    model_ptr->SetRawPose(pose);
  }
  auto &sdf_creator = this->internal_sys->sdf_creator_();
  sdf_creator->RequestRemoveEntity(this->internal_sys->world_entity_(), true);
  this->run_N(1);
  sdf_creator->CreateEntities(world_ptr, this->internal_sys->world_entity_());
  this->run_N(2);
  this->server->AddSystem(this->internal_sys);
}

void DRLServer::request_contact_data(std::string model_name)
{
  this->internal_sys->request_contact_data(model_name);
}

double DRLServer::step_size()
{
  return this->internal_sys->step_size_();
}

void DRLServer::set_srt_rate_limiter_time_constants(std::string model_name, std::string link_name, double tau_up, double tau_down,
                                                    const Eigen::VectorXd &initial_value)
{
  srt_rate_limiters[{model_name, link_name}] = std::make_unique<FirstOrderFilter<Eigen::VectorXd>>(tau_up, tau_down, initial_value);
}
void DRLServer::set_ctbr_rate_limiter_time_constants(std::string model_name, std::string link_name, double tau_up, double tau_down,
                                                     const Eigen::VectorXd &initial_value)
{
  ctbr_rate_limiters[{model_name, link_name}] = std::make_unique<FirstOrderFilter<Eigen::VectorXd>>(tau_up, tau_down, initial_value);
}
void DRLServer::set_ctbt_rate_limiter_time_constants(std::string model_name, std::string link_name, double tau_up, double tau_down,
                                                     const Eigen::VectorXd &initial_value)
{
  ctbt_rate_limiters[{model_name, link_name}] = std::make_unique<FirstOrderFilter<Eigen::VectorXd>>(tau_up, tau_down, initial_value);
}
void DRLServer::reset_srt_rate_limiter(std::string model_name, std::string link_name)
{
  auto it = srt_rate_limiters.find({model_name, link_name});
  if (it != srt_rate_limiters.end())
  {
    it->second->Reset();
  }
}
void DRLServer::reset_ctbr_rate_limiter(std::string model_name, std::string link_name)
{
  auto it = ctbr_rate_limiters.find({model_name, link_name});
  if (it != ctbr_rate_limiters.end())
  {
    it->second->Reset();
  }
}
void DRLServer::reset_ctbt_rate_limiter(std::string model_name, std::string link_name)
{
  auto it = ctbt_rate_limiters.find({model_name, link_name});
  if (it != ctbt_rate_limiters.end())
  {
    it->second->Reset();
  }
}
bool DRLServer::set_marker(std::string model_name, std::string link_name){
   if (auto it = this->marker_managers.find({model_name, link_name}); it != this->marker_managers.end()){
    it->second->set_marker(std::get<0>(this->control_states.at(model_name).at(link_name)).segment<3>(0));
    return true;
  }
  else{
    return false;
  }
}
DRLServer::~DRLServer()
{
  this->server->Stop();
  if (this->sensor_plugin)
    this->sensor_plugin->Stop();
}
