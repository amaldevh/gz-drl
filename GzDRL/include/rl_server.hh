// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#pragma once
#ifndef GZ_SIM_DRL_SERVER_HH
#define GZ_SIM_DRL_SERVER_HH

#if defined(__GNUC__) && defined(GZDRL_NATIVE_OPTIMIZATIONS)
#pragma GCC optimize("O3", "fast-math", "unroll-loops", "inline-functions")
#pragma GCC target("avx2", "fma")
#endif
#define DRL_SERVER_TYPE gz::sim::Server

#include <unordered_map>
#include <mutex>
#include <utility>
#include <type_traits>
#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <chrono>
#include <thread>
#include <cstring>
#include "common/common.hh"
#include "common/op_process.hh"
#include "gz/sim/Server.hh"
#include "gz/sim/ServerConfig.hh"
#include "gz/sim/System.hh"
#include "gz/sim/Util.hh"
#include "gz/sim/Model.hh"
#include "gz/sim/Link.hh"
#include "gz/sim/components/Actuators.hh"
#include "gz/sim/components/ContactSensorData.hh"
#include "gz/sim/components/AngularVelocityCmd.hh"
#include "gz/sim/components/LinearVelocityCmd.hh"
#include "gz/sim/components/JointPosition.hh"
#include "gz/sim/SdfEntityCreator.hh"
#include <gz/msgs/contacts.pb.h>
#include <gz/msgs/actuators.pb.h>

class DRLServer;
using namespace gz;
using namespace sim;

/**
 * @brief Compare two actuator messages for exact equality
 * @param a First actuator message
 * @param b Second actuator message
 * @return true if messages are identical (same size and all velocities equal)
 * @note O(n) complexity where n is velocity_size()
 */
[[gnu::always_inline, gnu::hot, gnu::pure]]
static inline bool EqualActuatorMsgs(const msgs::Actuators &a, const msgs::Actuators &b) noexcept
{
  const auto n = a.velocity_size();
  if (n != b.velocity_size()) [[unlikely]]
    return false;

  const double *__restrict__ pa = a.velocity().data();
  const double *__restrict__ pb = b.velocity().data();

  for (int i = 0; i < n; ++i)
  {
    if (pa[i] != pb[i]) [[unlikely]]
      return false;
  }
  return true;
}

/**
 * @brief Compare two 3D vectors for exact equality
 * @param a First vector
 * @param b Second vector
 * @return true if all three components are exactly equal
 */
[[gnu::always_inline, gnu::hot, gnu::pure]]
static inline bool EqualVectors3(const math::Vector3d &a, const math::Vector3d &b) noexcept
{
  return (a[0] == b[0]) & (a[1] == b[1]) & (a[2] == b[2]);
}

/**
 * @brief Compare two 2D vectors for exact equality
 * @param a First vector
 * @param b Second vector
 * @return true if both components are exactly equal
 */
[[gnu::always_inline, gnu::hot, gnu::pure]]
static inline bool EqualVectors2(const math::Vector2d &a, const math::Vector2d &b) noexcept
{
  return (a[0] == b[0]) & (a[1] == b[1]);
}

/**
 * @brief Compare two std::vector<double> for exact equality
 * @param a First vector
 * @param b Second vector
 * @return true if vectors have same size and all elements are exactly equal
 */
[[gnu::always_inline, gnu::hot, gnu::pure]]
static inline bool EqualStdVec(const std::vector<double> &a, const std::vector<double> &b) noexcept
{
  const size_t n = a.size();
  if (n != b.size()) [[unlikely]]
    return false;

  const double *__restrict__ pa = a.data();
  const double *__restrict__ pb = b.data();

  for (size_t i = 0; i < n; ++i)
  {
    if (pa[i] != pb[i]) [[unlikely]]
      return false;
  }
  return true;
}

/**
 * @class DRLHelperSystem
 * @brief Core Gazebo system plugin that manages simulation state and commands
 *
 * This system integrates with Gazebo's Entity-Component-System (ECS) architecture
 * to provide low-latency DRL interface for reinforcement learning environments.
 *
 */
// =============================================================================
class DRLHelperSystem : public gz::sim::System,
                        public gz::sim::ISystemConfigure, // One-time setup
                        public gz::sim::ISystemPreUpdate, // Before physics step
                        public gz::sim::ISystemUpdate,    // During physics step
                        public gz::sim::ISystemPostUpdate // After physics step
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Constructor - initializes helper system with parent DRLServer
   * @param server Pointer to parent DRLServer (must remain valid for lifetime)
   */
  explicit DRLHelperSystem(DRLServer *server);

  /** @brief Destructor - cleans up SDF entity creator */
  ~DRLHelperSystem();

  /**
   * @brief Configure the system from SDF parameters
   * @param _entity World entity ID
   * @param _sdf SDF element containing configuration
   * @param _ecm Entity Component Manager reference (cached for apply-now)
   * @param _eventMgr Event manager for simulation events
   * @details One-time initialization: parses SDF, caches ECM pointer,
   *          initializes noise models, and creates entity lookup tables
   */
  void Configure(const gz::sim::Entity &_entity,
                 const std::shared_ptr<const sdf::Element> &_sdf,
                 gz::sim::EntityComponentManager &_ecm,
                 gz::sim::EventManager &_eventMgr) override;

  /**
   * @brief PreUpdate phase - handle entity resets
   * @param _info Update information (time, iteration count)
   * @param _ecm Entity Component Manager
   * @details Processes entity removal/respawn requests. All velocity commands
   *          are applied immediately via set_* methods, not here.
   */
  void PreUpdate(const gz::sim::UpdateInfo &_info,
                 gz::sim::EntityComponentManager &_ecm) override;

  /**
   * @brief Update phase - currently unused
   * @param _info Update information
   * @param _ecm Entity Component Manager
   * @details Reserved for future extensions
   */
  void Update(const gz::sim::UpdateInfo &_info,
              gz::sim::EntityComponentManager &_ecm) override;

  /**
   * @brief PostUpdate phase - extract state and apply noise
   * @param _info Update information
   * @param _ecm Entity Component Manager (const - read-only access)
   * @details Reads pose/velocity/acceleration from ECM using fast_link_slots
   *          for zero-copy access, then applies Ornstein-Uhlenbeck noise
   */
  void PostUpdate(const gz::sim::UpdateInfo &_info,
                  const gz::sim::EntityComponentManager &_ecm) override;

  /**
   * @brief Get state information for all links in a model
   * @param model_name Name of the model to query
   * @return Map of link names to their GZ_state structures
   * @details Returns a copy of state data. For performance-critical code,
   *          use copy_state_info_fast() or for_each_state_fast() instead.
   */
  [[nodiscard]] std::unordered_map<std::string, GZ_state> state_info(std::string_view model_name);

  /**
   * @brief Fast zero-copy state extraction (const method)
   * @param model_name Name of the model to query
   * @param out Output map to populate with state data
   * @return true if model found and states copied, false otherwise
   * @details Uses fast_link_slots for direct access to state data without
   *          intermediate copies. Preferred for high-frequency state queries.
   * @note Output map is cleared before population
   */
  [[nodiscard]] bool copy_state_info_fast(std::string_view model_name,
                                          std::unordered_map<std::string, GZ_state,
                                                             std::hash<std::string>, std::equal_to<std::string>,
                                                             Eigen::aligned_allocator<std::pair<const std::string, GZ_state>>> &out) const;

  /**
   * @brief Iterate over all link states with a callback
   * @param model_name Name of the model to query
   * @param fn Callback function: void(const string& link_name, const GZ_state& state)
   * @return true if model found and callback invoked, false otherwise
   * @details Most efficient for processing states without creating intermediate
   *          data structures. Callback is invoked for each link.
   */
  [[nodiscard]] bool for_each_state_fast(std::string_view model_name,
                                         const std::function<void(const std::string &, const GZ_state &)> &fn) const;

  /**
   * @brief Reset model position and orientation
   * @param model_name Name of the model to reset
   * @param position New position vector [x, y, z] in world frame (meters)
   * @param orientation New orientation [roll, pitch, yaw] in radians
   * @details Queues a pose and command-state reset for the next simulation
   *          updates. DRLServer drives the required stabilization steps.
   */
  void reset_pos(std::string model_name, Eigen::Vector3d &position, Eigen::Vector3d &orientation);

  void respawn_model(std::string model_name, Eigen::Vector3d &position, Eigen::Vector3d &orientation);

  /**
   * @brief Set rotor velocities for a specific link
   * @param model_name Name of the model containing the link
   * @param link_name Name of the link with rotor actuators
   * @param cmd Vector of rotor velocities (rad/s), one per rotor
   * @details Applies velocity commands to MulticopterVelocityControl component.
   *          Number of elements must match the number of rotors in the link.
   * @note Command is applied immediately to ECM (apply-now architecture)
   */
  void set_rotor_velocity_cmd(std::string model_name, std::string link_name, Eigen::VectorXd &cmd);
  /**
   * @brief Set rotor velocities for a specific link
   * @param model_name Name of the model containing the link
   * @param link_name Name of the link with rotor actuators
   * @param cmd Vector of rotor velocities (rad/s), one per rotor
   * @details Applies velocity commands to MulticopterVelocityControl component.
   *          Number of elements must match the number of rotors in the link.
   * @note Command is applied immediately to ECM (apply-now architecture)
   */
  void set_rotor_velocity_cmd(std::string model_name, std::string link_name, Eigen::VectorXd &&cmd);

  /**
   * @param model_name Name of the model
   * @param base_link Canonical link for the model
   * @param link_names Vector of the link names of each rotor
   * @param turning_dir Vector of turning directions
   * @param cmd Vector of rotor thrusts (N), one per rotor
   * @param ktau Rotor torque constant, such that tau_z = ktau*T
   * @details Applies velocity commands to MulticopterVelocityControl component.
   *          Number of elements must match the number of rotors in the link.
   * @note Command is applied immediately to ECM (apply-now architecture)
   */
  void set_srt_cmd(std::string model_name, std::string base_link, const std::vector<std::string> &link_names,
                   const std::vector<int> &turning_dir, Eigen::VectorXd &cmd, double ktau);
  /**
   * @param model_name Name of the model
   * @param base_link Canonical link for the model
   * @param link_names Vector of the link names of each rotor
   * @param turning_dir Vector of turning directions
   * @param cmd Vector of rotor thrusts (N), one per rotor
   * @param ktau Rotor torque constant, such that tau_z = ktau*T
   * @details Applies velocity commands to MulticopterVelocityControl component.
   *          Number of elements must match the number of rotors in the link.
   * @note Command is applied immediately to ECM (apply-now architecture)
   */
  void set_srt_cmd(std::string model_name, std::string base_link, const std::vector<std::string> &link_names,
                   const std::vector<int> &turning_dir, Eigen::VectorXd &&cmd, double ktau);

  /**
   * @brief Set linear velocity command for a specific link
   * @param model_name Name of the model containing the link
   * @param link_name Name of the link to control
   * @param cmd Linear velocity [vx, vy, vz] in body frame (m/s)
   * @details Applies velocity command via LinearVelocityCmd component.
   * @note This is a kinematic controller - may violate dynamics constraints
   */
  void set_velocity_cmd(std::string model_name, std::string link_name, Eigen::Vector3d &cmd);
  /**
   * @brief Set linear velocity command for a specific link
   * @param model_name Name of the model containing the link
   * @param link_name Name of the link to control
   * @param cmd Linear velocity [vx, vy, vz] in body frame (m/s)
   * @details Applies velocity command via VelocityCmd component.
   *          The command is expected to be in link-frame (a.k.a body-fixed frame)
   *          Physics engine will compute forces to achieve this velocity.
   * @note This is a kinematic controller - may violate dynamics constraints
   */
  void set_velocity_cmd(std::string model_name, std::string link_name, Eigen::Vector3d &&cmd);

  /**
   * @brief Set angular velocity command for a specific link
   * @param model_name Name of the model containing the link
   * @param link_name Name of the link to control
   * @param cmd Angular velocity [wx, wy, wz] in body frame (rad/s)
   * @details Applies angular velocity via AngularVelocityCmd component.
   *          The command is expected to be in link-frame (a.k.a body-fixed frame)
   *          Physics engine will compute torques to achieve this velocity.
   * @note This is a kinematic controller - may violate dynamics constraints
   */
  void set_angular_velocity_cmd(std::string model_name, std::string link_name, Eigen::Vector3d &cmd);
  /**
   * @brief Set angular velocity command for a specific link
   * @param model_name Name of the model containing the link
   * @param link_name Name of the link to control
   * @param cmd Angular velocity [wx, wy, wz] in body frame (rad/s)
   * @details Applies angular velocity via AngularVelocityCmd component.
   *          Physics engine will compute torques to achieve this velocity.
   * @note This is a kinematic controller - may violate dynamics constraints
   */
  void set_angular_velocity_cmd(std::string model_name, std::string link_name, Eigen::Vector3d &&cmd);

  /**
   * @brief Set Ackermann steering velocity command
   * @param model_name Name of the model containing the link
   * @param link_name Name of the link to control
   * @param cmd [linear_velocity (m/s), steering_angle (rad)]
   * @details For wheeled vehicles with Ackermann steering geometry.
   *          cmd[0] = forward/backward velocity, cmd[1] = steering angle
   */
  void set_ackermann_velocity_cmd(std::string model_name, std::string link_name, Eigen::Vector2d &cmd);
  /**
   * @brief Set Ackermann steering velocity command
   * @param model_name Name of the model containing the link
   * @param link_name Name of the link to control
   * @param cmd [linear_velocity (m/s), steering_angle (rad)]
   * @details For wheeled vehicles with Ackermann steering geometry.
   *          cmd[0] = forward/backward velocity, cmd[1] = steering angle
   */
  void set_ackermann_velocity_cmd(std::string model_name, std::string link_name, Eigen::Vector2d &&cmd);

  /**
   * @brief Set joint position command
   * @param model_name Name of the model containing the joint
   * @param joint_name Name of the joint to control
   * @param cmd Joint position target (units depend on joint type)
   * @details For revolute joints: radians, for prismatic joints: meters.
   *          Uses JointPositionController to servo to target position.
   */
  void set_joint_position_cmd(std::string model_name, std::string joint_name, Eigen::Vector3d &&cmd);
  /**
   * @brief Set joint position command
   * @param model_name Name of the model containing the joint
   * @param joint_name Name of the joint to control
   * @param cmd Joint position target (units depend on joint type)
   * @details For revolute joints: radians, for prismatic joints: meters.
   *          Uses JointPositionController to servo to target position.
   */
  void set_joint_position_cmd(std::string model_name, std::string joint_name, Eigen::Vector3d &cmd);

  /**
   * @brief Set link mass (updates SDF cache, requires reset_pos to apply)
   * @param model_name Name of the model containing the link
   * @param link_name Name of the link to modify
   * @param mass New mass value (kg)
   * @details Modifies the SDF cache only. Changes take effect after next
   *          respawn_model() call which respawns the entity with new parameters.
   * @warning Does not affect running simulation until respawn_model() is called
   */
  void set_mass(std::string model_name, std::string link_name, double mass);

  /**
   * @brief Set link inertia tensor (updates SDF cache, requires reset to apply)
   * @param model_name Name of the model containing the link
   * @param link_name Name of the link to modify
   * @param inertia 3x3 inertia tensor (kg·m²) in link frame
   * @details Modifies the SDF cache only. Changes take effect after next
   *          respawn_model() call which respawns the entity with new parameters.
   * @warning Does not affect running simulation until respawn_model() is called
   */
  void set_inertia(std::string model_name, std::string link_name, Eigen::Matrix3d &inertia);
  /**
   * @brief Set link inertia tensor (updates SDF cache, requires reset to apply)
   * @param model_name Name of the model containing the link
   * @param link_name Name of the link to modify
   * @param inertia 3x3 inertia tensor (kg·m²) in link frame
   * @details Modifies the SDF cache only. Changes take effect after next
   *          respawn_model() call which respawns the entity with new parameters.
   * @warning Does not affect running simulation until respawn_model() is called
   */
  void set_inertia(std::string model_name, std::string link_name, Eigen::Matrix3d &&inertia);

  /**
   * @brief Get link inertia
   * @param model_name Name of the model containing the link
   * @param link_name Name of the link to query
   * @return 3x3 inertia tensor (kg·m²) in link frame
   */
  [[nodiscard]] Eigen::Matrix3d get_inertia(std::string model_name, std::string link_name);

  /**
   * @brief Get link mass
   * @param model_name Name of the model containing the link
   * @param link_name Name of the link to query
   * @return Mass value (kg)
   */
  [[nodiscard]] double get_mass(std::string model_name, std::string link_name);
  /**
   * @brief Set link center of mass (updates SDF cache, requires reset to apply)
   * @param model_name Name of the model containing the link
   * @param link_name Name of the link to modify
   * @param params Rotor parameters structure
   * @details Modifies the SDF cache only. Changes take effect after next
   *          respawn_model() call which respawns the entity with new parameters.
   * @warning Does not affect running simulation until respawn_model() is called
   */
  void set_rotor_parameters(std::string model_name, const RotorParameters &params);

  /**
   * @brief Get link rotor parameters
   * @param model_name Name of the model containing the MultiRotorPlugin
   * @return Rotor parameters structure
   */
  [[nodiscard]] RotorParameters get_rotor_parameters(std::string model_name);

  /**
   * @brief Get rotor velocity allocation matrix for UAV model, only usable if model has a MultiRotorPlugin
   * @param model_name Name of the model containing the MultiRotorPlugin
   * @return Rotor allocation matrix (4 x num_rotors) )
   * @details Rows: [thrust; roll_moment; pitch_moment; yaw_moment]
   * The allocation matrix maps rotor thrusts to total forces/moments.
   */
  [[nodiscard]] Eigen::MatrixXd get_rotor_thrust_allocation_matrix(std::string model_name);

  /**
   * @brief Get the rotor thrust allocation matrix for UAV model
   * @param model_name Name of the model
   * @param base_link Base link of the model
   * @param rotor_links Vector containing rotor link names
   * @param turning_dir Vector containing turning direction
   * @param ktau Rotor torque constant
   */
  [[nodiscard]] Eigen::MatrixXd get_rotor_thrust_allocation_matrix(std::string model_name,
                                                                   const std::vector<std::string> &rotor_links, const std::vector<int> &turning_dir, double ktau);

  /**
   * @brief Request contact data collection for a model
   * @param model_name Name of the model to enable contact sensing
   * @details Adds ContactSensorData components to all links in the model.
   *          Contact data can then be retrieved via get_contacts().
   * @note Has performance impact - only enable for models that need it
   */
  void request_contact_data(std::string_view model_name);

  /**
   * @brief Get contact data for all links in a model
   * @param model_name Name of the model to query
   * @return Map of link names to vectors of contact messages
   * @details Returns all collision contacts detected for each link.
   *          Empty vector for links with no contacts.
   * @note Must call request_contact_data() first to enable contact sensing
   */
  [[nodiscard]] std::unordered_map<std::string, std::vector<gz::msgs::Contacts>> get_contacts(std::string_view model_name);

  /**
   * @brief Get reference to SDF root object
   * @return Reference to the parsed SDF root
   * @warning Direct SDF manipulation can break simulation state
   */
  [[nodiscard]] sdf::Root &sdf_root_() { return this->sdf_root; }

  /**
   * @brief Get reference to SDF world object
   * @return Reference to the SDF world description
   * @warning Direct SDF manipulation can break simulation state
   */
  [[nodiscard]] sdf::World &sdf_world_() { return this->world_sdf_obj; }

  /**
   * @brief Get simulation time step size
   * @return Time step in seconds (typically 0.001 for 1ms steps)
   */
  [[nodiscard]] double step_size_() const { return this->step_size; }

  /**
   * @brief Get raw pointer to Entity Component Manager
   * @return Pointer to ECM (cached during Configure)
   * @warning Use with extreme caution - bypasses safety checks
   */
  [[nodiscard]] gz::sim::EntityComponentManager *ecm_() const { return this->_configure_ecm; }

  /**
   * @brief Get reference to SDF entity creator
   * @return Unique pointer to entity creator (for spawning new entities)
   * @warning Direct entity creation can corrupt internal state
   */
  [[nodiscard]] std::unique_ptr<gz::sim::SdfEntityCreator> &sdf_creator_() { return this->sdf_creator; }

  /**
   * @brief Get world entity ID
   * @return Entity ID of the simulation world
   */
  [[nodiscard]] gz::sim::Entity world_entity_() const { return this->world_entity; }

  /**
   * @brief Apply wrench (force + torque) to a link
   * @param model_name Name of the model containing the link
   * @param link_name Name of the link to apply wrench to
   * @param force Force vector [fx, fy, fz] in world frame (N)
   * @param moments Torque vector [tx, ty, tz] in world frame (N·m)
   * @details Applies external wrench directly to link via ExternalWorldWrenchCmd.
   *          Force and torque are applied at the link's center of mass.
   * @note Commands are applied immediately (apply-now architecture)
   */
  void set_wrench(std::string model_name, std::string link_name, Eigen::Vector3d &&force, Eigen::Vector3d &&moments);
  /**
   * @brief Apply wrench (force + torque) to a link
   * @param model_name Name of the model containing the link
   * @param link_name Name of the link to apply wrench to
   * @param force Force vector [fx, fy, fz] in world frame (N)
   * @param moments Torque vector [tx, ty, tz] in world frame (N·m)
   * @details Applies external wrench directly to link via ExternalWorldWrenchCmd.
   *          Force and torque are applied at the link's center of mass.
   * @note Commands are applied immediately (apply-now architecture)
   */
  void set_wrench(std::string model_name, std::string link_name, Eigen::Vector3d &force, Eigen::Vector3d &moments);

private:
  using LinkMap = std::unordered_map<std::string, gz::sim::Link>;
  using StateMap = std::unordered_map<
      std::string,
      GZ_state,
      std::hash<std::string>,
      std::equal_to<std::string>,
      Eigen::aligned_allocator<std::pair<const std::string, GZ_state>>>;
  /**
   * @brief A struct for holding Link and associated state, and the contacts
   *
   */
  struct LinkSlot
  {
    gz::sim::Link link;                        ///< gz::sim::Link
    GZ_state *state;                           ///< State associated with the link
    std::vector<gz::msgs::Contacts> *contacts; ///< Contacts if requested
  };

  // Cached names and handles
  std::vector<std::string> model_names;                                   ///< Model names managed by this DRLServer
  std::unordered_map<std::string, gz::sim::Model> models;                 ///< Model objects for the model_name
  std::unordered_map<std::string, LinkMap> links;                         ///< Link objects for each link. The links is indexed by model name, and the LinkMap indexed by link name
  std::unordered_map<std::string, StateMap> pose_datas;                   ///< Pose info for each link. Indexing is performed similar to links
  std::unordered_map<std::string, std::vector<LinkSlot>> fast_link_slots; ///< Fast access link slots

  // Reset pipelines (kept)
  std::vector<std::pair<std::string, Eigen::Matrix<double, 6, 1>>> reset_queue;        ///< Models that needs pose to be reset should be pushed here
  std::vector<std::pair<std::string, Eigen::Matrix<double, 6, 1>>> post_reset_queue;   ///< Reset takes place in two iterations, this should not be directly modified
  std::vector<std::pair<std::string, Eigen::Matrix<double, 6, 1>>> respawn_queue;      ///< Models that needs to respawned should be pushed here
  std::vector<std::pair<std::string, Eigen::Matrix<double, 6, 1>>> post_respawn_queue; ///< Respawn takes place in two iterations, this should not be directly modified
  // Reused buffers
  gz::msgs::Actuators actuator_cmd_msg; ///< Base template msg for actuator msg, prevents re-alloc

  std::unordered_map<std::string,
                     std::unordered_map<std::string, std::vector<gz::msgs::Contacts>>>
      contacts; ///< Contacts msgs map [indexed as model_name->link_name->contacts]

  DRLServer *server_ptr{};                                         ///< Raw DRLServer ptr, do not manipulate this!
  std::unique_ptr<gz::sim::SdfEntityCreator> sdf_creator{nullptr}; ///< Entity creator raw ptr
  sdf::Root sdf_root;                                              ///< The root sdf loaded during construction. Should not be changed once DRLServer is constructed
  std::unordered_map<std::string, sdf::Model> model_sdf_objs;      ///< SDF for the models managed by DRLServer
  sdf::World world_sdf_obj;                                        ///< SDF for the world associated with the root sdf
  gz::sim::Entity world_entity{};                                  ///< Entity of the world (is constant throughout sim)
  gz::sim::Entity main_parent_entity{};                            ///< Parent for all models. This is usually world_entity.
  gz::msgs::SerializedState ecm_init_state;                        ///< Initial state for ECM, stored for future API development
  bool has_ecm_init_state{false};                                  ///< Flag indicating ecm_init_state was configured
  gz::sim::EntityComponentManager *_configure_ecm{};               ///< Direct access to ECM (needs to be very careful)
  bool skip_post_update{false};                                    ///< Flag specifying to skip postupdate (legacy)
  double step_size{0};                                             ///< Physics step size

public:
  std::unordered_map<std::string, std::unique_ptr<OrnsteinUhlenbeckMultivar<19>>> op_procs; ///< OP processes for each model

private:
  /**
   * @brief Internal initialziation helper. This needs to be called during construction.
   * It is also essential to call this after respawning model to ensure all caches are updated.
   * The entity will change during respawn
   *
   * @param _ecm  EntityComponentManager
   */
  void init_method(gz::sim::EntityComponentManager &_ecm);
  /**
   * @brief Recursively finds nested models. In SDF you can have nested models.
   * Nested models can be specified to DRLServer as model1::nested_model1, model::nested_model2
   *
   * @param nested_names A vector that splits the nested name , e.g. model1::nested_model1 is split
   * as [model1, nested_model1]
   * @param parent_model A parent model sim::Model obj, e.g. Model(model1) will be parent for Model(nested_model1)
   * @param idx Index of nested model currently iterated
   * @param _ecm  ECM
   * @return gz::sim::Model the sdf associated with the nested model
   */
  gz::sim::Model recursive_model_finder(const std::vector<std::string> &nested_names, const gz::sim::Model &parent_model, int idx, gz::sim::EntityComponentManager &_ecm);
  /**
   * @brief Splits nested model names, e.g. m1::m2::m3 is split as [m1, m2, m3]
   *
   * @param s nested model name
   * @return std::vector<std::string> split names
   */
  std::vector<std::string> model_name_splitter(const std::string &s);

  /**
   * @brief Get vector parameter from SDF element
   * @tparam T type of data to extract from sdf
   * @param _sdf SDF element pointer
   * @param name Name of the parameter to extract
   * @param param Output vector to populate
   * @param def Default vector value if parameter not found
   * @return true if parameter found and extracted, false if default used
   */
  template <typename T>
  bool get_vector_from_sdf(const sdf::ElementPtr _sdf,
                           std::string name,
                           std::vector<T> &param,
                           std::vector<T> &def)
  {
    std::string val;
    const bool ret = _sdf->Get(name, val, std::string{});
    if (!ret)
    {
      param = def;
      return false;
    }
    auto res = split_to_vec<T>(val);
    param.assign(res.begin(), res.end());
    return true;
  }

  /**
   * @brief Split string into vector of type T
   * @tparam T type of object to cast each item
   * @param input Input string to split
   * @return std::vector<T>  Vector of type T containing parsed values
   */
  template <typename T>
  std::vector<T> split_to_vec(const std::string &input)
  {
    std::vector<T> result;
    std::istringstream iss(input);
    T value;
    while (iss >> value)
      result.push_back(value);
    return result;
  }
};

/**
 * @brief Main interface for controlling Gazebo simulations in RL environments
 *
 * This class provides a simplified API for Deep reinforcement learning,
 * wrapping the Gazebo server and DRLHelperSystem with convenient methods.
 *
 */
class DRLServer
{
private:
  /**
   * @brief FirstOrder rate limiter
   *
   * @tparam T Eigen::Vector type
   */
  template <typename T>
  class FirstOrderFilter
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /**
     * @brief Construct a new First Order Filter object
     *
     * @param timeConstantUp time constant for rising cmd
     * @param timeConstantDown time constant for falling cmd
     * @param initial initial state
     */
    FirstOrderFilter(double timeConstantUp, double timeConstantDown, const T &initial)
        : timeConstantUp_(timeConstantUp),
          timeConstantDown_(timeConstantDown),
          previousState_(initial),
          initial_state_(initial) {}

    /**
     * @brief Reset the filter to initial state
     *
     */
    void Reset()
    {
      previousState_ = initial_state_;
    }
    /**
     * @brief Update the filter with new cmd
     *
     * @param input new cmd
     * @param dt sampling time
     * @return T Eigen::Vector type after filtering
     */
    T UpdateFilter(const T &input, double dt)
    {
      using Scalar = typename T::Scalar;
      T is_up = (input.array() > previousState_.array()).template cast<Scalar>();
      T is_down = (input.array() <= previousState_.array()).template cast<Scalar>();

      T tau = is_up * timeConstantUp_ + is_down * timeConstantDown_;

      tau = tau.cwiseMax(1e-9);
      T alpha = (-dt / tau.array()).exp().matrix();
      previousState_ = (alpha.array() * previousState_.array() +
                        (1.0 - alpha.array()) * input.array())
                           .matrix();

      return previousState_;
    }

  private:
    double timeConstantUp_;   ///< rising time constant
    double timeConstantDown_; ///< falling time constant
    const T initial_state_;   ///< intiial state
    T previousState_;         ///< last state for smoothing
  };

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /** @brief Environment setup flag (singleton pattern for partition config) */
  static inline std::atomic<bool> env_set_{false};

  /**
   * @brief Constructor for standard DRL server
   * @param partition Gazebo partition name (for process isolation)
   * @param sdf_file Path to SDF world file
   * @param model_names List of model names to track/control
   * @param enable_sensors Whether to enable sensor interfaces
   * @param config Server configuration (update rate, rendering, etc.)
   * @throws std::runtime_error if SDF file cannot be loaded
   */
  DRLServer(const std::string &partition, const std::string &sdf_file,
            const std::vector<std::string> &model_names, bool enable_sensors,
            const DRLServerConfig &config = Default_DRLServer_Config);

  /**
   * @brief Set environment variable for Gazebo partition
   * @param partition Partition name to set in GZ_PARTITION env var
   * @details Must be called before constructing DRLServer if using custom partitions
   */
  static void set_env(std::string partition);

  /**
   * @brief Initialize server internals (called by constructor)
   *  Sets up the Gazebo server, registers DRLHelperSystem plugin,
   *          configures sensors, and initializes controllers
   */
  void init_method();

  /** @brief Destructor - cleanly shuts down simulation server */
  ~DRLServer();

  /**
   * @brief Step simulation forward by one time step
   * Advances physics by step_size() seconds. Typically ~1ms per step.
   *          Commands issued before this call take effect during the step.
   */
  void run_once();

  /**
   * @brief Step simulation forward by N time steps
   * @param N Number of simulation steps to execute
   * @details Equivalent to calling run_once() N times, but more efficient.
   *          Total simulated time = N * step_size() seconds.
   */
  void run_N(int N);

  /**
   * @brief Get simulation time step size
   * @return Time step duration in seconds (e.g., 0.001 for 1ms steps)
   */
  [[nodiscard]] double step_size();

  /**
   * @brief Get current state of all links in a model. Delegates to DRLHelperSystem::state_info().
   * Includes low variance noise. Returns empty map if model not found.
   * @param model_name Name of the model to query
   * @return Map of link names to GZ_state structures containing pose, velocity, etc.
   */
  [[nodiscard]] std::unordered_map<std::string, GZ_state> state_info(std::string model_name);

  /**
   * @brief Reset model to a new position and orientation
   * Triggers entity teleportation.
   * Also resets velocities and wrench to zero.
   * @param model_name Name of the model to reset
   * @param position New position [x, y, z] in world frame (meters)
   * @param orientation New orientation [roll, pitch, yaw] in radians (default: zero)
   * @note Runs three stabilization physics steps before returning.
   */
  void reset_pos(std::string model_name, Eigen::Vector3d &position, Eigen::Vector3d &orientation = Zero_Ori);
  /**
   * @brief Reset model to a new position and orientation
   * Triggers entity teleportation.
   * Also resets velocities and wrench to zero.
   * @param model_name Name of the model to reset
   * @param position New position [x, y, z] in world frame (meters)
   * @param orientation New orientation [roll, pitch, yaw] in radians (default: zero)
   * @note Runs three stabilization physics steps before returning.
   */
  void reset_pos(std::string model_name, Eigen::Vector3d &&position, Eigen::Vector3d &&orientation);
  /**
   * @brief Respawn model SDF to a new position and orientation
   * Triggers entity removal and respawn. Model is recreated with new pose.
   * Also resets velocities and wrench to zero and reconfigures sensors.
   * @param model_name Name of the model to reset
   * @param position New position [x, y, z] in world frame (meters)
   * @param orientation New orientation [roll, pitch, yaw] in radians (default: zero)
   * @note Runs three stabilization physics steps before returning.
   * @warning Expensive operation - avoid calling every time step
   */
  void respawn_model(std::string model_name, Eigen::Vector3d &position, Eigen::Vector3d &orientation);
  /**
   * @brief Respawn model SDF to a new position and orientation
   * Triggers entity removal and respawn. Model is recreated with new pose.
   * Also resets velocities and wrench to zero and reconfigures sensors.
   * @param model_name Name of the model to reset
   * @param position New position [x, y, z] in world frame (meters)
   * @param orientation New orientation [roll, pitch, yaw] in radians (default: zero)
   * @note Runs three stabilization physics steps before returning.
   * @warning Expensive operation - avoid calling every time step
   */
  void respawn_model(std::string model_name, Eigen::Vector3d &&position, Eigen::Vector3d &&orientation);

  /**
   * @brief Set the rotor velocity commands for a link in a model. The rotor velocity (SRV)
   * is managed by the MultiRotorPlugin. Ensure this plugin is attached to the model or the link.
   * If MultiRotorPlugin doesn't exist, then this is a no-op call
   *
   * @param model_name Name of the model with MultiRotorPlugin
   * @param link_name  Name of the link in the model with MultuRotorPlugin
   * @param cmd Rotor velocity cmd
   */
  void set_rotor_velocity_cmd(std::string model_name, std::string link_name, Eigen::VectorXd &cmd);
  /**
   * @brief Set the rotor velocity commands for a link in a model. The rotor velocity (SRV)
   * is managed by the MultiRotorPlugin. Ensure this plugin is attached to the model or the link.
   * If MultiRotorPlugin doesn't exist, then this is a no-op call
   *
   * @param model_name Name of the model with MultiRotorPlugin
   * @param link_name  Name of the link in the model with MultuRotorPlugin
   * @param cmd Rotor velocity cmd
   */
  void set_rotor_velocity_cmd(std::string model_name, std::string link_name, Eigen::VectorXd &&cmd);
  /**
   * @brief Set the ctbr cmd for a link in a model. This overload directly applies the
   * angular velocity cmd (rate-limiter applied if a time constant is set for this {model_name, link_name} pair
   * using DRLServer::set_ctbr_rate_limiter_time_constants)
   *
   * @param model_name Name of the model
   * @param link_name Name of the link for which the ctbr cmd is to be set
   * @param cmd  CTBR cmd
   */
  void set_ctbr_cmd(std::string model_name, std::string link_name, Eigen::VectorXd &cmd);
  /**
   * @brief Set the ctbr cmd for a link in a model. This overload directly applies the
   * angular velocity cmd (rate-limiter applied if a time constant is set for this {model_name, link_name} pair
   * using DRLServer::set_ctbr_rate_limiter_time_constants)
   *
   * @param model_name Name of the model
   * @param link_name Name of the link for which the ctbr cmd is to be set
   * @param cmd  CTBR cmd
   */
  void set_ctbr_cmd(std::string model_name, std::string link_name, Eigen::VectorXd &&cmd);

  /**
   * @brief Set the ctbr cmd for a link in a model. This overload has an addition inner-loop controller.
   * The actual control here is the CTBT, whose torque is obtained as Torque =  kp*(omega_cmd - omega) + kd*(-alpha)
   * (rate-limiter applied if a time constant is set for this {model_name, link_name} pair
   * using DRLServer::set_ctbr_rate_limiter_time_constants)
   *
   * @param model_name Model name
   * @param link_name Link name
   * @param cmd CTBR cmd
   * @param kp kp gain for inner-loop
   * @param kd kd gain for inner-loop
   */
  void set_ctbr_cmd(std::string model_name, std::string link_name, Eigen::VectorXd &cmd, const Eigen::VectorXd &kp, const Eigen::VectorXd &kd);
  /**
   * @brief Set the ctbr cmd for a link in a model. This overload has an addition inner-loop controller.
   * The actual control here is the CTBT, whose torque is obtained as Torque =  kp*(omega_cmd - omega) + kd*(-alpha)
   * (rate-limiter applied if a time constant is set for this {model_name, link_name} pair
   * using DRLServer::set_ctbr_rate_limiter_time_constants)
   *
   * @param model_name Model name
   * @param link_name Link name
   * @param cmd CTBR cmd
   * @param kp kp gain for inner-loop
   * @param kd kd gain for inner-loop
   */
  void set_ctbr_cmd(std::string model_name, std::string link_name, Eigen::VectorXd &&cmd, const Eigen::VectorXd &kp, const Eigen::VectorXd &kd);

  /**
   * @brief Set the ctbt cmd for a model. The ctbt commands are converted to world frame
   * force and torques, then set via DRLServer::set_wrench method.
   *
   * @param model_name Name of the model
   * @param link_name Name of the link
   * @param cmd CTBT cmd
   */
  void set_ctbt_cmd(std::string model_name, std::string link_name, Eigen::VectorXd &cmd);
  /**
   * @brief Set the ctbt cmd for a model. The ctbt commands are converted to world frame
   * force and torques, then set via DRLServer::set_wrench method.
   *
   * @param model_name Name of the model
   * @param link_name Name of the link
   * @param cmd CTBT cmd
   */
  void set_ctbt_cmd(std::string model_name, std::string link_name, Eigen::VectorXd &&cmd);

  /**
   * @brief Set the srt cmd for a model. The SRT control requries the tuning directions of each rotor,
   * and the motor torque constant. The yaw torque generated by each rotor is defined as
   * -turning_dir*ktau*Thrust
   *
   * @param model_name Name of the model
   * @param base_link Name of the base link in the model
   * @param link_names Rotor link names
   * @param turning_dir Turning direction for each rotor
   * @param cmd SRT command
   * @param ktau motor torque constant
   */
  void set_srt_cmd(std::string model_name, std::string base_link, const std::vector<std::string> &link_names, const std::vector<int> &turning_dir,
                   Eigen::VectorXd &cmd, double ktau);
  /**
   * @brief Set the srt cmd for a model. The SRT control requries the tuning directions of each rotor,
   * and the motor torque constant. The yaw torque generated by each rotor is defined as
   * -turning_dir*ktau*Thrust
   *
   * @param model_name Name of the model
   * @param base_link Name of the base link in the model
   * @param link_names Rotor link names
   * @param turning_dir Turning direction for each rotor
   * @param cmd SRT command
   * @param ktau motor torque constant
   */
  void set_srt_cmd(std::string model_name, std::string base_link, const std::vector<std::string> &link_names, const std::vector<int> &turning_dir,
                   Eigen::VectorXd &&cmd, double ktau);

  /**
   * @brief Set the velocity cmd for a link in a model. This is a kinematic control.
   * The object will immediately achieve the commanded velocity.
   *
   * @param model_name  Name of the model
   * @param link_name Name of the link in the model
   * @param cmd Velocity command in body frame (m/s)
   */
  void set_velocity_cmd(std::string model_name, std::string link_name, Eigen::Vector3d &cmd);
  /**
   * @brief Set the velocity cmd for a link in a model. This is a kinematic control.
   * The object will immediately achieve the commanded velocity. The command is expected to be in link-frame (a.k.a body-fixed frame)
   *
   * @param model_name  Name of the model
   * @param link_name Name of the link in the model
   * @param cmd Velocity command in body frame (m/s)
   */
  void set_velocity_cmd(std::string model_name, std::string link_name, Eigen::Vector3d &&cmd);

  /**
   * @brief Set the angular velocity cmd  (in body-fixed frame) for a link in a model. This is kinematic control
   * The object will immediately achieve the commanded angular velocity. The command is expected to be in link-frame (a.k.a body-fixed frame)
   *
   * @param model_name Name of the model
   * @param link_name Name of the link in the model
   * @param cmd Body-fixed frame angular velocity cmd
   */
  void set_angular_velocity_cmd(std::string model_name, std::string link_name, Eigen::Vector3d &cmd);
  /**
   * @brief Set the angular velocity cmd  (in body-fixed frame) for a link in a model. This is kinematic control
   * The object will immediately achieve the commanded angular velocity.
   *
   * @param model_name Name of the model
   * @param link_name Name of the link in the model
   * @param cmd Body-fixed frame angular velocity cmd
   */
  void set_angular_velocity_cmd(std::string model_name, std::string link_name, Eigen::Vector3d &&cmd);

  /**
   * @brief Set the ackermann velocity cmd for a link in the model
   * The Ackermann cmd involves the linear velocity in x direction and the angular velocity
   * in the yaw axis. This requires the AckermannSteering plugin provided as part of the
   * GzDRL library (a custom implementation)
   *
   * @param model_name Model name
   * @param link_name Link name (chassis link for cars)
   * @param cmd ackermann cmd
   */
  void set_ackermann_velocity_cmd(std::string model_name, std::string link_name, Eigen::Vector2d &cmd);
  /**
   * @brief Set the ackermann velocity cmd for a link in the model
   * The Ackermann cmd involves the linear velocity in x direction and the angular velocity
   * in the yaw axis. This requires the AckermannSteering plugin provided as part of the
   * GzDRL library (a custom implementation)
   *
   * @param model_name Model name
   * @param link_name Link name (chassis link for cars)
   * @param cmd ackermann cmd
   */
  void set_ackermann_velocity_cmd(std::string model_name, std::string link_name, Eigen::Vector2d &&cmd);

  /**
   * @brief Set the joint position cmd for a join in a model. This requires the JointPositionController plugin
   * (a customzied implementation)
   * to be attached to the model or joint.
   * The cmd is 3D, for each axis.
   *
   * @param model_name Name of the model
   * @param joint_name Name of the joint
   * @param cmd Joint angle cmd
   */
  void set_joint_position_cmd(std::string model_name, std::string joint_name, Eigen::Vector3d &&cmd);
  /**
   * @brief Set the joint position cmd for a join in a model. This requires the JointPositionController plugin
   * (a customzied implementation)
   * to be attached to the model or joint.
   * The cmd is 3D, for each axis.
   *
   * @param model_name Name of the model
   * @param joint_name Name of the joint
   * @param cmd Joint angle cmd
   */
  void set_joint_position_cmd(std::string model_name, std::string joint_name, Eigen::Vector3d &cmd);

  /**
   * @brief Set the mass of a link in a model. Used for domain randomization.
   *  Note that changed mass will not be in effect until model is respawned using DRLServer::respawn_model
   *
   * @param model_name Name of the model
   * @param link_name Name of the link
   * @param mass mass to set
   */
  void set_mass(std::string model_name, std::string link_name, double mass);
  /**
   * @brief Set the inertia of a link in a model. Used for domain randomization.
   * Note that changed inertia will not be in effect until model is respawned using DRLServer::respawn_model
   *
   * @param model_name Name of the model
   * @param link_name Name of the link
   * @param inertia Inertia to set
   */
  void set_inertia(std::string model_name, std::string link_name, Eigen::Matrix3d &inertia);
  /**
   * @brief Set the inertia of a link in a model. Used for domain randomization.
   *  Note that changed inertia will not be in effect until model is respawned using DRLServer::respawn_model
   *
   * @param model_name Name of the model
   * @param link_name Name of the link
   * @param inertia Inertia to set
   */
  void set_inertia(std::string model_name, std::string link_name, Eigen::Matrix3d &&inertia);
  /**
   * @brief Get the inertia of a link in a model
   *
   * @param model_name Name of the model
   * @param link_name Name of the link
   * @return Eigen::Matrix3d inertia matrix
   */
  [[nodiscard]] Eigen::Matrix3d get_inertia(std::string model_name, std::string link_name);
  /**
   * @brief Get the mass of a link in a model
   *
   * @param model_name Name of the model
   * @param link_name Name of the link
   * @return double The mass of the link
   */
  [[nodiscard]] double get_mass(std::string model_name, std::string link_name);
  /**
   * @brief Set the rotor parameters for a model. The model must have the MultiRotorPlugin.
   * Otherwise, this is a no-op
   *
   * @param model_name Name of the model with MultiRotorPlugin
   * @param params RotorParameters (see docs)
   */
  void set_rotor_parameters(std::string model_name, const RotorParameters &params);
  /**
   * @brief Get the rotor parameters for a model. The model must have the MultiRotorPlugin.
   * Otherwise, this return default RotorParameters.
   *
   * @param model_name Name of the model with MultiRotorPlugin
   * @return RotorParameters
   */
  [[nodiscard]] RotorParameters get_rotor_parameters(std::string model_name);
  /**
   * @brief Get the rotor thrust allocation matrix. This matrix allocates the CTBT command,
   * thrust and moments as [T, taux, tauy, tauz] -> [T1, T2, T3, ..., TN], which are
   * the single rotor thrusts. This overload only
   * works if the model has the MultiRotorPlugin. Otherwise, it returns an allocation matrix
   * determined based on default RotorParameters struct.
   * This method automatically identifies the motor torque constant, geometry of the UAV
   * based on the rotor link positions from CoG, and the turining directions (some are obtained
   * from MultiRotorPlugin).
   *
   * @param model_name Name of the model with MultiRotorPlugin
   * @return Eigen::MatrixXd  The CTBT -> SRT allocation matrix
   */
  [[nodiscard]] Eigen::MatrixXd get_rotor_thrust_allocation_matrix(std::string model_name);
  /**
   * @brief Get the inverse rotor thrust allocation matrix. This maps SRT -> CTBR
   * See DRLServer::get_rotor_thrust_allocation_matrix for details
   *
   * @param model_name Name of the model with MultiRotorPlugin
   * @return Eigen::MatrixXd The SRT -> CTBT allocation matrix
   */
  [[nodiscard]] Eigen::MatrixXd get_inverse_rotor_thrust_allocation_matrix(std::string model_name);

  /**
   * @brief Get the rotor thrust allocation matrix. This matrix allocates the CTBT command,
   * thrust and moments as [T, taux, tauy, tauz] -> [T1, T2, T3, ..., TN], which are
   * the single rotor thrusts. This overload requires explicit rotor link names, and turning directions,
   * and the motor torque constant. It doesn't require MultiRotorPlugin to be attached.
   * This method automatically geometry of the UAV
   * based on the rotor link positions from CoG.
   *
   * @param model_name  Name of the model
   * @param rotor_links Names of the rotor links
   * @param turning_dir Turning directions for each rotor
   * @param ktau Motor torque constant
   * @return Eigen::MatrixXd Allocation matrix that maps CTBT -> SRT
   */
  [[nodiscard]] Eigen::MatrixXd get_rotor_thrust_allocation_matrix(std::string model_name,
                                                                   const std::vector<std::string> &rotor_links, const std::vector<int> &turning_dir, double ktau);
  /**
   * @brief Get the inverse rotor thrust allocation matrix. see DRLServer::get_rotor_thrust_allocation_matrix
   * with this specific overload for details.
   *
   * @param model_name Name of the model
   * @param rotor_links Rotor link names
   * @param turning_dir Turning directions
   * @param ktau Motor Torque constant
   * @return Eigen::MatrixXd Allocation matrix that maps SRT -> CTBR
   */
  [[nodiscard]] Eigen::MatrixXd get_inverse_rotor_thrust_allocation_matrix(std::string model_name,
                                                                           const std::vector<std::string> &rotor_links, const std::vector<int> &turning_dir, double ktau);

  /**
   * @brief Get thrust/moments to rotor velocity mapping function for UAV model.
   * The returned function takes an Eigen::Vector4d input:
   * [total_thrust; roll_moment; pitch_moment; yaw_moment
   * @param model_name Name of the model containing the MultiRotorPlugin
   * @return Lambda function that maps [thrust; roll_moment; pitch_moment; yaw_moment] to rotor velocities
   */
  [[nodiscard]] std::function<Eigen::VectorXd(const Eigen::Vector4d &)> get_thrust_moment_to_rotor_velocity_mapping_function(std::string model_name);

  /**
   * @brief Get thrust/moments to rotor thrust mapping function for UAV model.The returned function takes an Eigen::Vector4d input:
   *  [total_thrust; roll_moment; pitch_moment; yaw_moment
   * @param model_name Name of the model containing the MultiRotorPlugin
   * @return Lambda function that maps [thrust; roll_moment; pitch_moment; yaw_moment] to rotor thrusts
   */
  [[nodiscard]] std::function<Eigen::VectorXd(const Eigen::Vector4d &)> get_thrust_moment_to_rotor_thrust_mapping_function(std::string model_name);

  /**
   * @brief Get CTBT to rotor thrust mapping function for UAV model
   * @param model_name Name of the model
   * @param link_name Vector containing rotor links
   * @param tuning_dir Turning directions of rotors
   * @param ktau Rotor torque constant
   * @return Lambda function that maps [thrust; roll_moment; pitch_moment; yaw_moment] to rotor thrusts
   * @details The returned function takes an Eigen::Vector4d input:
   *          [total_thrust; roll_moment; pitch_moment; yaw_moment
   */
  [[nodiscard]] std::function<Eigen::VectorXd(const Eigen::Vector4d &)> get_thrust_moment_to_rotor_thrust_mapping_function(std::string model_name,
                                                                                                                           const std::vector<std::string> &link_names, std::vector<int> &turning_dir, double ktau);

  /**
   * @brief Get the contacts for a model
   *
   * @param model_name Name of the model
   * @return std::unordered_map<std::string, std::vector<gz::msgs::Contacts>> The contacts for all the links in the model
   */
  [[nodiscard]] std::unordered_map<std::string, std::vector<gz::msgs::Contacts>> get_contacts(std::string model_name);

  /**
   * @brief Set the wrench (expressed in world frame) to a link in a model
   *
   * @param model_name Name of the model
   * @param link_name Name of the link in the model
   * @param force Force in world frame
   * @param moments Moments in world frame
   */
  void set_wrench(std::string model_name, std::string link_name, Eigen::Vector3d &&force, Eigen::Vector3d &&moments);
  /**
   * @brief Set the wrench (expressed in world frame) to a link in a model
   *
   * @param model_name Name of the model
   * @param link_name Name of the link in the model
   * @param force Force in world frame
   * @param moments Moments in world frame
   */
  void set_wrench(std::string model_name, std::string link_name, Eigen::Vector3d &force, Eigen::Vector3d &moments);

  /** Type alias for (desired_state, current_state) tuple */
  using ControlTuple = std::tuple<Stated, Stated>;

  /** Type alias for map of link names to control state tuples */
  using LinkControlMap = std::unordered_map<std::string,
                                            ControlTuple,
                                            std::hash<std::string>, std::equal_to<std::string>,
                                            Eigen::aligned_allocator<std::pair<const std::string, ControlTuple>>>;

  /** Map of (model_name, link_name) pairs to their controller instances */
  std::unordered_map<std::pair<std::string, std::string>, std::shared_ptr<UAVController>, PairHash, PairEq> link_controllers; ///< controllers attached to a link. Indexed as {model_name, link_name} pair

  /** Map of lambdas that map Thrust and Moments to rotor velocity for each model */
  std::unordered_map<std::pair<std::string, std::string>, std::function<Eigen::VectorXd(const Eigen::Vector4d &)>, PairHash, PairEq> thrust_moments_to_rotor_velocity_map; ///< CTBT to SRV mapping functions, automatically set when a controller is set
  /** Map of model names to their link control states */
  std::unordered_map<std::string, LinkControlMap> control_states; ///< Full states and derivatives for each link, indexed as model_name->link_name. Each state is a tuple of <X, X_dot>

  /**
   * @brief Update link using controller and rotor velocity commands.
   * Controller computes CTBT to reach desired_state,
   *  then runs N simulation steps while tracking progress. This requires users
   * to set a controller for the {model_name, link_name } pair. Also this requries MultiRotorPlugin to
   * be attached to the model.
   *
   * @param model_name Name of the model
   * @param link_name Name of the link to control
   * @param desired_state Target state (position, velocity, orientation, etc.)
   * @param N Number of simulation steps to run with this control
   */
  void control_with_rotor_velocity(std::string model_name, std::string link_name, const Stated &desired_state, int N);
  /**
   * @brief Update link using controller and rotor velocity commands.
   * Controller computes CTBT to reach desired_state,
   *  then runs N simulation steps while tracking progress. This requires users
   * to set a controller for the {model_name, link_name } pair. Also this requries MultiRotorPlugin to
   * be attached to the model.
   *
   * @param model_name Name of the model
   * @param link_name Name of the link to control
   * @param desired_state Target state (position, velocity, orientation, etc.)
   * @param N Number of simulation steps to run with this control
   */
  void control_with_rotor_velocity(std::string model_name, std::string link_name, const Stated &&desired_state, int N)
  {
    control_with_rotor_velocity(model_name, link_name, desired_state, N);
  };
  /**
   * @brief Update link using controller and direct wrench commands. Controller computes CTBT to reach desired_state,
   *  then runs N simulation steps while tracking progress. This requires users
   * to set a controller for the {model_name, link_name } pair.
   * @param model_name Name of the model
   * @param link_name Name of the link to control
   * @param desired_state Target state
   * @param N Number of simulation steps
   */
  void control_with_wrench(std::string model_name, std::string link_name, const Stated &desired_state, int N);

  /**
   * @brief Update link using controller and direct wrench commands. Controller computes CTBT to reach desired_state,
   *  then runs N simulation steps while tracking progress. This requires users
   * to set a controller for the {model_name, link_name } pair.
   * @param model_name Name of the model
   * @param link_name Name of the link to control
   * @param desired_state Target state
   * @param N Number of simulation steps
   */
  void control_with_wrench(std::string model_name, std::string link_name, const Stated &&desired_state, int N)
  {
    control_with_wrench(model_name, link_name, desired_state, N);
  };
  /**
   * @brief Attach a controller to a specific link for control_with_rotor_velocity() or control_with_wrench()
   * @param model_name Name of the model
   * @param link_name Name of the link to control
   * @param controller Shared pointer to UAVController instance
   */
  void set_controller(std::string model_name, std::string link_name, std::shared_ptr<UAVController> controller);

  /**
   * @brief Update all control state tuples with current simulation state
   * Refreshes current_state in all control_states entries
   */
  void update_control_states();

  std::vector<std::string> model_names;                                                                                                             ///< All model names specified during construction
  std::unique_ptr<DRL_SERVER_TYPE> server{nullptr};                                                                                                 ///< gz::sim::Server instance
  std::shared_ptr<DRLHelperSystem> internal_sys{nullptr};                                                                                           ///< DRLHelperSystem ptr
  std::unique_ptr<gz::sim::ServerConfig> server_config{nullptr};                                                                                    ///< ServerConfig for this DRLServer
  std::string identity;                                                                                                                             ///< Unique identifier for DRLServer, usually partition
  const std::string _partition;                                                                                                                     ///< Partition specified during construction
  const std::string _sdf_file;                                                                                                                      ///< SDF file name loaded
  const std::vector<std::string> _model_names;                                                                                                      ///< const copy of model_names
  bool headless_render{true};                                                                                                                       ///> Rendering is done headless
  std::unordered_map<std::pair<std::string, std::string>, std::unique_ptr<FirstOrderFilter<Eigen::VectorXd>>, PairHash, PairEq> srt_rate_limiters;  ///< FirstOrder rate limiters for SRT
  std::unordered_map<std::pair<std::string, std::string>, std::unique_ptr<FirstOrderFilter<Eigen::VectorXd>>, PairHash, PairEq> ctbr_rate_limiters; ///< FirstOrder rate limiters for CTBR
  std::unordered_map<std::pair<std::string, std::string>, std::unique_ptr<FirstOrderFilter<Eigen::VectorXd>>, PairHash, PairEq> ctbt_rate_limiters; ///< FirstOrder rate limiters for CTBT
  /**
   * @brief Set rendering mode
   * @param mode true for headless (no GUI), false for GUI rendering
   */
  void set_headless_render_mode(bool mode);

  /**
   * @brief Enable trajectory visualization markers for a link. Creates visual markers that trace the link's path through space
   * @param model_name Name of the model
   * @param link_name Name of the link to trace
   * @param config_ Optional config for marker appearance (color, size, etc.)
   */
  void set_trajectory_trace(std::string model_name, std::string link_name, DRLServerConfig *config_ = nullptr);

  DRLServerConfig drl_server_config;                                                                                            ///< Default DRLSErverConfig
  std::unordered_map<std::pair<std::string, std::string>, std::unique_ptr<MarkerManagerDRL>, PairHash, PairEq> marker_managers; ///< map of marker managers to {model_name, link_name}

  /**
   * @brief Reset all models in the world to new poses.Efficiently resets multiple models at once. Position in meters,
   *          orientation as [roll, pitch, yaw] in radians.
   * @param model_poses Map of model names to (position, orientation) pairs
   *
   */
  void reset_world(const std::unordered_map<std::string, std::pair<Eigen::Vector3d, Eigen::Vector3d>> &model_poses);

  std::shared_ptr<systems::custom_plugins::Sensors> sensor_plugin; ///< Sensor plugin instance (manages cameras, lidars, etc.)

  /**
   * @brief Get image from a camera sensor
   * @param name Sensor name (as defined in SDF)
   * @return Image message (RGB or depth data)
   * @details Returns latest image. May be stale if sensor hasn't updated.
   *          Empty image returned if sensor not found.
   */
  [[nodiscard]] gz::msgs::Image get_sensor_image(std::string name);

  /**
   * @brief Get Camera Info from a camera sensor
   * @param name Sensor name (as defined in SDF)
   * @return Camera Info message
   *          Empty camera info msg returned if sensor not found.
   */
  [[nodiscard]] gz::msgs::CameraInfo get_camera_info(std::string name);

  /**
   * @brief Get Camera Pose from a camera sensor
   * @param name Sensor name (as defined in SDF)
   * @return Camera Pose as gz::math::Pose3d
   *          Empty Pose3d returned if sensor not found.
   */
  [[nodiscard]] gz::math::Pose3d get_camera_pose(std::string name);

  /**
   * @brief Get Lidar Pose from a Lidar sensor
   * @param name Sensor name (as defined in SDF)
   * @return Lidar Pose as gz::math::Pose3d
   *          Empty Pose3d returned if sensor not found.
   */
  [[nodiscard]] gz::math::Pose3d get_lidar_pose(std::string name);

  /**
   * @brief Get GPU lidar scan data
   * @param name Sensor name (as defined in SDF)
   * @return LidarFrameView with pointer to scan data and dimensions
   * @warning Returned pointer is non-owning and valid only until next sensor update
   */
  [[nodiscard]] systems::custom_plugins::Sensors::LidarFrameView get_sensor_gpu_lidar(std::string name);
  /**
   * @brief Print names of all available sensors to console
   * Useful for discovering sensor names in an SDF file
   */
  void print_sensor_names();

  /**
   * @brief Get list of all camera sensor names
   * @return Vector of camera sensor names found in the world
   */
  [[nodiscard]] std::vector<std::string> camera_sensor_names();

  /**
   * @brief Get list of all lidar sensor names
   * @return Vector of lidar sensor names found in the world
   */
  [[nodiscard]] std::vector<std::string> lidar_sensor_names();
  /**
   * @brief Bind a callback for a specific camera. Use camera_sensor_names to get all the
   * available cameras. The callback is expected to take const gz::msgs::Image&
   *
   * @param cb callback to be called when a new image is rendered
   * @param name name of the camera
   */
  void bind_img_cb(std::function<void(const gz::msgs::Image &)> cb, std::string_view name);
  /**
   * @brief Bind a callback for a specific lidar. Use lidar_sensor_names to get all the
   * available lidars. The callback is expected to take const systems::custom_plugins::Sensors::LidarFrameView&
   *
   * @param cb callback to be called when a new scan is completed
   * @param name name of the lidar sensor
   */
  void bind_lidar_cb(std::function<void(const systems::custom_plugins::Sensors::LidarFrameView &)> cb,
                     std::string_view name);

  
  /**
   * @brief  Starts a new camera recording
   * 
   * @param cam_name Unique camera name
   * @param height height of each frame
   * @param width width of each frane
   * @param fps fps of video
   * @param pos Initial camera pos
   * @param ori Initial camera ori
   * @param output_file Output file [.mp4, .avi, or .ogv]
   */
  void start_camera_recording(std::string cam_name, int height, int width, int fps,
                        const Eigen::Vector3d& pos, const Eigen::Vector3d& ori, 
                        std::string output_file);
  /**
   * @brief Updates the camera position and orientation
   * 
   * @param cam_name Name of the camera
   * @param pos New position
   * @param ori New orientation
   */
  void update_camera_recording_pose(std::string cam_name,const Eigen::Vector3d& pos, const Eigen::Vector3d& ori);
  /**
   * @brief Stops recording and finalizes the video 
   * 
   * @param cam_name Name of the camera
   */
  void stop_camera_recording(std::string cam_name);
  /**
   * @brief Request contact data collection for a model.  Delegates to DRLHelperSystem::request_contact_data()
   * @param model_name Name of the model to enable contact sensing
   *
   */
  void request_contact_data(std::string model_name);

  /**
   * @brief Set the time constants for SRT first order rate_limiters
   * @param model_name name of the model for which this rate limit is applied
   * @param link_name name of the link name in the model for which the rate limit is applied
   * @param tau_up rising time constant
   * @param tau_down falling time constant
   * @param initial_value Initial value for the cmd
   */
  void set_srt_rate_limiter_time_constants(std::string model_name, std::string link_name, double tau_up, double tau_down,
                                           const Eigen::VectorXd &initial_value);
  /**
   * @brief Set the time constants for CTBR first order rate_limiters
   * @param model_name name of the model for which this rate limit is applied
   * @param link_name name of the link name in the model for which the rate limit is applied
   * @param tau_up rising time constant
   * @param tau_down falling time constant
   * @param initial_value Initial value for the cmd
   */
  void set_ctbr_rate_limiter_time_constants(std::string model_name, std::string link_name, double tau_up, double tau_down,
                                            const Eigen::VectorXd &initial_value);
  /**
   * @brief Set the time constants for CTBT first order rate_limiters
   * @param model_name name of the model for which this rate limit is applied
   * @param link_name name of the link name in the model for which the rate limit is applied
   * @param tau_up rising time constant
   * @param tau_down falling time constant
   * @param initial_value Initial value for the cmd
   */
  void set_ctbt_rate_limiter_time_constants(std::string model_name, std::string link_name, double tau_up, double tau_down,
                                            const Eigen::VectorXd &initial_value);

  /**
   * @brief Reset the SRT first order rate_limiters to initial state
   * @param model_name name of the model for which this rate limit is applied
   * @param link_name name of the link name in the model for which the rate limit is applied
   */
  void reset_srt_rate_limiter(std::string model_name, std::string link_name);
  /**
   * @brief Reset the CTBR first order rate_limiters to initial state
   * @param model_name name of the model for which this rate limit is applied
   * @param link_name name of the link name in the model for which the rate limit is applied
   */
  void reset_ctbr_rate_limiter(std::string model_name, std::string link_name);
  /**
   * @brief Reset the CTBT first order rate_limiters to initial state
   * @param model_name name of the model for which this rate limit is applied
   * @param link_name name of the link name in the model for which the rate limit is applied
   */
  void reset_ctbt_rate_limiter(std::string model_name, std::string link_name);

  /**
   * @brief Get current simulation iteration count
   * @return Number of simulation steps executed since start
   */
  [[nodiscard]] uint64_t sim_iterations() const { return *(server->IterationCount()); };
  
  /**
  * @brief Manually set a marker for a robot
  * @param model_name name of the model
  * @param link_name name of the link for which the marker will be set
  * @return true if succesfully set
   */
  bool set_marker(std::string model_name, std::string link_name);
  
  /** @brief Global lock variable */
  inline static std::mutex construction_mutex;

  /** @brief sensors enabled flag */
  const bool sensors_enabled_{false};

};

#endif
