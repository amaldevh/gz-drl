// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef COMMON_DRL_
#define COMMON_DRL_
#include <memory>
#include <array>
#include <string>

#include <gz/utils/ImplPtr.hh>
#include "gz/sim/EntityComponentManager.hh"
#include "gz/sim/Export.hh"
#include "gz/sim/ServerConfig.hh"
#include "gz/sim/Link.hh"
#include "gz/sim/Entity.hh"
#include "gz/msgs/actuators.pb.h"
#include "gz/sim/components/Actuators.hh"
#include "gz/sim/components/JointPosition.hh"
#include <mutex>
#include <cstdlib>
#include "gz/sim/components/ContactSensorData.hh"
#include "gz/sim/components/PoseCmd.hh"
#include "gz/msgs/contacts.pb.h"
#include "gz/transport/Node.hh"
#include "gz/transport/NodeOptions.hh"
#include "gz/msgs/world_control.pb.h"
#include "gz/msgs/boolean.pb.h"
#include "gz/msgs/marker.pb.h"
#include "gz/msgs/material.pb.h"
#include "gz/msgs/serialized.pb.h"
#include <gz/common/Console.hh>
#include "controllers/uav_controllers.hh"
#include <tuple>
#include "gz/sim/SdfEntityCreator.hh"
#include "gz/sim/components/LinearVelocityCmd.hh"
#include "gz/sim/components/AngularVelocityCmd.hh"
#include "gz/sim/components/Component.hh"
#include "gz/msgs/float_v.pb.h"
#include "sensor/sensor_interface.hh"
#include <sdf/sdf.hh>
#include "gz/rendering/Camera.hh"
#include <queue>
#include <filesystem>

using namespace gz;
using namespace sim;

// ackerman cmd comp
// class AckerMannCmdCompTag;
using AckerMannVelocityCmdComp = gz::sim::components::Component<gz::math::Vector2d, class AckerMannVelocityCmdCompTag>;
GZ_SIM_REGISTER_COMPONENT("AckerMannVelocityCmdComp", AckerMannVelocityCmdComp)
using JointPositionCmdComp = gz::sim::components::Component<std::vector<double>, class JointPositionCmdCompTag>;
GZ_SIM_REGISTER_COMPONENT("JointPositionCmdComp", JointPositionCmdComp)

/**
 * @brief Debug log printer, useful for printing variadic types.
 * This is the termination case for the variadic case.
 *
 * @tparam T variadic arg type, must have operator<< defined
 * @param arg instant of T
 */
template <typename T>
void log_debug(T arg)
{
  std::cout << arg << std::endl;
}
/**
 * @brief Multiple variadic args printer
 *
 * @tparam T first variadic type
 * @tparam Args Remaining variadic types
 * @param arg T instant
 * @param args instants of Args...
 */
template <typename T, typename... Args>
void log_debug(T arg, Args... args)
{
  std::cout << arg << " ";
  log_debug(args...);
}
#define LOG_DEBUG(...) log_debug("[", __FILE__, " :@line ", __LINE__, " ]", __VA_ARGS__)

typedef Eigen::Matrix<double, 19, 1> GZ_state;
typedef Eigen::Matrix<double, 6, 1> Wrench;
typedef Eigen::Matrix<double, 13, 1> Stated;
typedef Eigen::Matrix<float, 13, 1> Statef;
static const Stated Zero_stated{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
static const Statef Zero_statef{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
static Eigen::Vector3d Zero_Ori{0.0, 0.0, 0.0};

/**
 * @brief RotorParameters struct. This structure holds a mirror to the
 * constants defined in the MultirotorPlugin. See quadrotor.sdf for details.
 * This plugin supports real-time modification of MultiRotorPlugin, without
 * restarting the simulation. Parameters can be modified in this struct, and
 * passed to DRLServer::set_rotor_parameters to modify them.
 *
 */
struct RotorParameters
{
  double max_rot_velocity{2246.0};                                                              ///< maxRotVelocity parameter in MultiRotorPlugin
  Eigen::Vector3d thrust_constant_quadratic_params{1.9533e-6, 0.0, 0.0};              ///< thrustConstantQuadraticParams parameter in MultiRotorPlugin
  Eigen::Vector3d torque_constant_quadratic_params{2.8400982e-08, 0.0, 0.0}; ///< momentConstantQuadraticParams parameter in MultiRotorPlugin
  double ground_effect_constant{1.17198e-09};                                                   ///< groundEffectConstant parameter in MultiRotorPlugin
  double time_constant_up{0.0125};                                                              ///< timeConstantUp parameter in MultiRotorPlugin
  double time_constant_down{0.025};                                                             ///< timeConstantDown parameter in MultiRotorPlugin
  double rotor_drag_coefficient{0.000020673};                                                   ///< rotorDragCoefficient parameter in MultiRotorPlugin
  double rotor_inertia{4.92e-6};                                                                ///< rotorInertia parameter in MultiRotorPlugin
  double rolling_moment_coefficient{0.0};                                                       ///< rollingMomentCoefficient parameter in MultiRotorPlugin
};

struct DRLServerConfig
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Construct a new DRLServerConfig object.
   */
  DRLServerConfig() = default;

  /// Enable trajectory visualization for all models.
  bool trajectory_viz{false};

  /**
   * @brief Maximum number of trajectory markers per model.
   *
   * Markers are reused in a ring buffer after this limit. Since trajectory
   * markers have a finite lifetime, this mainly acts as a safety bound for
   * very high-frequency simulations.
   */
  int max_markers{2048};

  /**
   * @brief Publish one trajectory marker every N position samples.
   *
   * A value of 2 provides a smooth trajectory while avoiding unnecessary
   * Gazebo Transport and rendering overhead.
   */
  int marker_interval{2};

  /**
   * @brief Core trajectory material.
   *
   * Rows are:
   *   0: ambient
   *   1: diffuse
   *   2: specular
   *   3: emissive
   *
   * Default: electric cyan / telemetry blue.
   *
   * Diffuse is based on #33B1FF and is intentionally the strongest
   * component. Ambient and emissive are kept lower to avoid washing out
   * the marker while still making it highly visible against typical
   * Gazebo scenes.
   */
  Eigen::Matrix<float, 4, 4> color{
      {0.05f, 0.22f, 0.32f, 1.00f},  // Ambient
      {0.20f, 0.69f, 1.00f, 1.00f},  // Diffuse  (#33B1FF)
      {0.15f, 0.18f, 0.22f, 1.00f},  // Specular
      {0.06f, 0.24f, 0.34f, 1.00f}   // Emissive
  };
};
/**
 * @brief Manages trajectory visualization markers for a model.
 *
 * Publishes a two-layer "neon breadcrumb" trajectory through Gazebo Transport:
 *   - core: crisp trajectory spheres
 *   - halo: larger, translucent spheres with a shorter lifetime
 *
 * Marker IDs use [1, max_markers] because Gazebo treats ID 0 as "unspecified"
 * and generates a random ID.
 */
struct MarkerManagerDRL
{
  /**
   * @brief Construct a trajectory marker manager.
   *
   * @param config DRL server visualization configuration.
   * @param partition GZ_PARTITION. Use a unique partition per DRLServer to
   * isolate marker traffic between simulations.
   */
  MarkerManagerDRL(
      const DRLServerConfig &config,
      const std::string &partition)
      : max_markers_(config.max_markers),
        marker_interval_(config.marker_interval)
  {
    if (max_markers_ <= 0)
    {
      throw std::invalid_argument(
          "DRLServerConfig.max_markers must be greater than zero");
    }

    if (marker_interval_ <= 0)
    {
      throw std::invalid_argument(
          "DRLServerConfig.marker_interval must be greater than zero");
    }

    gz::transport::NodeOptions options;
    if (!options.SetPartition(partition))
    {
      throw std::invalid_argument("Invalid Gazebo transport partition");
    }

    node_ = std::make_unique<gz::transport::Node>(options);

    configureMarker(
        core_marker_,
        "drl_trajectory/core",
        0.065,                               // 6.5 cm
        std::chrono::seconds{7});

    configureMarker(
        halo_marker_,
        "drl_trajectory/halo",
        0.11,                                // 11 cm
        std::chrono::milliseconds{2500});

    configureCoreMaterial(config);
    configureHaloMaterial(config);
  }

  /**
   * @brief Add a trajectory sample.
   *
   * The first sample is drawn immediately. Subsequent samples are decimated
   * according to marker_interval.
   *
   * @param pos Position of the model in world coordinates.
   */
  void set_marker(const Eigen::Vector3d &pos)
  {
    const std::uint64_t sample = samples_seen_++;

    if ((sample % static_cast<std::uint64_t>(marker_interval_)) != 0)
    {
      return;
    }

    // IDs intentionally start at 1. In Gazebo, ID 0 means "generate an ID".
    const std::uint64_t id =
        1u + (markers_published_ %
              static_cast<std::uint64_t>(max_markers_));

    ++markers_published_;

    setPoseAndId(core_marker_, id, pos);
    setPoseAndId(halo_marker_, id, pos);

    // Send halo first and the solid core second.
    //
    // Request() is asynchronous for this one-way marker service. Its return
    // value only tells us whether the request was successfully queued.
    (void)node_->Request(kMarkerService, halo_marker_);
    (void)node_->Request(kMarkerService, core_marker_);
  }

  /**
   * @brief Remove all trajectory markers owned by this manager.
   *
   * DELETE_ALL with a namespace is much cheaper and cleaner than issuing one
   * DELETE_MARKER request for every possible ID.
   */
  void reset()
  {
    deleteNamespace(core_marker_.ns());
    deleteNamespace(halo_marker_.ns());

    samples_seen_ = 0;
    markers_published_ = 0;
  }

private:
  static constexpr const char *kMarkerService = "/marker";

  /**
   * @brief Configure common properties of a sphere marker.
   */
  static void configureMarker(
      gz::msgs::Marker &marker,
      const std::string &markerNamespace,
      double diameter,
      std::chrono::nanoseconds lifetime)
  {
    marker.set_ns(markerNamespace);
    marker.set_action(gz::msgs::Marker::ADD_MODIFY);
    marker.set_type(gz::msgs::Marker::SPHERE);
    marker.set_visibility(gz::msgs::Marker::GUI);

    gz::msgs::Set(
        marker.mutable_scale(),
        gz::math::Vector3d{diameter, diameter, diameter});

    setLifetime(marker, lifetime);
  }

  /**
   * @brief Preserve the user-configured Gazebo material for the core trail.
   */
  void configureCoreMaterial(const DRLServerConfig &config)
  {
    auto *material = core_marker_.mutable_material();

    setColor(
        material->mutable_ambient(),
        config.color(0, 0),
        config.color(0, 1),
        config.color(0, 2),
        config.color(0, 3));

    setColor(
        material->mutable_diffuse(),
        config.color(1, 0),
        config.color(1, 1),
        config.color(1, 2),
        config.color(1, 3));

    setColor(
        material->mutable_specular(),
        config.color(2, 0),
        config.color(2, 1),
        config.color(2, 2),
        config.color(2, 3));

    setColor(
        material->mutable_emissive(),
        config.color(3, 0),
        config.color(3, 1),
        config.color(3, 2),
        config.color(3, 3));

    // A trajectory should remain conspicuous regardless of scene lighting.
    material->set_lighting(false);
  }

  /**
   * @brief Configure a soft translucent halo around recent trajectory points.
   */
  void configureHaloMaterial(const DRLServerConfig &config)
  {
    // Derive the halo from the configured diffuse color, but brighten it.
    constexpr double kBoost = 1.35;

    const double r = std::clamp(config.color(1, 0) * kBoost, 0.0, 1.0);
    const double g = std::clamp(config.color(1, 1) * kBoost, 0.0, 1.0);
    const double b = std::clamp(config.color(1, 2) * kBoost, 0.0, 1.0);

    auto *material = halo_marker_.mutable_material();

    // Diffuse alpha drives the useful transparency effect in Ogre2.
    setColor(material->mutable_ambient(),  r, g, b, 0.16);
    setColor(material->mutable_diffuse(),  r, g, b, 0.16);
    setColor(material->mutable_specular(), 0.0, 0.0, 0.0, 1.0);

    // Mild emissive component. The translucent shell is what produces the
    // dependable "glow" appearance; emissive alone isn't a bloom effect.
    setColor(
        material->mutable_emissive(),
        r * 0.30,
        g * 0.30,
        b * 0.30,
        1.0);

    material->set_lighting(false);
  }

  /**
   * @brief Set an RGBA protobuf color.
   */
  static void setColor(
      gz::msgs::Color *color,
      double r,
      double g,
      double b,
      double a)
  {
    color->set_r(r);
    color->set_g(g);
    color->set_b(b);
    color->set_a(a);
  }

  /**
   * @brief Set marker lifetime from a chrono duration.
   */
  static void setLifetime(
      gz::msgs::Marker &marker,
      std::chrono::nanoseconds lifetime)
  {
    const auto seconds =
        std::chrono::duration_cast<std::chrono::seconds>(lifetime);

    const auto nanoseconds =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            lifetime - seconds);

    marker.mutable_lifetime()->set_sec(seconds.count());
    marker.mutable_lifetime()->set_nsec(
        static_cast<std::int32_t>(nanoseconds.count()));
  }

  /**
   * @brief Update marker pose and deterministic ring-buffer ID.
   */
  static void setPoseAndId(
      gz::msgs::Marker &marker,
      std::uint64_t id,
      const Eigen::Vector3d &pos)
  {
    marker.set_id(id);

    gz::msgs::Set(
        marker.mutable_pose(),
        gz::math::Pose3d{
            pos.x(),
            pos.y(),
            pos.z(),
            0.0,
            0.0,
            0.0});
  }

  /**
   * @brief Delete every marker in one Gazebo marker namespace.
   */
  void deleteNamespace(const std::string &markerNamespace)
  {
    gz::msgs::Marker delete_msg;
    delete_msg.set_ns(markerNamespace);
    delete_msg.set_action(gz::msgs::Marker::DELETE_ALL);

    (void)node_->Request(kMarkerService, delete_msg);
  }

private:
  gz::msgs::Marker core_marker_;
  gz::msgs::Marker halo_marker_;

  int max_markers_;
  int marker_interval_;

  std::uint64_t samples_seen_{0};
  std::uint64_t markers_published_{0};

  std::unique_ptr<gz::transport::Node> node_;
};

static DRLServerConfig Default_DRLServer_Config;

/**
 * @brief Hash function for pair<std::string, std::string>
 * Optimized for fast lookup
 *
 */
struct PairHash
{
  /**
   * @brief Call operator overload
   *
   * @param p <string, string> pair
   * @return size_t Unique hash
   */
  size_t operator()(const std::pair<std::string, std::string> &p) const noexcept
  {
    std::hash<std::string> h;
    // basic but fast combine
    size_t seed = h(p.first);
    seed ^= h(p.second) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    return seed;
  }
};
/**
 * @brief pair<string, string> comparator
 *
 */
struct PairEq
{
  /**
   * @brief Call overload for comparison
   *
   * @param a pair<string, string> a
   * @param b pair<string, string> b
   * @return true  if a == b
   * @return false if a != b
   */
  bool operator()(const std::pair<std::string, std::string> &a,
                  const std::pair<std::string, std::string> &b) const noexcept
  {
    return a.first == b.first && a.second == b.second;
  }
};

/**
 * @brief Wraps angles withing [0, 2pi]
 *
 * @param input input angle
 * @return double wrapped angle
 */
inline double wrap_angle(double input)
{
  // from rotors (github)
  double wrapped = std::fmod(std::abs(input), 2 * M_PI);
  wrapped = std::copysign(wrapped, input);
  if (std::abs(wrapped - 2 * M_PI) < 1e-8)
  {
    wrapped = 0;
  }
  if (wrapped < 0)
  {
    wrapped += 2 * M_PI;
  }
  return wrapped;
}
#endif
