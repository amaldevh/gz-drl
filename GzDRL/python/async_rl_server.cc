// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#include <chrono>
#include <future>
#include <memory>

#include "async_rl_server.hh"
#include "rl_server.hh"
#include "controllers/uav_controllers.hh"
#include "pybind_common.hh"

using async::AsyncDRLServerPool;

/** @brief AsyncToken for void futures */
class AsyncToken {
public:
  explicit AsyncToken(std::future<void>&& fut) : fut_(std::make_shared<std::future<void>>(std::move(fut))) {}

  bool done() {
    using namespace std::chrono_literals;
    return fut_->wait_for(0s) == std::future_status::ready;
  }

  bool wait(double timeout_sec = -1.0) {
    if (timeout_sec < 0.0) {
      fut_->wait();
      return true;
    }
    using namespace std::chrono;
    auto dur = duration_cast<std::chrono::milliseconds>(duration<double>(timeout_sec));
    return fut_->wait_for(dur) == std::future_status::ready;
  }

  void result() {
    // get() will rethrow exceptions from the worker thread
    fut_->get();
  }

private:
  std::shared_ptr<std::future<void>> fut_;  // keep movable future alive for Python object lifetime
};


/** @brief Registers AsyncToken and AsyncDRLServerPool on the gzdrl module. */
void bind_async_drl_server_pool(py::module_ &m) {
    py::class_<AsyncToken>(m, "AsyncToken", R"pbdoc(
        Handle for asynchronous operations in AsyncDRLServerPool.

        Used to track completion status and retrieve results of
        asynchronous pool operations.

        Methods
        -------
        done()
            Check if the task has completed.
        wait(timeout_sec=-1.0)
            Wait for task completion with optional timeout.
        result()
            Block until done and propagate any exceptions.
    )pbdoc")
        .def("done", &AsyncToken::done,
          R"pbdoc(
            Check if the task has completed.

            Returns
            -------
            bool
                True if the task finished, False otherwise.
          )pbdoc")
        .def("wait", &AsyncToken::wait,
          py::arg("timeout_sec") = -1.0,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
            Wait for task completion with optional timeout.

            Parameters
            ----------
            timeout_sec : float, optional
                Maximum time to wait in seconds. Negative means wait forever.

            Returns
            -------
            bool
                True if completed before timeout, False otherwise.
          )pbdoc")
        .def("result", &AsyncToken::result,
            py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
            Block until done and propagate any exceptions.

            Returns
            -------
            None
                Returns None on success, raises exception on failure.
            )pbdoc");

    py::class_<AsyncDRLServerPool, std::shared_ptr<AsyncDRLServerPool>>(m, "AsyncDRLServerPool", R"pbdoc(
        Pool of DRLServer instances for parallel simulation.

        Manages multiple DRLServer instances for efficient parallel training
        of reinforcement learning agents. Each server runs in its own
        isolated Gazebo partition.

        Parameters
        ----------
        num_servers : int
            Number of parallel environments to create.
        partition : str
            Base partition name (servers get partition0, partition1, etc.).
        sdf_file : str
            Path to the SDF world file.
        model_names : list[str]
            List of model names to track in each environment.
        enable_sensors : bool
            Whether to enable sensor interfaces.
        config : DRLServerConfig, optional
            Server configuration.

        Notes
        -----
        - All methods accept env_ids to specify which environments to operate on
        - Arguments can be broadcast (size 1) or per-environment (size N)
        - Operations are executed in parallel across environments

        Example
        -------
        >>> import gzdrl
        >>> import numpy as np
        >>> pool = gzdrl.AsyncDRLServerPool(
        ...     num_servers=8,
        ...     partition="train",
        ...     sdf_file=str(gzdrl.get_sdf_path("world_simple.sdf")),
        ...     model_names=["quadrotor"],
        ...     enable_sensors=False
        ... )
        >>>
        >>> # Reset all environments
        >>> env_ids = list(range(8))
        >>> positions = [np.array([i, 0., 1.]) for i in range(8)]
        >>> pool.reset_pos(env_ids, ["quadrotor"], positions, [np.zeros(3)])
        >>>
        >>> # Step all environments
        >>> pool.run_once(env_ids)
        >>>
        >>> # Get states from all environments
        >>> states = pool.state_info(env_ids, ["quadrotor"])

        See Also
        --------
        DRLServer : Single environment server.
    )pbdoc")
          // ----------------- Constructors (unchanged) -----------------
          .def(py::init([](std::size_t num_servers,
                          const std::string& partition,
                          const std::string& sdf_file,
                          const std::vector<std::string>& model_names,
                          bool enable_sensors,
                          const DRLServerConfig& config) {
               return std::make_shared<AsyncDRLServerPool>(
               num_servers,
               [=](std::size_t i) {
                 auto srv = std::make_unique<DRLServer>(
                   partition + std::to_string(i), sdf_file, model_names, enable_sensors, config);
                 return srv;
               }
             );
           }),
           py::arg("num_servers"),
           py::arg("partition"),
           py::arg("sdf_file"),
           py::arg("model_names"),
           py::arg("enable_sensors"),
           py::arg("config"),
           py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Create a pool of DRLServer instances with custom configuration.

            Each environment ``i`` runs in its own server with partition
            ``partition + str(i)``.

            Parameters
            ----------
            num_servers : int
                Number of parallel environments to create.
            partition : str
                Base partition name for transport isolation.
            sdf_file : str
                Path to the SDF world file.
            model_names : list[str]
                List of model names to track in each environment.
            enable_sensors : bool
                Whether to enable sensor interfaces (cameras, LiDAR).
            config : DRLServerConfig
                Custom server configuration applied to every environment.
           )pbdoc")
      .def(py::init([](std::size_t num_servers,
                       const std::string& partition,
                       const std::string& sdf_file,
                       const std::vector<std::string>& model_names,
                      bool enable_sensors) {
             return std::make_shared<AsyncDRLServerPool>(
               num_servers,
               [=](std::size_t i) {
                 auto srv = std::make_unique<DRLServer>(
                   partition + std::to_string(i), sdf_file, model_names, enable_sensors);
                 return srv;
               }
             );
           }),
           py::arg("num_servers"),
           py::arg("partition"),
           py::arg("sdf_file"),
           py::arg("model_names"),
           py::arg("enable_sensors"),
           py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Create a pool of DRLServer instances with default configuration.

            Each environment ``i`` runs in its own server with partition
            ``partition + str(i)``.

            Parameters
            ----------
            num_servers : int
                Number of parallel environments to create.
            partition : str
                Base partition name for transport isolation.
            sdf_file : str
                Path to the SDF world file.
            model_names : list[str]
                List of model names to track in each environment.
            enable_sensors : bool
                Whether to enable sensor interfaces (cameras, LiDAR).
           )pbdoc")

      // ----------------- Pool control -----------------
      .def("size", &AsyncDRLServerPool::size, py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Get the number of environments (servers) in the pool.

            Returns
            -------
            int
                Number of parallel environments managed by the pool.
          )pbdoc")
      .def("close", &AsyncDRLServerPool::close, py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Shut down the pool and all underlying servers.

            Stops the worker threads and releases every DRLServer instance.
            The pool must not be used after calling this.
          )pbdoc")

      // ----------------- run_once / run_N -----------------
      .def("run_once", [](AsyncDRLServerPool& self,
                          const std::vector<std::size_t>& env_ids) {
             std::vector<decltype(self.call(std::size_t{}, &DRLServer::run_once))> futs;
             futs.reserve(env_ids.size());
             for (auto id : env_ids) futs.emplace_back(self.call(id, &DRLServer::run_once));
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Execute one physics step in the selected environments in parallel.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to step.

            Notes
            -----
            Blocks until all selected environments have completed the step.
          )pbdoc")

      .def("run_N", [](AsyncDRLServerPool& self,
                       const std::vector<std::size_t>& env_ids,
                       const std::vector<int>& Ns) {
             const auto n = env_ids.size();
             check_broadcastable(n, Ns, "Ns");
             std::vector<decltype(self.call(std::size_t{}, &DRLServer::run_N, 0))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i) {
               futs.emplace_back(self.call(env_ids[i], &DRLServer::run_N, pick(Ns, i)));
             }
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("Ns"), py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Execute N physics steps in the selected environments in parallel.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to step.
            Ns : list[int]
                Number of steps per environment (broadcast if size 1).

            Notes
            -----
            Blocks until all selected environments have completed their steps.
          )pbdoc")

      // ----------------- state_info (returns vector) -----------------
      .def("state_info", [](AsyncDRLServerPool& self,
                            const std::vector<std::size_t>& env_ids,
                            const std::vector<std::string>& model_names) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             using FutT = decltype(self.call(std::size_t{}, &DRLServer::state_info, std::string{}));
             std::vector<FutT> futs; futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], &DRLServer::state_info, pick(model_names, i)));
             using RetT = decltype(futs[0].get());
             std::vector<RetT> out; out.reserve(n);
             for (auto& f : futs) out.emplace_back(f.get());
             return out;
           }, py::arg("env_ids"), py::arg("model_names"), py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Get current state of all links in a model for each environment.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to query.
            model_names : list[str]
                Model name per environment (broadcast if size 1).

            Returns
            -------
            list[dict[str, numpy.ndarray]]
                One mapping of link name to 19-element state vector per
                environment, in the same order as env_ids.
          )pbdoc")
      // ----------------- reset_pos -----------------
      .def("reset_pos", [](AsyncDRLServerPool& self,
                           const std::vector<std::size_t>& env_ids,
                           const std::vector<std::string>& model_names,
                           const std::vector<Eigen::Vector3d>& positions,
                           const std::vector<Eigen::Vector3d>& orientations) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             check_broadcastable(n, positions, "positions");
             check_broadcastable(n, orientations, "orientations");
             using MF = void (DRLServer::*)(std::string, Eigen::Vector3d&&, Eigen::Vector3d&&);
             MF mf = &DRLServer::reset_pos;
             std::vector<decltype(self.call(std::size_t{}, mf, std::string{}, Eigen::Vector3d{}, Eigen::Vector3d{}))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i) {
               futs.emplace_back(self.call(env_ids[i], mf,
                                           pick(model_names, i),
                                           Eigen::Vector3d(pick(positions, i)),
                                           Eigen::Vector3d(pick(orientations, i))));
             }
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("model_names"), py::arg("positions"), py::arg("orientations"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Reset a model to a new pose in each environment (teleport).

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            model_names : list[str]
                Model name per environment (broadcast if size 1).
            positions : list[numpy.ndarray]
                New position [x, y, z] in meters per environment (broadcast if size 1).
            orientations : list[numpy.ndarray]
                New orientation [roll, pitch, yaw] in radians per environment
                (broadcast if size 1).

            Notes
            -----
            Each underlying server performs three stabilization steps before
            this call returns.
          )pbdoc")

      // ------------------respawn_model-----------------------------
      .def("respawn_model", [](AsyncDRLServerPool& self,
                           const std::vector<std::size_t>& env_ids,
                           const std::vector<std::string>& model_names,
                           const std::vector<Eigen::Vector3d>& positions,
                           const std::vector<Eigen::Vector3d>& orientations) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             check_broadcastable(n, positions, "positions");
             check_broadcastable(n, orientations, "orientations");
             using MF = void (DRLServer::*)(std::string, Eigen::Vector3d&&, Eigen::Vector3d&&);
             MF mf = &DRLServer::respawn_model;
             std::vector<decltype(self.call(std::size_t{}, mf, std::string{}, Eigen::Vector3d{}, Eigen::Vector3d{}))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i) {
               futs.emplace_back(self.call(env_ids[i], mf,
                                           pick(model_names, i),
                                           Eigen::Vector3d(pick(positions, i)),
                                           Eigen::Vector3d(pick(orientations, i))));
             }
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("model_names"), py::arg("positions"), py::arg("orientations"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Respawn a model at a new pose in each environment.

            Triggers entity removal and recreation, applying any pending mass,
            inertia or rotor parameter changes.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            model_names : list[str]
                Model name per environment (broadcast if size 1).
            positions : list[numpy.ndarray]
                New position [x, y, z] in meters per environment (broadcast if size 1).
            orientations : list[numpy.ndarray]
                New orientation [roll, pitch, yaw] in radians per environment
                (broadcast if size 1).

            Warning
            -------
            This is an expensive operation. Each underlying server performs
            three stabilization steps before this call returns.
          )pbdoc")

      // ----------------- set_rotor_velocity_cmd -----------------
      .def("set_rotor_velocity_cmd", [](AsyncDRLServerPool& self,
                                        const std::vector<std::size_t>& env_ids,
                                        const std::vector<std::string>& model_names,
                                        const std::vector<std::string>& link_names,
                                        const std::vector<Eigen::VectorXd>& cmds) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             check_broadcastable(n, link_names, "link_names");
             check_broadcastable(n, cmds, "cmds");
             using MF = void (DRLServer::*)(std::string, std::string, Eigen::VectorXd&&);
             MF mf = &DRLServer::set_rotor_velocity_cmd;
             std::vector<decltype(self.call(std::size_t{}, mf, std::string{}, std::string{}, Eigen::VectorXd{}))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], mf,
                                           pick(model_names, i), pick(link_names, i),
                                           Eigen::VectorXd(pick(cmds, i))));
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("model_names"), py::arg("link_names"), py::arg("cmds"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Set rotor velocities for a link in each environment.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            model_names : list[str]
                Model name per environment (broadcast if size 1).
            link_names : list[str]
                Link name with rotor actuators per environment (broadcast if size 1).
            cmds : list[numpy.ndarray]
                Rotor velocities in rad/s (one per rotor) per environment
                (broadcast if size 1).
          )pbdoc")


      // ----------------- set_velocity_cmd -----------------
      .def("set_velocity_cmd", [](AsyncDRLServerPool& self,
                                  const std::vector<std::size_t>& env_ids,
                                  const std::vector<std::string>& model_names,
                                  const std::vector<std::string>& link_names,
                                  const std::vector<Eigen::Vector3d>& cmds) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             check_broadcastable(n, link_names, "link_names");
             check_broadcastable(n, cmds, "cmds");
             using MF = void (DRLServer::*)(std::string, std::string, Eigen::Vector3d&&);
             MF mf = &DRLServer::set_velocity_cmd;
             std::vector<decltype(self.call(std::size_t{}, mf, std::string{}, std::string{}, Eigen::Vector3d{}))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], mf,
                                           pick(model_names, i), pick(link_names, i),
                                           Eigen::Vector3d(pick(cmds, i))));
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("model_names"), py::arg("link_names"), py::arg("cmds"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Set linear velocity command for a link in each environment.

            Kinematic control - the link immediately achieves the commanded
            velocity.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            model_names : list[str]
                Model name per environment (broadcast if size 1).
            link_names : list[str]
                Link name per environment (broadcast if size 1).
            cmds : list[numpy.ndarray]
                Linear velocity [vx, vy, vz] in m/s (world frame) per environment
                (broadcast if size 1).
          )pbdoc")


      // ----------------- set_angular_velocity_cmd -----------------
      .def("set_angular_velocity_cmd", [](AsyncDRLServerPool& self,
                                          const std::vector<std::size_t>& env_ids,
                                          const std::vector<std::string>& model_names,
                                          const std::vector<std::string>& link_names,
                                          const std::vector<Eigen::Vector3d>& cmds) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             check_broadcastable(n, link_names, "link_names");
             check_broadcastable(n, cmds, "cmds");
             using MF = void (DRLServer::*)(std::string, std::string, Eigen::Vector3d&&);
             MF mf = &DRLServer::set_angular_velocity_cmd;
             std::vector<decltype(self.call(std::size_t{}, mf, std::string{}, std::string{}, Eigen::Vector3d{}))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], mf,
                                           pick(model_names, i), pick(link_names, i),
                                           Eigen::Vector3d(pick(cmds, i))));
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("model_names"), py::arg("link_names"), py::arg("cmds"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Set angular velocity command for a link in each environment.

            Kinematic control - the link immediately achieves the commanded
            angular velocity.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            model_names : list[str]
                Model name per environment (broadcast if size 1).
            link_names : list[str]
                Link name per environment (broadcast if size 1).
            cmds : list[numpy.ndarray]
                Angular velocity [wx, wy, wz] in rad/s (body-fixed frame) per
                environment (broadcast if size 1).
          )pbdoc")


      // ----------------- set_ackermann_velocity_cmd -----------------
      .def("set_ackermann_velocity_cmd", [](AsyncDRLServerPool& self,
                                            const std::vector<std::size_t>& env_ids,
                                            const std::vector<std::string>& model_names,
                                            const std::vector<std::string>& link_names,
                                            const std::vector<Eigen::Vector2d>& cmds) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             check_broadcastable(n, link_names, "link_names");
             check_broadcastable(n, cmds, "cmds");
             using MF = void (DRLServer::*)(std::string, std::string, Eigen::Vector2d&&);
             MF mf = &DRLServer::set_ackermann_velocity_cmd;
             std::vector<decltype(self.call(std::size_t{}, mf, std::string{}, std::string{}, Eigen::Vector2d{}))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], mf,
                                           pick(model_names, i), pick(link_names, i),
                                           Eigen::Vector2d(pick(cmds, i))));
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("model_names"), py::arg("link_names"), py::arg("cmds"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Set Ackermann steering velocity command in each environment.

            Requires the AckermannSteering plugin to be attached to the model.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            model_names : list[str]
                Model name per environment (broadcast if size 1).
            link_names : list[str]
                Chassis link name per environment (broadcast if size 1).
            cmds : list[numpy.ndarray]
                [linear_velocity (m/s), angular_velocity (rad/s)] per environment
                (broadcast if size 1).
          )pbdoc")


      // ----------------- set_joint_position_cmd -----------------
      .def("set_joint_position_cmd", [](AsyncDRLServerPool& self,
                                        const std::vector<std::size_t>& env_ids,
                                        const std::vector<std::string>& model_names,
                                        const std::vector<std::string>& joint_names,
                                        const std::vector<Eigen::Vector3d>& cmds) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             check_broadcastable(n, joint_names, "joint_names");
             check_broadcastable(n, cmds, "cmds");
             using MF = void (DRLServer::*)(std::string, std::string, Eigen::Vector3d&&);
             MF mf = &DRLServer::set_joint_position_cmd;
             std::vector<decltype(self.call(std::size_t{}, mf, std::string{}, std::string{}, Eigen::Vector3d{}))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], mf,
                                           pick(model_names, i), pick(joint_names, i),
                                           Eigen::Vector3d(pick(cmds, i))));
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("model_names"), py::arg("joint_names"), py::arg("cmds"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Set joint position command in each environment.

            Requires the JointPositionController plugin to be attached.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            model_names : list[str]
                Model name per environment (broadcast if size 1).
            joint_names : list[str]
                Joint name per environment (broadcast if size 1).
            cmds : list[numpy.ndarray]
                Joint position target per environment (radians for revolute,
                meters for prismatic; broadcast if size 1).
          )pbdoc")

      // ----------------- get_contacts (returns vector) -----------------
      .def("get_contacts", [](AsyncDRLServerPool& self,
                              const std::vector<std::size_t>& env_ids,
                              const std::vector<std::string>& model_names) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             using FutT = decltype(self.call(std::size_t{}, &DRLServer::get_contacts, std::string{}));
             std::vector<FutT> futs; futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], &DRLServer::get_contacts, pick(model_names, i)));
             using RetT = decltype(futs[0].get());
             std::vector<RetT> out; out.reserve(n);
             for (auto& f : futs) out.emplace_back(f.get());
             return out;
           }, py::arg("env_ids"), py::arg("model_names"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Get contact data for all links in a model for each environment.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to query.
            model_names : list[str]
                Model name per environment (broadcast if size 1).

            Returns
            -------
            list[dict[str, list[Contacts]]]
                One mapping of link name to contact batches per environment.

            Notes
            -----
            Call request_contact_data() first to enable contact sensing.
          )pbdoc")

      // ----------------- set_wrench -----------------
      .def("set_wrench", [](AsyncDRLServerPool& self,
                            const std::vector<std::size_t>& env_ids,
                            const std::vector<std::string>& model_names,
                            const std::vector<std::string>& link_names,
                            const std::vector<Eigen::Vector3d>& forces,
                            const std::vector<Eigen::Vector3d>& moments) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             check_broadcastable(n, link_names, "link_names");
             check_broadcastable(n, forces, "forces");
             check_broadcastable(n, moments, "moments");
             using MF = void (DRLServer::*)(std::string, std::string, Eigen::Vector3d&&, Eigen::Vector3d&&);
             MF mf = &DRLServer::set_wrench;
             std::vector<decltype(self.call(std::size_t{}, mf, std::string{}, std::string{}, Eigen::Vector3d{}, Eigen::Vector3d{}))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], mf,
                                           pick(model_names, i), pick(link_names, i),
                                           Eigen::Vector3d(pick(forces, i)),
                                           Eigen::Vector3d(pick(moments, i))));
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("model_names"), py::arg("link_names"),
              py::arg("forces"), py::arg("moments"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Apply a wrench (force + torque) to a link in each environment.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            model_names : list[str]
                Model name per environment (broadcast if size 1).
            link_names : list[str]
                Link name per environment (broadcast if size 1).
            forces : list[numpy.ndarray]
                Force [fx, fy, fz] in Newtons (world frame) per environment
                (broadcast if size 1).
            moments : list[numpy.ndarray]
                Torque [tx, ty, tz] in N*m (world frame) per environment
                (broadcast if size 1).
          )pbdoc")

      // ----------------- set_controller -----------------
      .def("set_controller", [](AsyncDRLServerPool& self,
                                const std::vector<std::size_t>& env_ids,
                                const std::vector<std::string>& model_names,
                                const std::vector<std::string>& link_names,
                                const std::vector<std::shared_ptr<UAVController>>& controllers) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             check_broadcastable(n, link_names, "link_names");
             check_broadcastable(n, controllers, "controllers");
             std::vector<decltype(self.call(std::size_t{}, &DRLServer::set_controller, std::string{}, std::string{}, std::shared_ptr<UAVController>{}))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], &DRLServer::set_controller,
                                           pick(model_names, i), pick(link_names, i),
                                           pick(controllers, i)));
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("model_names"), py::arg("link_names"), py::arg("controllers"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Attach a controller to a link in each environment.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            model_names : list[str]
                Model name per environment (broadcast if size 1).
            link_names : list[str]
                Link name per environment (broadcast if size 1).
            controllers : list[UAVController]
                Controller instance per environment (broadcast if size 1).

            See Also
            --------
            control_with_rotor_velocity : Use the attached controller.
          )pbdoc")

      // ----------------- control_states getter -----------------
      .def("get_control_states", [](AsyncDRLServerPool& self,
                                    const std::vector<std::size_t>& env_ids) {
             using FutT = decltype(self.submit(std::size_t{}, [](DRLServer& s){ return s.control_states; }));
             std::vector<FutT> futs; futs.reserve(env_ids.size());
             for (auto id : env_ids)
               futs.emplace_back(self.submit(id, [](DRLServer& s){ return s.control_states; }));
             using RetT = decltype(futs[0].get());
             std::vector<RetT> out; out.reserve(futs.size());
             for (auto& f : futs) out.emplace_back(f.get());
             return out;
           }, py::arg("env_ids"), py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Get the control states for the selected environments.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to query.

            Returns
            -------
            list[dict]
                One control-states mapping per environment. Each maps link names
                to (desired_state, current_state) tuples.
          )pbdoc")

      // ----------------- controller-driven updates -----------------
      .def("control_with_rotor_velocity", [](AsyncDRLServerPool& self,
                                                      const std::vector<std::size_t>& env_ids,
                                                      const std::vector<std::string>& model_names,
                                                      const std::vector<std::string>& link_names,
                                                      const std::vector<Stated>& desired_states,
                                                      const std::vector<int>& Ns) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             check_broadcastable(n, link_names, "link_names");
             check_broadcastable(n, desired_states, "desired_states");
             check_broadcastable(n, Ns, "Ns");
             using MF = void (DRLServer::*)(std::string, std::string, const Stated&&, int);
             MF mf = &DRLServer::control_with_rotor_velocity;
             std::vector<decltype(self.call(std::size_t{}, mf, std::string{}, std::string{}, Stated{}, 0))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], mf,
                                           pick(model_names, i), pick(link_names, i),
                                           Stated(pick(desired_states, i)), pick(Ns, i)));
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("model_names"), py::arg("link_names"),
              py::arg("desired_states"), py::arg("Ns"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Drive links via their controllers using rotor velocity commands.

            For each environment the attached controller computes rotor velocity
            commands to reach the desired state, then N steps are simulated.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            model_names : list[str]
                Model name per environment (broadcast if size 1).
            link_names : list[str]
                Link name per environment (broadcast if size 1).
            desired_states : list[numpy.ndarray]
                Target state per environment (broadcast if size 1).
            Ns : list[int]
                Number of simulation steps per environment (broadcast if size 1).

            Notes
            -----
            Requires a controller set via set_controller() and the model must
            have the MultiRotorPlugin attached.
          )pbdoc")

      .def("control_with_wrench", [](AsyncDRLServerPool& self,
                                                    const std::vector<std::size_t>& env_ids,
                                                    const std::vector<std::string>& model_names,
                                                    const std::vector<std::string>& link_names,
                                                    const std::vector<Stated>& desired_states,
                                                    const std::vector<int>& Ns) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             check_broadcastable(n, link_names, "link_names");
             check_broadcastable(n, desired_states, "desired_states");
             check_broadcastable(n, Ns, "Ns");
             using MF = void (DRLServer::*)(std::string, std::string, const Stated&&, int);
             MF mf = &DRLServer::control_with_wrench;
             std::vector<decltype(self.call(std::size_t{}, mf, std::string{}, std::string{}, Stated{}, 0))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], mf,
                                           pick(model_names, i), pick(link_names, i),
                                           Stated(pick(desired_states, i)), pick(Ns, i)));
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("model_names"), py::arg("link_names"),
              py::arg("desired_states"), py::arg("Ns"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Drive links via their controllers using direct wrench commands.

            For each environment the attached controller computes forces/torques
            to reach the desired state, then N steps are simulated.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            model_names : list[str]
                Model name per environment (broadcast if size 1).
            link_names : list[str]
                Link name per environment (broadcast if size 1).
            desired_states : list[numpy.ndarray]
                Target state per environment (broadcast if size 1).
            Ns : list[int]
                Number of simulation steps per environment (broadcast if size 1).

            Notes
            -----
            Requires a controller set via set_controller(). Does not require a
            MultiRotorPlugin.
          )pbdoc")

      .def("update_control_states", [](AsyncDRLServerPool& self,
                                       const std::vector<std::size_t>& env_ids) {
             std::vector<decltype(self.call(std::size_t{}, &DRLServer::update_control_states))> futs;
             futs.reserve(env_ids.size());
             for (auto id : env_ids) futs.emplace_back(self.call(id, &DRLServer::update_control_states));
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Refresh control states with the current simulation state.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to update.

            Notes
            -----
            Call before using controllers to ensure current_state is up to date.
          )pbdoc")

      // ----------------- model_names getter per env -----------------
      .def("model_names", [](AsyncDRLServerPool& self,
                             const std::vector<std::size_t>& env_ids) {
             using FutT = decltype(self.submit(std::size_t{}, [](DRLServer& s){ return s.model_names; }));
             std::vector<FutT> futs; futs.reserve(env_ids.size());
             for (auto id : env_ids)
               futs.emplace_back(self.submit(id, [](DRLServer& s){ return s.model_names; }));
             using RetT = decltype(futs[0].get());
             std::vector<RetT> out; out.reserve(futs.size());
             for (auto& f : futs) out.emplace_back(f.get());
             return out;
           }, py::arg("env_ids"), py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Get the tracked model names for the selected environments.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to query.

            Returns
            -------
            list[list[str]]
                Tracked model names per environment.
          )pbdoc")

      // ----------------- trajectory trace -----------------
      .def("set_trajectory_trace", [](AsyncDRLServerPool& self,
                                      const std::vector<std::size_t>& env_ids,
                                      const std::vector<std::string>& model_names,
                                      const std::vector<std::string>& link_names,
                                      const std::vector<DRLServerConfig*>& configs) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             check_broadcastable(n, link_names, "link_names");
             // empty -> treat as [nullptr]
             std::vector<DRLServerConfig*> cfgs = configs.empty() ? std::vector<DRLServerConfig*>{nullptr} : configs;
             check_broadcastable(n, cfgs, "configs");
             std::vector<decltype(self.call(std::size_t{}, &DRLServer::set_trajectory_trace, std::string{}, std::string{}, (DRLServerConfig*)nullptr))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], &DRLServer::set_trajectory_trace,
                                           pick(model_names, i), pick(link_names, i), pick(cfgs, i)));
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("model_names"), py::arg("link_names"), py::arg("configs"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Enable trajectory visualization markers for a link in each environment.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            model_names : list[str]
                Model name per environment (broadcast if size 1).
            link_names : list[str]
                Link name to trace per environment (broadcast if size 1).
            configs : list[DRLServerConfig]
                Marker configuration per environment. Pass an empty list to use
                the default configuration for all environments.
          )pbdoc")

      // ----------------- sensors -----------------
      .def("get_sensor_image", [](AsyncDRLServerPool& self,
                                  const std::vector<std::size_t>& env_ids,
                                  const std::vector<std::string>& sensor_names) {
             const auto n = env_ids.size();
             check_broadcastable(n, sensor_names, "sensor_names");
             using FutT = decltype(self.call(std::size_t{}, &DRLServer::get_sensor_image, std::string{}));
             std::vector<gz::msgs::Image> messages;
             messages.reserve(n);
             {
               py::gil_scoped_release release;
               std::vector<FutT> futs; futs.reserve(n);
               for (std::size_t i = 0; i < n; ++i)
                 futs.emplace_back(self.call(env_ids[i], &DRLServer::get_sensor_image,
                                             pick(sensor_names, i)));
               for (auto& f : futs) messages.emplace_back(f.get());
             }
             using RetT = py::array;
             std::vector<RetT> out; out.reserve(n);
             for (const auto& message : messages)
               out.emplace_back(convert_image_msg_to_numpy_copy(message));
             return out;
           }, py::arg("env_ids"),
              py::arg("sensor_names"),
           R"pbdoc(
            Get a camera image from each environment.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to query.
            sensor_names : list[str]
                Camera sensor name per environment (broadcast if size 1).

            Returns
            -------
            list[numpy.ndarray]
                One image array (height, width, channels) per environment.
          )pbdoc")
        .def("get_sensor_gpu_lidar", [](AsyncDRLServerPool& self,
                                  const std::vector<std::size_t>& env_ids,
                                  const std::vector<std::string>& sensor_names) {
             const auto n = env_ids.size();
             check_broadcastable(n, sensor_names, "sensor_names");
             using FutT = decltype(self.call(std::size_t{}, &DRLServer::get_sensor_gpu_lidar, std::string{}));
             std::vector<systems::custom_plugins::Sensors::LidarFrameView> data;
             data.reserve(n);
             {
               py::gil_scoped_release release;
               std::vector<FutT> futs; futs.reserve(n);
               for (std::size_t i = 0; i < n; ++i)
                 futs.emplace_back(self.call(env_ids[i], &DRLServer::get_sensor_gpu_lidar,
                                             pick(sensor_names, i)));
               for (auto& f : futs) data.emplace_back(f.get());
             }
             using RetT = py::array;
             std::vector<RetT> out; out.reserve(n);
             for (const auto& data_tuple : data) {
               if (data_tuple.data)
                 out.emplace_back(py::array_t<float>(
                     {data_tuple.h, data_tuple.w, data_tuple.c}, data_tuple.data));
               else
                 out.emplace_back(py::array_t<float>({0, 0, 0}));
             }
             return out;
           }, py::arg("env_ids"),
              py::arg("sensor_names"),
           R"pbdoc(
            Get GPU LiDAR scan data from each environment.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to query.
            sensor_names : list[str]
                LiDAR sensor name per environment (broadcast if size 1).

            Returns
            -------
            list[numpy.ndarray]
                One scan array (height, width, channels) per environment.

            Warning
            -------
            Returned data is valid only until the next sensor update.
          )pbdoc")
        .def("print_sensor_names", [](AsyncDRLServerPool& self,
                                  const std::vector<std::size_t>& env_ids) {
             std::vector<decltype(self.call(std::size_t{}, &DRLServer::print_sensor_names))> futs;
             futs.reserve(env_ids.size());
             for (auto id : env_ids)
               futs.emplace_back(self.call(id, &DRLServer::print_sensor_names));
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Print the available sensor names of the selected environments.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to query.
          )pbdoc")
        .def("camera_sensor_names", [](AsyncDRLServerPool& self,
                                  const std::vector<std::size_t>& env_ids) {
             using FutT = decltype(self.call(std::size_t{}, &DRLServer::camera_sensor_names));
             std::vector<FutT> futs; futs.reserve(env_ids.size());
             for (auto id : env_ids)
               futs.emplace_back(self.call(id, &DRLServer::camera_sensor_names));
             using RetT = decltype(futs[0].get());
             std::vector<RetT> out; out.reserve(futs.size());
             for (auto& f : futs) out.emplace_back(f.get());
             return out;
           }, py::arg("env_ids"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Get the camera sensor names for the selected environments.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to query.

            Returns
            -------
            list[list[str]]
                Camera sensor names per environment.
          )pbdoc")
        .def("lidar_sensor_names", [](AsyncDRLServerPool& self,
                                  const std::vector<std::size_t>& env_ids) {
             using FutT = decltype(self.call(std::size_t{}, &DRLServer::lidar_sensor_names));
             std::vector<FutT> futs; futs.reserve(env_ids.size());
             for (auto id : env_ids)
               futs.emplace_back(self.call(id, &DRLServer::lidar_sensor_names));
             using RetT = decltype(futs[0].get());
             std::vector<RetT> out; out.reserve(futs.size());
             for (auto& f : futs) out.emplace_back(f.get());
             return out;
           }, py::arg("env_ids"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Get the LiDAR sensor names for the selected environments.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to query.

            Returns
            -------
            list[list[str]]
                LiDAR sensor names per environment.
          )pbdoc")
        .def("bind_img_cb", [](AsyncDRLServerPool& self,
                                  const std::vector<std::size_t>& env_ids,
                                  const std::vector<py::function>& callbacks,
                                  const std::vector<std::string>& sensor_names) {
             const auto n = env_ids.size();
             check_broadcastable(n, sensor_names, "sensor_names");
             check_broadcastable(n, callbacks, "callbacks");
             std::vector<std::shared_ptr<py::function>> callback_holders;
             callback_holders.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               callback_holders.emplace_back(
                   keep_python_callback(py::function(pick(callbacks, i))));
             std::vector<decltype(self.call(std::size_t{}, &DRLServer::bind_img_cb,
             std::function<void (const gz::msgs::Image&)>{},
             std::string{}))> futs;
             futs.reserve(n);
             {
               py::gil_scoped_release release;
               for (std::size_t i = 0; i < n; ++i){
                 auto cb_ptr = callback_holders[i];
                 futs.emplace_back(self.call(env_ids[i], &DRLServer::bind_img_cb,
                  [cb_ptr](const gz::msgs::Image& img_msg){
                  if (!Py_IsInitialized()) return;
                  py::gil_scoped_acquire gil;
                  auto img = convert_image_msg_to_numpy_copy(img_msg);
                  (*cb_ptr)(img);
                  }, pick(sensor_names, i)));
               }
               for (auto& f : futs) f.get();
             }
            },
             py::arg("env_ids"),
              py::arg("callbacks"),
              py::arg("sensor_names"),
           R"pbdoc(
            Bind a camera image callback in each environment.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            callbacks : list[callable]
                Callback per environment, invoked with a numpy array image
                (broadcast if size 1).
            sensor_names : list[str]
                Camera sensor name per environment (broadcast if size 1).
          )pbdoc")
      .def("bind_lidar_cb", [](AsyncDRLServerPool& self,
                                  const std::vector<std::size_t>& env_ids,
                                  const std::vector<py::function>& callbacks,
                                  const std::vector<std::string>& sensor_names) {
             const auto n = env_ids.size();
             check_broadcastable(n, sensor_names, "sensor_names");
             check_broadcastable(n, callbacks, "callbacks");
             std::vector<std::shared_ptr<py::function>> callback_holders;
             callback_holders.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               callback_holders.emplace_back(
                   keep_python_callback(py::function(pick(callbacks, i))));
             std::vector<decltype(self.call(std::size_t{}, &DRLServer::bind_lidar_cb,
             std::function<void (const systems::custom_plugins::Sensors::LidarFrameView&)>{},
             std::string{}))> futs;
             futs.reserve(n);
             {
               py::gil_scoped_release release;
               for (std::size_t i = 0; i < n; ++i){
                 auto cb_ptr = callback_holders[i];
                 futs.emplace_back(self.call(env_ids[i], &DRLServer::bind_lidar_cb,
                  [cb_ptr](const systems::custom_plugins::Sensors::LidarFrameView& lidar_data){
                  if (!Py_IsInitialized()) return;
                  py::gil_scoped_acquire gil;
                  if (lidar_data.data){
                    auto array = py::array_t<float>({lidar_data.h, lidar_data.w, lidar_data.c}, lidar_data.data);
                    (*cb_ptr)(array);
                  } else {
                    (*cb_ptr)(py::none());
                  }
                  }, pick(sensor_names, i)));
               }
               for (auto& f : futs) f.get();
             }
            },
             py::arg("env_ids"),
              py::arg("callbacks"),
              py::arg("sensor_names"),
           R"pbdoc(
            Bind a LiDAR data callback in each environment.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            callbacks : list[callable]
                Callback per environment, invoked with a numpy array or None
                (broadcast if size 1).
            sensor_names : list[str]
                LiDAR sensor name per environment (broadcast if size 1).
          )pbdoc")
      // ----------------- mass / inertia -----------------
      .def("set_mass", [](AsyncDRLServerPool& self,
                          const std::vector<std::size_t>& env_ids,
                          const std::vector<std::string>& model_names,
                          const std::vector<std::string>& link_names,
                          const std::vector<double>& masses) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             check_broadcastable(n, link_names, "link_names");
             check_broadcastable(n, masses, "masses");
             std::vector<decltype(self.call(std::size_t{}, &DRLServer::set_mass, std::string{}, std::string{}, 0.0))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], &DRLServer::set_mass,
                                           pick(model_names, i), pick(link_names, i), pick(masses, i)));
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("model_names"), py::arg("link_names"), py::arg("masses"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Set link mass in each environment (SDF cache only).

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            model_names : list[str]
                Model name per environment (broadcast if size 1).
            link_names : list[str]
                Link name per environment (broadcast if size 1).
            masses : list[float]
                New mass in kg per environment (broadcast if size 1).

            Warning
            -------
            Changes take effect only after respawn_model() is called.
          )pbdoc")

      .def("set_inertia", [](AsyncDRLServerPool& self,
                             const std::vector<std::size_t>& env_ids,
                             const std::vector<std::string>& model_names,
                             const std::vector<std::string>& link_names,
                             const std::vector<Eigen::Matrix3d>& inertias) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             check_broadcastable(n, link_names, "link_names");
             check_broadcastable(n, inertias, "inertias");
             using MF = void (DRLServer::*)(std::string, std::string, Eigen::Matrix3d&);
             MF mf = &DRLServer::set_inertia;
             // why: keep per-env storage alive to satisfy lvalue-ref parameter during async dispatch.
             std::vector<std::shared_ptr<Eigen::Matrix3d>> storage; storage.reserve(n);
             std::vector<decltype(self.call(std::size_t{}, mf, std::string{}, std::string{}, std::declval<Eigen::Matrix3d&>()))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i) {
               storage.emplace_back(std::make_shared<Eigen::Matrix3d>(pick(inertias, i)));
               futs.emplace_back(self.call(env_ids[i], mf,
                                           pick(model_names, i), pick(link_names, i), *storage.back()));
             }
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("model_names"), py::arg("link_names"), py::arg("inertias"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Set link inertia tensor in each environment (SDF cache only).

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            model_names : list[str]
                Model name per environment (broadcast if size 1).
            link_names : list[str]
                Link name per environment (broadcast if size 1).
            inertias : list[numpy.ndarray]
                3x3 inertia tensor in kg*m^2 per environment (broadcast if size 1).

            Warning
            -------
            Changes take effect only after respawn_model() is called.
          )pbdoc")
      .def("get_inertia", [](AsyncDRLServerPool& self,
                             const std::vector<std::size_t>& env_ids,
                             const std::vector<std::string>& model_names,
                             const std::vector<std::string>& link_names) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             check_broadcastable(n, link_names, "link_names");
             using FutT = decltype(self.call(std::size_t{}, &DRLServer::get_inertia, std::string{}, std::string{}));
             std::vector<FutT> futs; futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], &DRLServer::get_inertia,
                                           pick(model_names, i), pick(link_names, i)));
             using RetT = decltype(futs[0].get());
             std::vector<RetT> out; out.reserve(n);
             for (auto& f : futs) out.emplace_back(f.get());
             return out;
           }, py::arg("env_ids"), py::arg("model_names"), py::arg("link_names"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Get link inertia tensor for each environment.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to query.
            model_names : list[str]
                Model name per environment (broadcast if size 1).
            link_names : list[str]
                Link name per environment (broadcast if size 1).

            Returns
            -------
            list[numpy.ndarray]
                3x3 inertia tensor in kg*m^2 per environment.
          )pbdoc")
      .def("get_mass", [](AsyncDRLServerPool& self,
                             const std::vector<std::size_t>& env_ids,
                             const std::vector<std::string>& model_names,
                             const std::vector<std::string>& link_names) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             check_broadcastable(n, link_names, "link_names");
             using FutT = decltype(self.call(std::size_t{}, &DRLServer::get_mass, std::string{}, std::string{}));
             std::vector<FutT> futs; futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], &DRLServer::get_mass,
                                           pick(model_names, i), pick(link_names, i)));
             using RetT = decltype(futs[0].get());
             std::vector<RetT> out; out.reserve(n);
             for (auto& f : futs) out.emplace_back(f.get());
             return out;
           }, py::arg("env_ids"), py::arg("model_names"), py::arg("link_names"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Get link mass for each environment.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to query.
            model_names : list[str]
                Model name per environment (broadcast if size 1).
            link_names : list[str]
                Link name per environment (broadcast if size 1).

            Returns
            -------
            list[float]
                Mass in kg per environment.
          )pbdoc")
      .def("set_rotor_parameters", [](AsyncDRLServerPool& self,
                                      const std::vector<std::size_t>& env_ids,
                                      const std::vector<std::string>& model_names,
                                      const std::vector<RotorParameters>& rotor_params) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             check_broadcastable(n, rotor_params, "rotor_params");
             using MF = void (DRLServer::*)(std::string, const RotorParameters&);
             MF mf = &DRLServer::set_rotor_parameters;
             // why: keep per-env storage alive to satisfy lvalue-ref parameter during async dispatch.
             std::vector<std::shared_ptr<RotorParameters>> storage; storage.reserve(n);
             std::vector<decltype(self.call(std::size_t{}, mf, std::string{}, RotorParameters{}))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i) {
               futs.emplace_back(self.call(env_ids[i], mf,
                                           pick(model_names, i), pick(rotor_params, i)));
             }
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("model_names"), py::arg("rotor_params"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Set rotor parameters for domain randomization in each environment.

            Requires the MultiRotorPlugin to be attached; otherwise a no-op.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            model_names : list[str]
                Model name per environment (broadcast if size 1).
            rotor_params : list[RotorParameters]
                Rotor physics parameters per environment (broadcast if size 1).
          )pbdoc")
      .def("get_rotor_parameters", [](AsyncDRLServerPool& self,
                                      const std::vector<std::size_t>& env_ids,
                                      const std::vector<std::string>& model_names) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             using FutT = decltype(self.call(std::size_t{}, &DRLServer::get_rotor_parameters, std::string{}));
             std::vector<FutT> futs; futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], &DRLServer::get_rotor_parameters,
                                           pick(model_names, i)));
             using RetT = decltype(futs[0].get());
             std::vector<RetT> out; out.reserve(n);
             for (auto& f : futs) out.emplace_back(f.get());
             return out;
           }, py::arg("env_ids"), py::arg("model_names"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Get current rotor parameters for each environment.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to query.
            model_names : list[str]
                Model name per environment (broadcast if size 1).

            Returns
            -------
            list[RotorParameters]
                Rotor physics parameters per environment.
          )pbdoc")
      // Get rotor allocation matrix
      .def("get_rotor_thrust_allocation_matrix", [](AsyncDRLServerPool& self,
                                             const std::vector<std::size_t>& env_ids,
                                             const std::vector<std::string>& model_names) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             using FutT = decltype(self.call(std::size_t{}, static_cast<Eigen::MatrixXd (DRLServer::*)(std::string) >(&DRLServer::get_rotor_thrust_allocation_matrix), std::string{}));
             std::vector<FutT> futs; futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], static_cast<Eigen::MatrixXd (DRLServer::*)(std::string) >(&DRLServer::get_rotor_thrust_allocation_matrix),
                                           pick(model_names, i)));
             using RetT = decltype(futs[0].get());
             std::vector<RetT> out; out.reserve(n);
             for (auto& f : futs) out.emplace_back(f.get());
              return out;
            }, py::arg("env_ids"), py::arg("model_names"),
               py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Get the rotor thrust allocation matrix for each environment.

            This overload requires the MultiRotorPlugin to be attached.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to query.
            model_names : list[str]
                Model name per environment (broadcast if size 1).

            Returns
            -------
            list[numpy.ndarray]
                Allocation matrix mapping [T, taux, tauy, tauz] to rotor thrusts,
                per environment.
          )pbdoc")
      // Inverse rotor allocation matrix
      .def("get_inverse_rotor_thrust_allocation_matrix", [](AsyncDRLServerPool& self,
                                                     const std::vector<std::size_t>& env_ids,
                                                     const std::vector<std::string>& model_names) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             using FutT = decltype(self.call(std::size_t{}, static_cast<Eigen::MatrixXd (DRLServer::*)(std::string)>(&DRLServer::get_inverse_rotor_thrust_allocation_matrix), std::string{}));
             std::vector<FutT> futs; futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], static_cast<Eigen::MatrixXd (DRLServer::*)(std::string)>(&DRLServer::get_inverse_rotor_thrust_allocation_matrix),
                                           pick(model_names, i)));
             using RetT = decltype(futs[0].get());
             std::vector<RetT> out; out.reserve(n);
             for (auto& f : futs) out.emplace_back(f.get());
              return out;
            }, py::arg("env_ids"), py::arg("model_names"),
               py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Get the inverse rotor thrust allocation matrix for each environment.

            This overload requires the MultiRotorPlugin to be attached.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to query.
            model_names : list[str]
                Model name per environment (broadcast if size 1).

            Returns
            -------
            list[numpy.ndarray]
                Allocation matrix mapping rotor thrusts to [T, taux, tauy, tauz],
                per environment.
          )pbdoc")
      .def("get_thrust_moment_to_rotor_velocity_mapping_function", [](AsyncDRLServerPool& self,
                                                     const std::vector<std::size_t>& env_ids,
                                                     const std::vector<std::string>& model_names) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             using FutT = decltype(self.call(std::size_t{}, &DRLServer::get_thrust_moment_to_rotor_velocity_mapping_function, std::string{}));
             std::vector<FutT> futs; futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], &DRLServer::get_thrust_moment_to_rotor_velocity_mapping_function,
                                           pick(model_names, i)));
             using RetT = decltype(futs[0].get());
             std::vector<RetT> out; out.reserve(n);
             for (auto& f : futs) out.emplace_back(f.get());
              return out;
            }, py::arg("env_ids"), py::arg("model_names"),
               py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Get the thrust/moment to rotor velocity mapping function per environment.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to query.
            model_names : list[str]
                Model name per environment (broadcast if size 1).

            Returns
            -------
            list[callable]
                One function per environment mapping
                [thrust, roll_moment, pitch_moment, yaw_moment] to rotor velocities.
          )pbdoc")
      // ----------------- contact data requests -----------------
      .def("request_contact_data", [](AsyncDRLServerPool& self,
                                      const std::vector<std::size_t>& env_ids,
                                      const std::vector<std::string>& model_names) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             std::vector<decltype(self.call(std::size_t{}, &DRLServer::request_contact_data, std::string{}))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], &DRLServer::request_contact_data, pick(model_names, i)));
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("model_names"), py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Enable contact data collection for a model in each environment.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            model_names : list[str]
                Model name per environment (broadcast if size 1).

            Notes
            -----
            Has a performance impact. Contact data is retrieved via get_contacts().
          )pbdoc")


      // ----------------- step_size / sim_iterations (per-env getters) -----------------
      .def("step_size", [](AsyncDRLServerPool& self,
                           const std::vector<std::size_t>& env_ids) {
             using FutT = decltype(self.submit(std::size_t{}, [](DRLServer& s){ return s.step_size(); }));
             std::vector<FutT> futs; futs.reserve(env_ids.size());
             for (auto id : env_ids)
               futs.emplace_back(self.submit(id, [](DRLServer& s){ return s.step_size(); }));
             using RetT = decltype(futs[0].get());
             std::vector<RetT> out; out.reserve(futs.size());
             for (auto& f : futs) out.emplace_back(f.get());
             return out;
           }, py::arg("env_ids"), py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Get the physics timestep duration for each environment.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to query.

            Returns
            -------
            list[float]
                Time step in seconds per environment.
          )pbdoc")

      .def("sim_iterations", [](AsyncDRLServerPool& self,
                                const std::vector<std::size_t>& env_ids) {
             using FutT = decltype(self.submit(std::size_t{}, [](DRLServer& s){ return s.sim_iterations(); }));
             std::vector<FutT> futs; futs.reserve(env_ids.size());
             for (auto id : env_ids)
               futs.emplace_back(self.submit(id, [](DRLServer& s){ return s.sim_iterations(); }));
             using RetT = decltype(futs[0].get());
             std::vector<RetT> out; out.reserve(futs.size());
             for (auto& f : futs) out.emplace_back(f.get());
             return out;
           }, py::arg("env_ids"), py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Get the simulation iteration count for each environment.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to query.

            Returns
            -------
            list[int]
                Number of steps executed since start, per environment.
          )pbdoc")

      // ----------------- set_ctbr_cmd -----------------
      .def("set_ctbr_cmd", [](AsyncDRLServerPool& self,
                              const std::vector<std::size_t>& env_ids,
                              const std::vector<std::string>& model_names,
                              const std::vector<std::string>& link_names,
                              const std::vector<Eigen::VectorXd>& cmds) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             check_broadcastable(n, link_names, "link_names");
             check_broadcastable(n, cmds, "cmds");
             using MF = void (DRLServer::*)(std::string, std::string, Eigen::VectorXd&&);
             MF mf = &DRLServer::set_ctbr_cmd;
             std::vector<decltype(self.call(std::size_t{}, mf, std::string{}, std::string{}, Eigen::VectorXd{}))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], mf,
                                           pick(model_names, i), pick(link_names, i),
                                           Eigen::VectorXd(pick(cmds, i))));
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("model_names"), py::arg("link_names"), py::arg("cmds"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Set the CTBR command for a link in each environment.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            model_names : list[str]
                Model name per environment (broadcast if size 1).
            link_names : list[str]
                Link name per environment (broadcast if size 1).
            cmds : list[numpy.ndarray]
                CTBR command [T, wx, wy, wz] per environment (broadcast if size 1).
          )pbdoc")

      .def("set_ctbr_cmd", [](AsyncDRLServerPool& self,
                              const std::vector<std::size_t>& env_ids,
                              const std::vector<std::string>& model_names,
                              const std::vector<std::string>& link_names,
                              const std::vector<Eigen::VectorXd>& cmds,
                              const std::vector<Eigen::VectorXd>& kps,
                              const std::vector<Eigen::VectorXd>& kds) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             check_broadcastable(n, link_names, "link_names");
             check_broadcastable(n, cmds, "cmds");
             check_broadcastable(n, kps, "kps");
             check_broadcastable(n, kds, "kds");
             using MF = void (DRLServer::*)(std::string, std::string, Eigen::VectorXd&&, const Eigen::VectorXd&, const Eigen::VectorXd&);
             MF mf = &DRLServer::set_ctbr_cmd;
             std::vector<decltype(self.call(std::size_t{}, mf, std::string{}, std::string{}, Eigen::VectorXd{}, Eigen::VectorXd{}, Eigen::VectorXd{}))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], mf,
                                           pick(model_names, i), pick(link_names, i),
                                           Eigen::VectorXd(pick(cmds, i)),
                                           Eigen::VectorXd(pick(kps, i)),
                                           Eigen::VectorXd(pick(kds, i))));
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("model_names"), py::arg("link_names"), py::arg("cmds"),
              py::arg("kps"), py::arg("kds"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Set the CTBR command with an inner-loop stabilizer in each environment.

            The inner-loop torque is tau = kp*e_W + kd*e_A + cross(W, J*W).

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            model_names : list[str]
                Model name per environment (broadcast if size 1).
            link_names : list[str]
                Link name per environment (broadcast if size 1).
            cmds : list[numpy.ndarray]
                CTBR command [T, wx, wy, wz] per environment (broadcast if size 1).
            kps : list[numpy.ndarray]
                Proportional gains for the inner stabilizer per environment
                (broadcast if size 1).
            kds : list[numpy.ndarray]
                Derivative gains for the inner stabilizer per environment
                (broadcast if size 1).
          )pbdoc")

      // ----------------- set_ctbt_cmd -----------------
      .def("set_ctbt_cmd", [](AsyncDRLServerPool& self,
                              const std::vector<std::size_t>& env_ids,
                              const std::vector<std::string>& model_names,
                              const std::vector<std::string>& link_names,
                              const std::vector<Eigen::VectorXd>& cmds) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             check_broadcastable(n, link_names, "link_names");
             check_broadcastable(n, cmds, "cmds");
             using MF = void (DRLServer::*)(std::string, std::string, Eigen::VectorXd&&);
             MF mf = &DRLServer::set_ctbt_cmd;
             std::vector<decltype(self.call(std::size_t{}, mf, std::string{}, std::string{}, Eigen::VectorXd{}))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], mf,
                                           pick(model_names, i), pick(link_names, i),
                                           Eigen::VectorXd(pick(cmds, i))));
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("model_names"), py::arg("link_names"), py::arg("cmds"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Set the CTBT command for a link in each environment.

            The CTBT is converted to a world-frame wrench and applied via set_wrench.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            model_names : list[str]
                Model name per environment (broadcast if size 1).
            link_names : list[str]
                Link name per environment (broadcast if size 1).
            cmds : list[numpy.ndarray]
                CTBT command [F, taux, tauy, tauz] per environment (broadcast if size 1).
          )pbdoc")

      // ----------------- set_srt_cmd -----------------
      .def("set_srt_cmd", [](AsyncDRLServerPool& self,
                             const std::vector<std::size_t>& env_ids,
                             const std::vector<std::string>& model_names,
                             const std::vector<std::string>& link_names,
                             const std::vector<std::vector<std::string>>& rotor_links,
                             const std::vector<std::vector<int>>& turning_directions,
                             const std::vector<Eigen::VectorXd>& cmds,
                             const std::vector<double>& ktaus) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             check_broadcastable(n, link_names, "link_names");
             check_broadcastable(n, rotor_links, "rotor_links");
             check_broadcastable(n, turning_directions, "turning_directions");
             check_broadcastable(n, cmds, "cmds");
             check_broadcastable(n, ktaus, "ktaus");
             using MF = void (DRLServer::*)(std::string, std::string, const std::vector<std::string>&, const std::vector<int>&, Eigen::VectorXd&&, double);
             MF mf = &DRLServer::set_srt_cmd;
             std::vector<decltype(self.call(std::size_t{}, mf, std::string{}, std::string{}, std::vector<std::string>{}, std::vector<int>{}, Eigen::VectorXd{}, 0.0))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], mf,
                                           pick(model_names, i), pick(link_names, i),
                                           pick(rotor_links, i), pick(turning_directions, i),
                                           Eigen::VectorXd(pick(cmds, i)), pick(ktaus, i)));
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("model_names"), py::arg("link_names"),
              py::arg("rotor_links"), py::arg("turning_directions"), py::arg("cmds"), py::arg("ktaus"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Set the SRT (single-rotor-thrust) command in each environment.

            Each rotor's yaw torque is -turning_dir * ktau * thrust.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            model_names : list[str]
                Model name per environment (broadcast if size 1).
            link_names : list[str]
                Base/canonical link name per environment (broadcast if size 1).
            rotor_links : list[list[str]]
                Rotor link names per environment (broadcast if size 1).
            turning_directions : list[list[int]]
                Turning direction per rotor, per environment (broadcast if size 1).
            cmds : list[numpy.ndarray]
                Thrust per rotor, per environment (broadcast if size 1).
            ktaus : list[float]
                Motor torque constant per environment (broadcast if size 1).
          )pbdoc")

      // ----------------- reset_world -----------------
      .def("reset_world", [](AsyncDRLServerPool& self,
                             const std::vector<std::size_t>& env_ids,
                             const std::vector<std::unordered_map<std::string, std::pair<Eigen::Vector3d, Eigen::Vector3d>>>& model_poses) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_poses, "model_poses");
             using MapT = std::unordered_map<std::string, std::pair<Eigen::Vector3d, Eigen::Vector3d>>;
             using MF = void (DRLServer::*)(const MapT&);
             MF mf = &DRLServer::reset_world;
             std::vector<decltype(self.call(std::size_t{}, mf, MapT{}))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], mf, pick(model_poses, i)));
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("model_poses"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Reset all models in the world to new poses in each environment.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            model_poses : list[dict[str, tuple[numpy.ndarray, numpy.ndarray]]]
                Per environment, a mapping of model name to
                (position, orientation) pairs (broadcast if size 1).
                Position in meters, orientation as [roll, pitch, yaw] in radians.
          )pbdoc")

      // ----------------- get_thrust_moment_to_rotor_thrust_mapping_function -----------------
      .def("get_thrust_moment_to_rotor_thrust_mapping_function", [](AsyncDRLServerPool& self,
                             const std::vector<std::size_t>& env_ids,
                             const std::vector<std::string>& model_names) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             using MF = std::function<Eigen::VectorXd (const Eigen::Vector4d&)> (DRLServer::*)(std::string);
             MF mf = &DRLServer::get_thrust_moment_to_rotor_thrust_mapping_function;
             using FutT = decltype(self.call(std::size_t{}, mf, std::string{}));
             std::vector<FutT> futs; futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], mf, pick(model_names, i)));
             using RetT = decltype(futs[0].get());
             std::vector<RetT> out; out.reserve(n);
             for (auto& f : futs) out.emplace_back(f.get());
             return out;
           }, py::arg("env_ids"), py::arg("model_names"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Get the thrust/moment to rotor thrust mapping function per environment.

            This overload requires the MultiRotorPlugin to be attached.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to query.
            model_names : list[str]
                Model name per environment (broadcast if size 1).

            Returns
            -------
            list[callable]
                One function per environment mapping
                [thrust, roll_moment, pitch_moment, yaw_moment] to rotor thrusts.
          )pbdoc")

      .def("get_thrust_moment_to_rotor_thrust_mapping_function", [](AsyncDRLServerPool& self,
                             const std::vector<std::size_t>& env_ids,
                             const std::vector<std::string>& model_names,
                             const std::vector<std::vector<std::string>>& rotor_links,
                             const std::vector<std::vector<int>>& turning_directions,
                             const std::vector<double>& ktaus) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             check_broadcastable(n, rotor_links, "rotor_links");
             check_broadcastable(n, turning_directions, "turning_directions");
             check_broadcastable(n, ktaus, "ktaus");
             using MF = std::function<Eigen::VectorXd (const Eigen::Vector4d&)> (DRLServer::*)(std::string, const std::vector<std::string>&, std::vector<int>&, double);
             MF mf = &DRLServer::get_thrust_moment_to_rotor_thrust_mapping_function;
             // why: turning_dir is a non-const lvalue-ref param; keep per-env storage alive and pass an lvalue.
             std::vector<std::shared_ptr<std::vector<int>>> storage; storage.reserve(n);
             using FutT = decltype(self.call(std::size_t{}, mf, std::string{}, std::vector<std::string>{}, std::declval<std::vector<int>&>(), 0.0));
             std::vector<FutT> futs; futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i) {
               storage.emplace_back(std::make_shared<std::vector<int>>(pick(turning_directions, i)));
               futs.emplace_back(self.call(env_ids[i], mf,
                                           pick(model_names, i), pick(rotor_links, i),
                                           *storage.back(), pick(ktaus, i)));
             }
             using RetT = decltype(futs[0].get());
             std::vector<RetT> out; out.reserve(n);
             for (auto& f : futs) out.emplace_back(f.get());
             return out;
           }, py::arg("env_ids"), py::arg("model_names"), py::arg("rotor_links"),
              py::arg("turning_directions"), py::arg("ktaus"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Get the thrust/moment to rotor thrust mapping function per environment.

            This overload uses explicit rotor links and does not require the
            MultiRotorPlugin.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to query.
            model_names : list[str]
                Model name per environment (broadcast if size 1).
            rotor_links : list[list[str]]
                Rotor link names per environment (broadcast if size 1).
            turning_directions : list[list[int]]
                Turning direction per rotor, per environment (broadcast if size 1).
            ktaus : list[float]
                Motor torque constant per environment (broadcast if size 1).

            Returns
            -------
            list[callable]
                One function per environment mapping
                [thrust, roll_moment, pitch_moment, yaw_moment] to rotor thrusts.
          )pbdoc")

      // ----------------- get_rotor_thrust_allocation_matrix (explicit rotor links) -----------------
      .def("get_rotor_thrust_allocation_matrix", [](AsyncDRLServerPool& self,
                             const std::vector<std::size_t>& env_ids,
                             const std::vector<std::string>& model_names,
                             const std::vector<std::vector<std::string>>& rotor_links,
                             const std::vector<std::vector<int>>& turning_directions,
                             const std::vector<double>& ktaus) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             check_broadcastable(n, rotor_links, "rotor_links");
             check_broadcastable(n, turning_directions, "turning_directions");
             check_broadcastable(n, ktaus, "ktaus");
             using MF = Eigen::MatrixXd (DRLServer::*)(std::string, const std::vector<std::string>&, const std::vector<int>&, double);
             MF mf = &DRLServer::get_rotor_thrust_allocation_matrix;
             using FutT = decltype(self.call(std::size_t{}, mf, std::string{}, std::vector<std::string>{}, std::vector<int>{}, 0.0));
             std::vector<FutT> futs; futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], mf,
                                           pick(model_names, i), pick(rotor_links, i),
                                           pick(turning_directions, i), pick(ktaus, i)));
             using RetT = decltype(futs[0].get());
             std::vector<RetT> out; out.reserve(n);
             for (auto& f : futs) out.emplace_back(f.get());
             return out;
           }, py::arg("env_ids"), py::arg("model_names"), py::arg("rotor_links"),
              py::arg("turning_directions"), py::arg("ktaus"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Get the rotor thrust allocation matrix per environment (explicit links).

            This overload uses explicit rotor links and does not require the
            MultiRotorPlugin.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to query.
            model_names : list[str]
                Model name per environment (broadcast if size 1).
            rotor_links : list[list[str]]
                Rotor link names per environment (broadcast if size 1).
            turning_directions : list[list[int]]
                Turning direction per rotor, per environment (broadcast if size 1).
            ktaus : list[float]
                Motor torque constant per environment (broadcast if size 1).

            Returns
            -------
            list[numpy.ndarray]
                Allocation matrix mapping [T, taux, tauy, tauz] to rotor thrusts,
                per environment.
          )pbdoc")

      // ----------------- get_inverse_rotor_thrust_allocation_matrix (explicit rotor links) -----------------
      .def("get_inverse_rotor_thrust_allocation_matrix", [](AsyncDRLServerPool& self,
                             const std::vector<std::size_t>& env_ids,
                             const std::vector<std::string>& model_names,
                             const std::vector<std::vector<std::string>>& rotor_links,
                             const std::vector<std::vector<int>>& turning_directions,
                             const std::vector<double>& ktaus) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             check_broadcastable(n, rotor_links, "rotor_links");
             check_broadcastable(n, turning_directions, "turning_directions");
             check_broadcastable(n, ktaus, "ktaus");
             using MF = Eigen::MatrixXd (DRLServer::*)(std::string, const std::vector<std::string>&, const std::vector<int>&, double);
             MF mf = &DRLServer::get_inverse_rotor_thrust_allocation_matrix;
             using FutT = decltype(self.call(std::size_t{}, mf, std::string{}, std::vector<std::string>{}, std::vector<int>{}, 0.0));
             std::vector<FutT> futs; futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], mf,
                                           pick(model_names, i), pick(rotor_links, i),
                                           pick(turning_directions, i), pick(ktaus, i)));
             using RetT = decltype(futs[0].get());
             std::vector<RetT> out; out.reserve(n);
             for (auto& f : futs) out.emplace_back(f.get());
             return out;
           }, py::arg("env_ids"), py::arg("model_names"), py::arg("rotor_links"),
              py::arg("turning_directions"), py::arg("ktaus"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Get the inverse rotor thrust allocation matrix per environment (explicit links).

            This overload uses explicit rotor links and does not require the
            MultiRotorPlugin.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to query.
            model_names : list[str]
                Model name per environment (broadcast if size 1).
            rotor_links : list[list[str]]
                Rotor link names per environment (broadcast if size 1).
            turning_directions : list[list[int]]
                Turning direction per rotor, per environment (broadcast if size 1).
            ktaus : list[float]
                Motor torque constant per environment (broadcast if size 1).

            Returns
            -------
            list[numpy.ndarray]
                Allocation matrix mapping rotor thrusts to [T, taux, tauy, tauz],
                per environment.
          )pbdoc")

      // ----------------- set_srt_rate_limiter_time_constants -----------------
      .def("set_srt_rate_limiter_time_constants", [](AsyncDRLServerPool& self,
                             const std::vector<std::size_t>& env_ids,
                             const std::vector<std::string>& model_names,
                             const std::vector<std::string>& link_names,
                             const std::vector<double>& tau_ups,
                             const std::vector<double>& tau_downs,
                             const std::vector<Eigen::VectorXd>& initial_values) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             check_broadcastable(n, link_names, "link_names");
             check_broadcastable(n, tau_ups, "tau_up");
             check_broadcastable(n, tau_downs, "tau_down");
             check_broadcastable(n, initial_values, "initial_value");
             using MF = void (DRLServer::*)(std::string, std::string, double, double, const Eigen::VectorXd&);
             MF mf = &DRLServer::set_srt_rate_limiter_time_constants;
             std::vector<decltype(self.call(std::size_t{}, mf, std::string{}, std::string{}, 0.0, 0.0, Eigen::VectorXd{}))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], mf,
                                           pick(model_names, i), pick(link_names, i),
                                           pick(tau_ups, i), pick(tau_downs, i),
                                           Eigen::VectorXd(pick(initial_values, i))));
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("model_names"), py::arg("link_names"),
              py::arg("tau_up"), py::arg("tau_down"), py::arg("initial_value"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Set the SRT first-order rate limiter time constants in each environment.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            model_names : list[str]
                Model name per environment (broadcast if size 1).
            link_names : list[str]
                Link name per environment (broadcast if size 1).
            tau_up : list[float]
                Rising time constant per environment (broadcast if size 1).
            tau_down : list[float]
                Falling time constant per environment (broadcast if size 1).
            initial_value : list[numpy.ndarray]
                Initial command value per environment (broadcast if size 1).
          )pbdoc")

      // ----------------- set_ctbr_rate_limiter_time_constants -----------------
      .def("set_ctbr_rate_limiter_time_constants", [](AsyncDRLServerPool& self,
                             const std::vector<std::size_t>& env_ids,
                             const std::vector<std::string>& model_names,
                             const std::vector<std::string>& link_names,
                             const std::vector<double>& tau_ups,
                             const std::vector<double>& tau_downs,
                             const std::vector<Eigen::VectorXd>& initial_values) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             check_broadcastable(n, link_names, "link_names");
             check_broadcastable(n, tau_ups, "tau_up");
             check_broadcastable(n, tau_downs, "tau_down");
             check_broadcastable(n, initial_values, "initial_value");
             using MF = void (DRLServer::*)(std::string, std::string, double, double, const Eigen::VectorXd&);
             MF mf = &DRLServer::set_ctbr_rate_limiter_time_constants;
             std::vector<decltype(self.call(std::size_t{}, mf, std::string{}, std::string{}, 0.0, 0.0, Eigen::VectorXd{}))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], mf,
                                           pick(model_names, i), pick(link_names, i),
                                           pick(tau_ups, i), pick(tau_downs, i),
                                           Eigen::VectorXd(pick(initial_values, i))));
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("model_names"), py::arg("link_names"),
              py::arg("tau_up"), py::arg("tau_down"), py::arg("initial_value"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Set the CTBR first-order rate limiter time constants in each environment.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            model_names : list[str]
                Model name per environment (broadcast if size 1).
            link_names : list[str]
                Link name per environment (broadcast if size 1).
            tau_up : list[float]
                Rising time constant per environment (broadcast if size 1).
            tau_down : list[float]
                Falling time constant per environment (broadcast if size 1).
            initial_value : list[numpy.ndarray]
                Initial command value per environment (broadcast if size 1).
          )pbdoc")

      // ----------------- set_ctbt_rate_limiter_time_constants -----------------
      .def("set_ctbt_rate_limiter_time_constants", [](AsyncDRLServerPool& self,
                             const std::vector<std::size_t>& env_ids,
                             const std::vector<std::string>& model_names,
                             const std::vector<std::string>& link_names,
                             const std::vector<double>& tau_ups,
                             const std::vector<double>& tau_downs,
                             const std::vector<Eigen::VectorXd>& initial_values) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             check_broadcastable(n, link_names, "link_names");
             check_broadcastable(n, tau_ups, "tau_up");
             check_broadcastable(n, tau_downs, "tau_down");
             check_broadcastable(n, initial_values, "initial_value");
             using MF = void (DRLServer::*)(std::string, std::string, double, double, const Eigen::VectorXd&);
             MF mf = &DRLServer::set_ctbt_rate_limiter_time_constants;
             std::vector<decltype(self.call(std::size_t{}, mf, std::string{}, std::string{}, 0.0, 0.0, Eigen::VectorXd{}))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], mf,
                                           pick(model_names, i), pick(link_names, i),
                                           pick(tau_ups, i), pick(tau_downs, i),
                                           Eigen::VectorXd(pick(initial_values, i))));
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("model_names"), py::arg("link_names"),
              py::arg("tau_up"), py::arg("tau_down"), py::arg("initial_value"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Set the CTBT first-order rate limiter time constants in each environment.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            model_names : list[str]
                Model name per environment (broadcast if size 1).
            link_names : list[str]
                Link name per environment (broadcast if size 1).
            tau_up : list[float]
                Rising time constant per environment (broadcast if size 1).
            tau_down : list[float]
                Falling time constant per environment (broadcast if size 1).
            initial_value : list[numpy.ndarray]
                Initial command value per environment (broadcast if size 1).
          )pbdoc")

      // ----------------- reset_srt_rate_limiter -----------------
      .def("reset_srt_rate_limiter", [](AsyncDRLServerPool& self,
                             const std::vector<std::size_t>& env_ids,
                             const std::vector<std::string>& model_names,
                             const std::vector<std::string>& link_names) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             check_broadcastable(n, link_names, "link_names");
             std::vector<decltype(self.call(std::size_t{}, &DRLServer::reset_srt_rate_limiter, std::string{}, std::string{}))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], &DRLServer::reset_srt_rate_limiter,
                                           pick(model_names, i), pick(link_names, i)));
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("model_names"), py::arg("link_names"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Reset the SRT rate limiter state in each environment.

            The time constants are unchanged; the state is reset to its initial
            value.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            model_names : list[str]
                Model name per environment (broadcast if size 1).
            link_names : list[str]
                Link name per environment (broadcast if size 1).
          )pbdoc")

      // ----------------- reset_ctbr_rate_limiter -----------------
      .def("reset_ctbr_rate_limiter", [](AsyncDRLServerPool& self,
                             const std::vector<std::size_t>& env_ids,
                             const std::vector<std::string>& model_names,
                             const std::vector<std::string>& link_names) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             check_broadcastable(n, link_names, "link_names");
             std::vector<decltype(self.call(std::size_t{}, &DRLServer::reset_ctbr_rate_limiter, std::string{}, std::string{}))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], &DRLServer::reset_ctbr_rate_limiter,
                                           pick(model_names, i), pick(link_names, i)));
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("model_names"), py::arg("link_names"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Reset the CTBR rate limiter state in each environment.

            The time constants are unchanged; the state is reset to its initial
            value.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            model_names : list[str]
                Model name per environment (broadcast if size 1).
            link_names : list[str]
                Link name per environment (broadcast if size 1).
          )pbdoc")

      // ----------------- reset_ctbt_rate_limiter -----------------
      .def("reset_ctbt_rate_limiter", [](AsyncDRLServerPool& self,
                             const std::vector<std::size_t>& env_ids,
                             const std::vector<std::string>& model_names,
                             const std::vector<std::string>& link_names) {
             const auto n = env_ids.size();
             check_broadcastable(n, model_names, "model_names");
             check_broadcastable(n, link_names, "link_names");
             std::vector<decltype(self.call(std::size_t{}, &DRLServer::reset_ctbt_rate_limiter, std::string{}, std::string{}))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], &DRLServer::reset_ctbt_rate_limiter,
                                           pick(model_names, i), pick(link_names, i)));
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("model_names"), py::arg("link_names"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Reset the CTBT rate limiter state in each environment.

            The time constants are unchanged; the state is reset to its initial
            value.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            model_names : list[str]
                Model name per environment (broadcast if size 1).
            link_names : list[str]
                Link name per environment (broadcast if size 1).
          )pbdoc")

      // ----------------- camera recording -----------------
      .def("start_camera_recording", [](AsyncDRLServerPool& self,
                             const std::vector<std::size_t>& env_ids,
                             const std::vector<std::string>& cam_names,
                             const std::vector<int>& heights,
                             const std::vector<int>& widths,
                             const std::vector<int>& fps,
                             const std::vector<Eigen::Vector3d>& positions,
                             const std::vector<Eigen::Vector3d>& orientations,
                             const std::vector<std::string>& output_files) {
             const auto n = env_ids.size();
             check_broadcastable(n, cam_names, "cam_names");
             check_broadcastable(n, heights, "heights");
             check_broadcastable(n, widths, "widths");
             check_broadcastable(n, fps, "fps");
             check_broadcastable(n, positions, "positions");
             check_broadcastable(n, orientations, "orientations");
             check_broadcastable(n, output_files, "output_files");
             std::vector<decltype(self.call(std::size_t{}, &DRLServer::start_camera_recording,
                                            std::string{}, 0, 0, 0, Eigen::Vector3d{}, Eigen::Vector3d{}, std::string{}))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], &DRLServer::start_camera_recording,
                                           pick(cam_names, i),
                                           pick(heights, i), pick(widths, i), pick(fps, i),
                                           Eigen::Vector3d(pick(positions, i)),
                                           Eigen::Vector3d(pick(orientations, i)),
                                           pick(output_files, i)));
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("cam_names"), py::arg("heights"), py::arg("widths"),
              py::arg("fps"), py::arg("positions"), py::arg("orientations"), py::arg("output_files"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Start a new camera recording in each environment.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            cam_names : list[str]
                Unique camera name per environment (broadcast if size 1).
            heights : list[int]
                Frame height in pixels per environment (broadcast if size 1).
            widths : list[int]
                Frame width in pixels per environment (broadcast if size 1).
            fps : list[int]
                Frames per second per environment (broadcast if size 1).
            positions : list[numpy.ndarray]
                Initial camera position [x, y, z] in meters per environment
                (broadcast if size 1).
            orientations : list[numpy.ndarray]
                Initial camera orientation [roll, pitch, yaw] in radians per
                environment (broadcast if size 1).
            output_files : list[str]
                Output video file path (.mp4, .avi, or .ogv) per environment.

            Warning
            -------
            Broadcasting a single output file makes every environment write to
            the same path; pass one distinct file per environment.
          )pbdoc")

      .def("update_camera_recording_pose", [](AsyncDRLServerPool& self,
                             const std::vector<std::size_t>& env_ids,
                             const std::vector<std::string>& cam_names,
                             const std::vector<Eigen::Vector3d>& positions,
                             const std::vector<Eigen::Vector3d>& orientations) {
             const auto n = env_ids.size();
             check_broadcastable(n, cam_names, "cam_names");
             check_broadcastable(n, positions, "positions");
             check_broadcastable(n, orientations, "orientations");
             std::vector<decltype(self.call(std::size_t{}, &DRLServer::update_camera_recording_pose,
                                            std::string{}, Eigen::Vector3d{}, Eigen::Vector3d{}))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], &DRLServer::update_camera_recording_pose,
                                           pick(cam_names, i),
                                           Eigen::Vector3d(pick(positions, i)),
                                           Eigen::Vector3d(pick(orientations, i))));
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("cam_names"), py::arg("positions"), py::arg("orientations"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Update the recording camera pose in each environment.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            cam_names : list[str]
                Camera name per environment (broadcast if size 1).
            positions : list[numpy.ndarray]
                New position [x, y, z] in meters per environment (broadcast if size 1).
            orientations : list[numpy.ndarray]
                New orientation [roll, pitch, yaw] in radians per environment
                (broadcast if size 1).
          )pbdoc")

      .def("stop_camera_recording", [](AsyncDRLServerPool& self,
                             const std::vector<std::size_t>& env_ids,
                             const std::vector<std::string>& cam_names) {
             const auto n = env_ids.size();
             check_broadcastable(n, cam_names, "cam_names");
             std::vector<decltype(self.call(std::size_t{}, &DRLServer::stop_camera_recording, std::string{}))> futs;
             futs.reserve(n);
             for (std::size_t i = 0; i < n; ++i)
               futs.emplace_back(self.call(env_ids[i], &DRLServer::stop_camera_recording,
                                           pick(cam_names, i)));
             for (auto& f : futs) f.get();
           }, py::arg("env_ids"), py::arg("cam_names"),
              py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
            Stop recording and finalize the video file in each environment.

            Parameters
            ----------
            env_ids : list[int]
                Environment indices to operate on.
            cam_names : list[str]
                Camera name per environment (broadcast if size 1).
          )pbdoc")

      // ----------------- internal servers prop -----------------
      .def_property_readonly("servers",
        py::cpp_function([](AsyncDRLServerPool& self){ return self.servers_; },
        py::call_guard<py::gil_scoped_release>()),
        "list[DRLServer]: The internal DRLServer instances managed by the pool.");

}
