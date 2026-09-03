// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <string.h>
#include "async_rl_server.hh"
#include "rl_server.hh"
#include "controllers/uav_controllers.hh"
#include "controllers/geometric_controller.hh"
#include "controllers/inner_loop_controller.hh"
#include "controllers/smc_controller.hh"
#include "controllers/pid_controller.hh"
#include "second_order_lp_filter.hh"
#include "common/waypoint_generator.hh"
#include "controllers/tuned_gains.hh"
#include "pybind_common.hh"

void bind_async_drl_server_pool(py::module_ &m);

namespace py = pybind11;
/**
 * @brief Converts a std::vector to an Eigen dynamic column vector
 *
 * @tparam T Eigen scalar type (float, double, etc.)
 * @param array Vector
 * @return Eigen::Matrix<T, Eigen::Dynamic, 1>  converted eigen matrix
 */
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> convert_array_to_eigen(const std::vector<T> &array)
{
    Eigen::Matrix<T, Eigen::Dynamic, 1> res(array.size());
    memcpy(res.data(), array.data(), array.size() * sizeof(T));
    return res;
}
/**
 * @brief Converts a std::vector to an Eigen fixed-size column vector
 *
 * @tparam T Eigen scalar type
 * @tparam N Fixed size dim
 * @param array Vector
 * @return Eigen::Matrix<T, N, 1> converted eigen matrix
 */
template <typename T, size_t N>
Eigen::Matrix<T, N, 1> convert_array_to_eigen(const std::vector<T> &array)
{
    Eigen::Matrix<T, N, 1> res(array.size());
    memcpy(res.data(), array.data(), array.size() * sizeof(T));
    return res;
}

/**
 * @brief trampoline class for UAVController
 *
 */
class PyUAVController : public UAVController
{
public:
    using UAVController::UAVController;
    /**
     * @brief Override for calculate_force method
     *
     * @param state state
     * @param state_dot state dot
     * @param desired desired state
     * @return Eigen::Vector3d Force from outer loop
     */
    Eigen::Vector3d calculate_force(const Stated &state,
                                    const Stated &state_dot,
                                    const Stated &desired) override
    {
        PYBIND11_OVERRIDE_PURE(
            Eigen::Vector3d, // Return type
            UAVController,   // Parent class
            calculate_force, // Name of function in C++
            state,           // Arguments
            state_dot,
            desired);
    }
    /**
     * @brief Override for calculate_moments method
     *
     * @param state state
     * @param state_dot state dot
     * @param desired desired state
     * @param control_thrust thrust calculated using calculate_force
     * @return Eigen::Vector3d calculated moments
     */
    Eigen::Vector3d calculate_moments(const Stated &state,
                                      const Stated &state_dot,
                                      const Stated &desired,
                                      const Eigen::Vector3d &control_thrust) override
    {
        PYBIND11_OVERRIDE_PURE(
            Eigen::Vector3d,   // Return type
            UAVController,     // Parent class
            calculate_moments, // Name of function in C++
            state,             // Arguments
            state_dot,
            desired,
            control_thrust);
    }
};

/** @brief Python bindings for the private gzdrl._core module */
PYBIND11_MODULE(_core, m)
{
    m.doc() = R"pbdoc(
        GzDRL Python Bindings
        ========================
        High-performance deep reinforcement learning interface for Gazebo simulation.

        This module provides Python bindings for the GzDRL C++ library, enabling
        efficient multi-agent reinforcement learning with the Gazebo physics simulator.

        Quick Start
        -----------
        .. code-block:: python

            import gzdrl
            import numpy as np

            # Create server
            server = gzdrl.DRLServer(
                "partition",
                str(gzdrl.get_sdf_path("world_hover.sdf")),
                ["quadrotor"],
                False  # enable_sensors
            )

            # Control loop
            for step in range(1000):
                # Get state
                states = server.state_info("quadrotor")
                pos = states["quadrotor/base_link"][:3]

                # Apply action
                action = np.array([100., 100., 100., 100.])
                server.set_rotor_velocity_cmd(
                    "quadrotor", "quadrotor/base_link", action
                )

                # Step simulation
                server.run_once()

        State Vector Format (1)
        -------------------
        The state is a 19-element numpy array:

        - ``[0:3]``   Position (x, y, z) in meters
        - ``[3:7]``   Orientation (qw, qx, qy, qz) as quaternion
        - ``[7:10]``  Linear velocity (vx, vy, vz) in m/s
        - ``[10:13]`` Angular velocity (wx, wy, wz) in rad/s (body frame)
        - ``[13:16]`` Linear acceleration (ax, ay, az) in m/s²
        - ``[16:19]`` Angular acceleration in rad/s² (body frame)

        State Vector Format (2)
        -------------------
        Controller state is provided as a tuple of 13-element NumPy arrays
        representing the state and its derivative:

        Element 0:
            - ``[0:3]``   Position (x, y, z) in meters
            - ``[3:6]``   Linear velocity (vx, vy, vz) in m/s 
            - ``[6:10]``  Orientation (qw, qx, qy, qz) as quaternion
            - ``[10:13]`` Angular velocity (wx, wy, wz) in rad/s (body frame)
        Element 1:
             - ``[0:3]``   Linear velocity (vx, vy, vz) in m/s
            - ``[3:6]``    Linear acceleration (ax, ay, az) in m/s²
            - ``[6:10]``   Orientation derivative (qw_dot, qx_dot, qy_dot, qz_dot) 
            - ``[10:13]``  Angular acceleration in rad/s² (body frame)
        
        Classes
        -------
        - :class:`DRLServer` - Main simulation server interface for DRL
        - :class:`AsyncDRLServerPool` - Parallel environment pool for training
        - :class:`DRLServerConfig` - Server configuration options
        - :class:`RotorParameters` - Rotor physics parameters (for use with MultiRotorPlugin)
        - :class:`WaypointGenerator` - Generates waypoints along parametric curves
        - :class:`UAVController` - Base controller for UAV models
        - :class:`GeometricController` - PD controller based on nonlinear Geometric law
        - :class:`PIDController` - PID controller (linear)
        - :class:`SlidingModeController` - SMC controller (nonlinear)
        - :class:`InnerLoopController` - Inner-loop attitude controller (linear PD)
        - :class:`AsyncToken` - Async operation handle

        See Also
        --------
        - Gazebo Simulation: https://gazebosim.org
        The versioned user documentation is maintained in the independent
        gzdrl documentation repository.
    )pbdoc";

    using FeasibilityLimits = WaypointGenerator::FeasibilityLimits;
    py::class_<FeasibilityLimits>(m, "FeasibilityLimits")
        .def(py::init<>())
        .def_readwrite("min_speed", &FeasibilityLimits::min_speed)
        .def_readwrite("max_speed", &FeasibilityLimits::max_speed)

        .def_readwrite("max_normal_acceleration", &FeasibilityLimits::max_normal_acceleration)

        .def_readwrite("gravity", &FeasibilityLimits::gravity)
        .def_readwrite("max_specific_thrust", &FeasibilityLimits::max_specific_thrust)
        .def_readwrite("max_tilt_rad", &FeasibilityLimits::max_tilt_rad)

        .def_readwrite("safety_factor", &FeasibilityLimits::safety_factor)

        .def_readwrite("max_tangent_change_rad", &FeasibilityLimits::max_tangent_change_rad)

        .def_readwrite("feasibility_samples", &FeasibilityLimits::feasibility_samples)
        .def_readwrite("integration_arc_step", &FeasibilityLimits::integration_arc_step)
        .def_readwrite("max_parameter_step", &FeasibilityLimits::max_parameter_step)
        .def_readwrite("min_parameter_speed", &FeasibilityLimits::min_parameter_speed)

        .def_readwrite("max_heading_rate_rad_s", &FeasibilityLimits::max_heading_rate_rad_s)
        .def_readwrite("max_yaw_rate_rad_s", &FeasibilityLimits::max_yaw_rate_rad_s);
    /* @brief WaypointGenerator class for generating waypoints along parametric curves */
    using Point = WaypointGenerator::Point;
    py::class_<WaypointGenerator>(m, "WaypointGenerator",
                                  R"pbdoc(
        Waypoint Generator for Parametric Curves

        This class provides methods to generate waypoints along various
        parametric curves, such as circles and Lissajous curves. The generated
        waypoints can be used for trajectory planning in simulations.

        Methods
        -------
        GenerateCircleWaypoints(radius, center_x, center_y, min_dist, max_dist, z, feasibility_lims)
            Generates waypoints along a circle with random arc lengths.

        GenerateLissajousWaypoints(..., feasibility_lims)
            Generates 2D or 3D Lissajous waypoints through overloaded methods.

        Example
        -------
        >>> waypoint_gen = gzdrl.WaypointGenerator()
        >>> limits = gzdrl.FeasibilityLimits()
        >>> waypoints = waypoint_gen.GenerateCircleWaypoints(
        ...     5.0, 0.0, 0.0, 0.5, 1.5, 2.0, limits
        ... )
    )pbdoc")
        .def(py::init<>())
        .def("GenerateCircleWaypoints", 
            static_cast< std::vector<Point> (WaypointGenerator::*) (float,
                 float, float, float, float, float,
                  const FeasibilityLimits&)>(&WaypointGenerator::GenerateCircleWaypoints),
                   py::arg("radius"),
             py::arg("center_x"),
             py::arg("center_y"),
             py::arg("min_dist"),
             py::arg("max_dist"),
             py::arg("z"),
             py::arg("feasibility_lims"),
             py::call_guard<py::gil_scoped_release>(),
             "Generates N waypoints along a circle spaced by random arc lengths.")
        .def("GenerateLissajousWaypoints", static_cast< std::vector<Point> (WaypointGenerator::*) (float,
             float, float, float, float, float, float, float, float, float, float, float, 
              const FeasibilityLimits&)>( &WaypointGenerator::GenerateLissajousWaypoints), py::arg("A"),
             py::arg("B"),
             py::arg("C"),
             py::arg("a"),
             py::arg("b"),
             py::arg("c"),
             py::arg("delta_x"),
             py::arg("delta_y"),
             py::arg("delta_z"),
             py::arg("min_dist"),
             py::arg("max_dist"),
             py::arg("z_offset"),
              py::arg("feasibility_lims"),
             py::call_guard<py::gil_scoped_release>(),
             "Generates N waypoints along a Lissajous curve spaced by random distances.")
        .def("GenerateLissajousWaypoints", static_cast< std::vector<Point> (WaypointGenerator::*) (float,
             float,  float, float,  float, float,  float, float, float,
              const FeasibilityLimits&)>( &WaypointGenerator::GenerateLissajousWaypoints), 
              py::arg("A"),
             py::arg("B"),
             py::arg("a"),
             py::arg("b"),
             py::arg("delta_x"),
             py::arg("delta_y"),
             py::arg("min_dist"),
             py::arg("max_dist"),
             py::arg("z_offset"),
             py::arg("feasibility_lims"),
             py::call_guard<py::gil_scoped_release>())
        .def("PI", [](const WaypointGenerator &)
             { return WaypointGenerator::PI(); }, "Mathematical constant Pi.")
        .def("EPSILON", [](const WaypointGenerator &)
             { return WaypointGenerator::EPSILON(); }, "Small constant for numerical stability.")
        .def("calculate_period", static_cast<double (WaypointGenerator::*) (double, double) const>(&WaypointGenerator::calculate_period), py::arg("freq_a"), py::arg("freq_b"), "Calculates the time T required to complete one full closed cycle for 2 frequencies.")
        .def("calculate_period", static_cast<double (WaypointGenerator::*) (double, double, double) const>(&WaypointGenerator::calculate_period), py::arg("freq_a"), py::arg("freq_b"), py::arg("freq_c"), "Calculates the time T required to complete one full closed cycle for 3 frequencies.");

    py::class_<RotorParameters>(m, "RotorParameters", R"pbdoc(
        Rotor physics parameters for UAV models.

        This class holds the physical parameters of a rotor, used for
        domain randomization and accurate rotor dynamics simulation. Requires MultiRotorPlugin to be
        attached to the model.

        Attributes
        ----------
        max_rot_velocity : float
            Maximum rotational velocity in rad/s.
        thrust_constant_quadratic_params : list[float]
            Quadratic parameters for thrust constant computation.
        torque_constant_quadratic_params : list[float]
            Quadratic parameters for torque constant computation.
        ground_effect_constant : float
            Ground effect coefficient.
        time_constant_up : float
            Time constant for rotor spin-up (s).
        time_constant_down : float
            Time constant for rotor spin-down (s).
        rotor_drag_coefficient : float
            Drag coefficient of the rotor blade.
        rotor_inertia : float
            Moment of inertia of the rotor (kg·m²).
        rolling_moment_coefficient : float
            Rolling moment coefficient.

        Example
        -------
        >>> params = gzdrl.RotorParameters()
        >>> params.max_rot_velocity = 1100.0
        >>> params.time_constant_up = 0.0125
        >>> server.set_rotor_parameters("quadrotor", params)
    )pbdoc")
        .def(py::init<>(), "Create default rotor parameters.")
        .def_readwrite("max_rot_velocity", &RotorParameters::max_rot_velocity,
                       "Maximum rotational velocity in rad/s.")
        .def_readwrite("thrust_constant_quadratic_params", &RotorParameters::thrust_constant_quadratic_params,
                       "Quadratic parameters for thrust constant.")
        .def_readwrite("torque_constant_quadratic_params", &RotorParameters::torque_constant_quadratic_params,
                       "Quadratic parameters for torque constant.")
        .def_readwrite("ground_effect_constant", &RotorParameters::ground_effect_constant,
                       "Ground effect coefficient.")
        .def_readwrite("time_constant_up", &RotorParameters::time_constant_up,
                       "Time constant for rotor spin-up in seconds.")
        .def_readwrite("time_constant_down", &RotorParameters::time_constant_down,
                       "Time constant for rotor spin-down in seconds.")
        .def_readwrite("rotor_drag_coefficient", &RotorParameters::rotor_drag_coefficient,
                       "Drag coefficient of the rotor blade.")
        .def_readwrite("rotor_inertia", &RotorParameters::rotor_inertia,
                       "Moment of inertia of the rotor in kg·m².")
        .def_readwrite("rolling_moment_coefficient", &RotorParameters::rolling_moment_coefficient,
                       "Rolling moment coefficient.");

    py::class_<DRLServerConfig>(m, "DRLServerConfig", R"pbdoc(
        Configuration options for DRLServer.

        This class provides configuration settings for trajectory visualization
        and marker display in the Gazebo simulation.

        Attributes
        ----------
        trajectory_viz : bool
            Enable/disable trajectory visualization markers.
        max_markers : int
            Maximum number of trajectory markers to display.
        marker_interval : int
            Interval between marker updates (in simulation steps).
        color : numpy.ndarray
            A 4x4 matrix of RGBA rows for ambient, diffuse, specular, and
            emissive marker material colors.

        Example
        -------
        >>> config = gzdrl.DRLServerConfig()
        >>> config.trajectory_viz = True
        >>> config.max_markers = 100
        >>> config.color = np.array([
        ...     [1.0, 0.0, 0.0, 1.0],  # ambient
        ...     [1.0, 0.0, 0.0, 1.0],  # diffuse
        ...     [1.0, 1.0, 1.0, 1.0],  # specular
        ...     [1.0, 0.0, 0.0, 1.0],  # emissive
        ... ], dtype=np.float32)
        >>> server = gzdrl.DRLServer(
        ...     "part",
        ...     str(gzdrl.get_sdf_path("world_hover.sdf")),
        ...     ["quadrotor"],
        ...     False,
        ...     config,
        ... )
    )pbdoc")
        .def(py::init<>(), "Create default server configuration.")
        .def_readwrite("trajectory_viz", &DRLServerConfig::trajectory_viz,
                       "Enable/disable trajectory visualization markers.")
        .def_readwrite("max_markers", &DRLServerConfig::max_markers,
                       "Maximum number of trajectory markers to display.")
        .def_readwrite("marker_interval", &DRLServerConfig::marker_interval,
                       "Interval between marker updates in simulation steps.")
        .def_readwrite("color", &DRLServerConfig::color,
                       "4x4 RGBA rows: ambient, diffuse, specular, emissive.");

    py::class_<gz::msgs::Entity>(m, "Entity", R"pbdoc(
        Gazebo entity wrapper for contact sensor messages.

        Represents an entity (model, link, collision, etc.) in the simulation.
        Used primarily for parsing contact sensor data.

        Methods
        -------
        name()
            Get the name of the entity.
        id()
            Get the unique identifier of the entity.
    )pbdoc")
        .def(py::init<>(), "Create an empty entity.")
        .def("name", &gz::msgs::Entity::name,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Get the name of the entity.

            Returns
            -------
            str
                The entity name.
           )pbdoc")
        .def("id", &gz::msgs::Entity::id,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Get the unique identifier of the entity.

            Returns
            -------
            int
                The entity ID.
           )pbdoc");

    py::class_<gz::msgs::Contact>(m, "Contact", R"pbdoc(
        Contact information between two collision geometries.

        Represents a contact point detected by the physics engine between
        two collision geometries in the simulation.

        Methods
        -------
        collision1()
            Get the first collision entity.
        collision2()
            Get the second collision entity.
    )pbdoc")
        .def(py::init<>(), "Create an empty contact.")
        .def("collision1", &gz::msgs::Contact::collision1,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Get the first collision entity in this contact.

            Returns
            -------
            Entity
                The first collision entity.
          )pbdoc")
        .def("collision2", &gz::msgs::Contact::collision2,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Get the second collision entity in this contact.

            Returns
            -------
            Entity
                The second collision entity.
           )pbdoc");

    py::class_<gz::msgs::Contacts>(m, "Contacts", R"pbdoc(
        Collection of contact points from collision detection.

        Contains all contact points detected in a single simulation step
        for a particular collision sensor.

        Methods
        -------
        contact_size()
            Get the number of contacts.
        contact(idx)
            Get a specific contact by index.

        Example
        -------
        >>> contacts = server.get_contacts("quadrotor")
        >>> for link_name, contact_list in contacts.items():
        ...     for c in contact_list:
        ...         for i in range(c.contact_size()):
        ...             contact = c.contact(i)
        ...             print(f"Collision: {contact.collision1().name()}")
    )pbdoc")
        .def(py::init<>(), "Create an empty contacts collection.")
        .def("contact_size",
             &gz::msgs::Contacts::contact_size,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Get the number of contacts in this collection.

            Returns
            -------
            int
                Number of contact points.
          )pbdoc")
        .def("contact",
             py::overload_cast<int>(&gz::msgs::Contacts::contact, py::const_),
             py::arg("idx"), py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Get a contact by index.

            Parameters
            ----------
            idx : int
                Index of the contact (0 to contact_size()-1).

            Returns
            -------
            Contact
                The contact at the given index.
           )pbdoc");

    py::class_<ControllerParameters>(m, "ControllerParameters", R"pbdoc(
        A struct representing other parameters required by controllers.
        These include mass, inertia, gravity, etc.
        )pbdoc")
        .def(py::init<>())
        .def_readonly("inertia", &ControllerParameters::inertia, "Inertia of the UAV (kg m^2)")
        .def_readonly("mass", &ControllerParameters::mass, "Mass of the UAV (kg)")
        .def_readonly("gravity_mag", &ControllerParameters::gravity_mag, " Magnitude of gravitym/s^2")
        .def_readonly("gravity_vec", &ControllerParameters::gravity_vec, " Vector of gravity in inertial frame m/s^2")
        .def_readonly("max_accel", &ControllerParameters::max_accel, "Maximum acceleration the UAV can achieve in m/s^2")
        .def_readonly("miscellaneous", &ControllerParameters::miscellaneous);
    m.def("GAIN_MAP", []()
          { return GAIN_MAP; }, "Get the Gain MAP (tuned for specific drone model)");
    m.def("PARAMETER_MAP", []()
          { return PARAMETER_MAP; }, "Get parameter map  (tuned for specific drone model)");
    py::class_<UAVController, PyUAVController, std::shared_ptr<UAVController>>(m, "UAVController", R"pbdoc(
        Base controller class for UAV (Unmanned Aerial Vehicle) models.

        This class provides the interface for implementing controllers that
        compute control inputs for UAV models based on current and desired states.

        Methods
        -------
        calculate_force(state, state_dot, desired_state)
            Calculate the control input force vector.
        calculate_moments(state, state_dot, desired_state, force)
            Calculate the moments vector (e.g., moments in Nm).

        See Also
        --------
        InnerLoopController : Inner-loop attitude controller implementation.
        GeometricController : Geometric mechanics based PD controller implementation.

        Example
        -------
        >>> controller = gzdrl.GeometricController(...)
        >>> server.set_controller("quadrotor", "quadrotor/base_link", controller)
        >>> # Use it through control_with_rotor_velocity() or control_with_wrench().
    )pbdoc")
        .def(py::init<>(), "Create a base UAV controller.")
        .def("calculate_force",
             &UAVController::calculate_force,
             py::arg("state"),
             py::arg("state_dot"),
             py::arg("desired_state"),
             R"pbdoc(
            Calculate the control force for the UAV.

            Computes the abstract control force based on
            the current state, its derivative, and the desired state.

            Parameters
            ----------
            state : numpy.ndarray
                Current state vector (13 elements).
            state_dot : numpy.ndarray
                Time derivative of the state vector.
            desired_state : numpy.ndarray
                Target/desired state vector.

            Returns
            -------
            numpy.ndarray
                Control input force vector.
            )pbdoc")
        .def("calculate_moments",
             &UAVController::calculate_moments,
             py::arg("state"),
             py::arg("state_dot"),
             py::arg("desired_state"),
             py::arg("force"),
             R"pbdoc(
            Calculate the moments for the UAV.

            Computes the moments
            that should be applied to achieve the control of the UAV.

            Parameters
            ----------
            state : numpy.ndarray
                Current state vector (13 elements).
            state_dot : numpy.ndarray
                Time derivative of the state vector.
            desired_state : numpy.ndarray
                Target/desired state vector.
            force : numpy.ndarray
                Control input force vector.
            Returns
            -------
            numpy.ndarray
                Moments vector (e.g., 3 moments in Nm).
            )pbdoc")
        .def("calculate_thrust_moments",
             &UAVController::calculate_thrust_moments,
             py::arg("state"),
             py::arg("state_dot"),
             py::arg("desired_state"),
             R"pbdoc(
            Calculate the control force and moments for the UAV.
            Computes both the control force and moments
            that should be applied to achieve the control of the UAV.)pbdoc");
    py::class_<GeometricController, UAVController, std::shared_ptr<GeometricController>>(m, "GeometricController", R"pbdoc(
        Geometric controller class for UAV (Unmanned Aerial Vehicle) models.

        This class implements a geometric mechanics based PD controller that
        computes control inputs for UAV models based on current and desired states.

        Methods
        -------
        calculate_force(state, state_dot, desired_state)
            Calculate the control force vector.
        calculate_moments(state, state_dot, desired_state, force)
            Calculate the moments (e.g., rotor moments in Nm).
  
        See Also
        --------
        InnerLoopController : Inner-loop attitude controller implementation.
        UAVController : Base controller class.

        Example
        -------
        >>> import numpy as np
        >>> kp_position = np.array([6.0, 6.0, 10.0])
        >>> kd_position = np.array([2.0, 2.0, 3.0])
        >>> kp_attitude = np.array([8.0, 8.0, 10.0])
        >>> kd_attitude = np.array([2.0, 2.0, 3.0])
        >>> gravity_vec = np.array([0., 0., -9.81])
        >>> max_lin_accel = np.array([5.0, 5.0, 10.0])
        >>> mass = 1.5
        >>> inertia = np.eye(3) * 0.01
        >>> controller = gzdrl.GeometricController( 
        ...     kp_position, kd_position, 
        ...     kp_attitude, kd_attitude, 
        ...     max_lin_accel, gravity_vec,
        ...     mass, inertia 
        ... )
        >>> server.set_controller("quadrotor", "quadrotor/base_link", controller)
        >>> # Use it through control_with_rotor_velocity() or control_with_wrench().
    )pbdoc")
        .def(py::init<Eigen::Vector3d, Eigen::Vector3d,
                      Eigen::Vector3d, Eigen::Vector3d,
                      Eigen::Vector3d, Eigen::Vector3d, double, Eigen::Matrix3d>(),
             py::arg("kp_position"),
             py::arg("kd_position"),
             py::arg("kp_attitude"),
             py::arg("kd_attitude"),
             py::arg("max_linear_accel"),
             py::arg("gravity"),
             py::arg("mass"),
             py::arg("inertia"),
             R"pbdoc(
            Create a geometric controller with specified gains and parameters.
            Parameters
            ----------
            kp_position : numpy.ndarray
                Proportional gains for position error (3 elements for x, y, z).
            kd_position : numpy.ndarray
                Derivative gains for position error.
            kp_attitude : numpy.ndarray
                Proportional gains for attitude error (3 elements for roll, pitch, yaw).
            kd_attitude : numpy.ndarray
                Derivative gains for attitude error.
            max_linear_accel : numpy.ndarray
                Maximum allowable linear acceleration (3 elements).
            gravity : numpy.ndarray
                Gravity vector [gx, gy, gz] in m/s².
            mass : float
                Total mass of the UAV in kg.
            inertia : numpy.ndarray
                3x3 inertia tensor in kg·m².
            )pbdoc")
        .def("calculate_moments",
             &GeometricController::calculate_moments,
             py::arg("state"),
             py::arg("state_dot"),
             py::arg("desired_state"),
             py::arg("force"),
             R"pbdoc(
            Calculate the control moments for the UAV.

            Parameters
            ----------
            state : numpy.ndarray
                Current state vector (13 elements).
            state_dot : numpy.ndarray
                Time derivative of the state vector.
            desired_state : numpy.ndarray
                Target/desired state vector.
            force : numpy.ndarray
                Control input force vector.
            Returns
            -------
            numpy.ndarray
                Control moments vector.
            )pbdoc")
        .def("calculate_force",
             &GeometricController::calculate_force,
             py::arg("state"),
             py::arg("state_dot"),
             py::arg("desired_state"),
             R"pbdoc(
            Calculate the control force vector for UAV.

            Parameters
            ----------
            state : numpy.ndarray
                Current state vector (13 elements).
            state_dot : numpy.ndarray
                Time derivative of the state vector.
            desired_state : numpy.ndarray
                Target/desired state vector.

            Returns
            -------
            numpy.ndarray
                3D force vector.
            )pbdoc");

    py::class_<InnerLoopController, UAVController, std::shared_ptr<InnerLoopController>>(m, "InnerLoopController", R"pbdoc(
        Inner-loop attitude controller for UAV stabilization.

        This controller handles the inner-loop control of a UAV, managing
        attitude stabilization based on commands from an outer-loop position
        controller. It uses PD control on both angle and angular rate.

        Parameters
        ----------
        kp_angle : numpy.ndarray
            Proportional gains for angle error (3 elements for roll, pitch, yaw).
        kd_angle : numpy.ndarray
            Derivative gains for angle error.
        kp_angular_rate : numpy.ndarray
            Proportional gains for angular rate error.
        kd_angular_rate : numpy.ndarray
            Derivative gains for angular rate error.
        mass : float
            Total mass of the UAV in kg.
        inertia : numpy.ndarray
            3x3 inertia tensor in kg·m².
        gravity : numpy.ndarray
            Gravity vector [gx, gy, gz] in m/s².

        Methods
        -------
        calculate_force(state, state_dot, desired_state)
            Calculate control force from the current and desired states.
        calculate_moments(state, state_dot, desired_state, force)
            Calculate moments from the current and desired states and the control force.

        See Also
        --------
        UAVController : Base controller class.

        Example
        -------
        >>> import numpy as np
        >>> kp_angle = np.array([10.0, 10.0, 5.0])
        >>> kd_angle = np.array([1.0, 1.0, 0.5])
        >>> kp_rate = np.array([5.0, 5.0, 2.0])
        >>> kd_rate = np.array([0.5, 0.5, 0.2])
        >>> mass = 1.5
        >>> inertia = np.eye(3) * 0.01
        >>> gravity = np.array([0., 0., -9.81])
        >>> controller = gzdrl.InnerLoopController(
        ...     kp_angle, kd_angle, kp_rate, kd_rate, mass, inertia, gravity
        ... )
    )pbdoc")
        .def(py::init<const Gain &,
                      const Gain &,
                      const Gain &,
                      const Gain &,
                      const double,
                      const Eigen::Matrix3d &,
                      const Eigen::Vector3d &>(),
             py::arg("kp_angle"),
             py::arg("kd_angle"),
             py::arg("kp_angular_rate"),
             py::arg("kd_angular_rate"),
             py::arg("mass"),
             py::arg("inertia"),
             py::arg("gravity"),
             R"pbdoc(
            Create an inner-loop controller with specified gains.

            Parameters
            ----------
            kp_angle : numpy.ndarray
                Proportional gains for angle error.
            kd_angle : numpy.ndarray
                Derivative gains for angle error.
            kp_angular_rate : numpy.ndarray
                Proportional gains for angular rate error.
            kd_angular_rate : numpy.ndarray
                Derivative gains for angular rate error.
            mass : float
                Total mass of the UAV in kg.
            inertia : numpy.ndarray
                3x3 inertia tensor in kg·m².
            gravity : numpy.ndarray
                Gravity vector [gx, gy, gz] in m/s².
            )pbdoc")
        .def("calculate_force",
             &InnerLoopController::calculate_force,
             py::arg("state"),
             py::arg("state_dot"),
             py::arg("desired_state"),
             R"pbdoc(
            Calculate force commands from outer-loop command.

            Parameters
            ----------
            state : numpy.ndarray
                Current state vector (13 elements).
            state_dot : numpy.ndarray
                Time derivative of the state vector.
            desired_state : numpy.ndarray
                Target/desired state vector.

            Returns
            -------
            numpy.ndarray
                Force commands .
            )pbdoc")
        .def("calculate_moments",
             &InnerLoopController::calculate_moments,
             py::arg("state"),
             py::arg("state_dot"),
             py::arg("desired_state"),
             py::arg("force"),
             R"pbdoc(
            Get the moments computed.

            Parameters
            ----------
            state : numpy.ndarray
                Current state vector (13 elements).
            state_dot : numpy.ndarray
                Time derivative of the state vector.
            desired_state : numpy.ndarray
                Target/desired state vector.
            force : numpy.ndarray
                Control input force vector.

            Returns
            -------
            numpy.ndarray
                Moments vector (torques) in Nm.
           )pbdoc");

    py::class_<SlidingModeController, UAVController, std::shared_ptr<SlidingModeController>>(m, "SlidingModeController", R"pbdoc(
        Sliding Mode Controller for UAV models.

        This class implements a sliding mode controller that computes control
        inputs for UAV models based on current and desired states.

        Methods
        -------
        calculate_force(state, state_dot, desired_state)
            Calculate the control force vector.
        calculate_moments(state, state_dot, desired_state, force)
            Calculate the moments (e.g., rotor moments in Nm).
            
        Example
        -------
        >>> gains = gzdrl.GAIN_MAP()["qdrone2"]["sliding_mode_controller"]
        >>> params = gzdrl.PARAMETER_MAP()["qdrone2"]["sliding_mode_controller"]
        >>> controller = gzdrl.SlidingModeController(
        ...     gains["lambda_pos"], gains["kappa_pos"],
        ...     gains["lambda_att"], gains["kappa_att"],
        ...     gains["boundary_pos"], gains["boundary_att"],
        ...     params.max_accel, params.gravity_vec,
        ...     params.mass, params.inertia,
        ... )
        >>> server.set_controller("quadrotor", "quadrotor/base_link", controller)
    )pbdoc")
        .def(py::init<Eigen::Vector3d,
                      Eigen::Vector3d, // Switching gain position
                      Eigen::Vector3d,
                      Eigen::Vector3d, // Switching gain attitude
                      Eigen::Vector3d, // Boundary layer (phi)
                      Eigen::Vector3d,
                      Eigen::Vector3d,
                      Eigen::Vector3d,
                      double,
                      Eigen::Matrix3d>(),
             py::arg("lambda_pos"),
             py::arg("kappa_pos"),
             py::arg("lambda_att"),
             py::arg("kappa_att"),
             py::arg("boundary_pos"),
             py::arg("boundary_att"),
             py::arg("max_lin_acc"),
             py::arg("gravity"),
             py::arg("mass"),
             py::arg("inertia"),
             R"pbdoc(
                         Create a sliding mode controller.
                         Parameters
                         ----------
                         lambda_pos : numpy.ndarray
                             Convergence rate for position error (3 elements).
                         kappa_pos : numpy.ndarray
                             Switching gain for position error (3 elements).
                         lambda_att : numpy.ndarray
                             Convergence rate for attitude error (3 elements).
                         kappa_att : numpy.ndarray
                             Switching gain for attitude error (3 elements).
                         boundary_pos : numpy.ndarray
                              Boundary layer thickness for position (3 elements).
                          boundary_att : numpy.ndarray
                              Boundary layer thickness for attitude (3 elements).
                         max_lin_acc : numpy.ndarray
                             Maximum allowable linear acceleration (3 elements).
                         gravity : numpy.ndarray
                             Gravity vector [gx, gy, gz] in m/s².
                          mass : float  
                              Total mass of the UAV in kg.
                         inertia : numpy.ndarray
                             Inertia matrix of the UAV (3x3).
                         )pbdoc")
        .def("calculate_force",
             &SlidingModeController::calculate_force,
             py::arg("state"),
             py::arg("state_dot"),
             py::arg("desired_state"),
             R"pbdoc(
            Calculate the control force for the UAV.
            Computes the abstract control force based on
            the current state, its derivative, and the desired state.
            Parameters
            ----------
            state : numpy.ndarray
                Current state vector (13 elements).
            state_dot : numpy.ndarray
                Time derivative of the state vector.
            desired_state : numpy.ndarray
                Target/desired state vector.
            Returns
            -------
            numpy.ndarray
                Control input force vector.
            )pbdoc")
        .def("calculate_moments",
             &SlidingModeController::calculate_moments,
             py::arg("state"),
             py::arg("state_dot"),
             py::arg("desired_state"),
             py::arg("force"),
             R"pbdoc(
            Calculate the moments for the UAV.
            Computes the moments
            that should be applied to achieve the control of the UAV.
            Parameters
            ----------
            state : numpy.ndarray
                Current state vector (13 elements).
            state_dot : numpy.ndarray
                Time derivative of the state vector.
            desired_state : numpy.ndarray
                Target/desired state vector.
            force : numpy.ndarray
                Control input force vector.
            Returns
            -------
            numpy.ndarray
                moments vector (e.g., 3 moments in Nm).
            )pbdoc");
    py::class_<PIDController, UAVController, std::shared_ptr<PIDController>>(m, "PIDController", R"pbdoc(
        PIDController Controller for UAV models.

        This class implements a PID controller that computes control
        inputs for UAV models based on current and desired states. The controller
        is a mimic of the actual controller in PID for hardware.

        Methods
        -------
        calculate_force(state, state_dot, desired_state)
            Calculate the control force vector.
        calculate_moments(state, state_dot, desired_state, force)
            Calculate the moments (e.g., rotor moments in Nm).
            
        Example
        -------
        >>> gains = gzdrl.GAIN_MAP()["qdrone2"]["pid_controller"]
        >>> params = gzdrl.PARAMETER_MAP()["qdrone2"]["pid_controller"]
        >>> misc = params.miscellaneous
        >>> controller = gzdrl.PIDController(
        ...     gains["kp"], gains["kd"], gains["ki"],
        ...     gains["kp_att"], gains["kd_att"],
        ...     gains["kp_omega"], gains["kd_omega"],
        ...     float(misc["omega_n"][0]), float(misc["zeta"][0]),
        ...     float(misc["dt"][0]), gains["integral_sat_limit"],
        ...     gains["stabilization_command_sat_limit"],
        ...     gains["angle_cmd_limit"], gains["omega_cmd_limit"],
        ...     gains["max_torque"], params.mass, params.gravity_mag,
        ... )
        >>> server.set_controller("quadrotor", "quadrotor/base_link", controller)
    )pbdoc")
        .def(py::init<const Eigen::Vector4d &, // Position Gain
                      const Eigen::Vector4d &, // Derivative Gain
                      const Eigen::Vector4d &, // Integral Gain
                      const Eigen::Vector3d &, // Attitude P gain
                      const Eigen::Vector3d &, // Attitude D gain
                      const Eigen::Vector3d &, // Angular rate P gain
                      const Eigen::Vector3d &, // Angular rate D gain
                      const double &,          // w_n for second order low pass filter (for outer loop commands)
                      const double &,          // zeta for second order low pass filyter (for outer loop commands)
                      const double &,          // sampling time (for integral and filter)
                      const Eigen::Vector4d &, // Saturation for integral term [post gain multiplication]
                      const Eigen::Vector4d &, // Saturation limit for stabilization command
                      const Eigen::Vector3d &, // maximum angle commands from outer-loop
                      const Eigen::Vector3d &, // maximum omega cmd from pd controller for attitude
                      const Eigen::Vector3d &,
                      const double &,
                      const double &>(),
             py::arg("kp"),
             py::arg("kd"),
             py::arg("ki"),
             py::arg("kp_att"),
             py::arg("kd_att"),
             py::arg("kp_omega"),
             py::arg("kd_omega"),
             py::arg("omega_n"),
             py::arg("zeta"),
             py::arg("dt"),
             py::arg("integral_sat_limit"),
             py::arg("stabilization_command_sat_limit"),
             py::arg("angle_cmd_limit"),
             py::arg("omega_cmd_limit"),
             py::arg("max_torque"),
             py::arg("mass"),
             py::arg("gravity_mag"),
             R"pbdoc(
                         Create a PIDController controller.
                         Parameters
                         ----------
                         kp : np.ndarray
                            Position gain (4 elements for [x y z yaw])
                         kd : np.ndarray
                            Derivative gain (4 elements for [x y z yaw])
                         ki : np.ndarray
                            Integral gain (4 elements for [x y z yaw])
                         kp_att : np.ndarray
                            Position gain for angle commands (3 elements for [roll pitch yaw])
                         kd_att : np.ndarray
                            Derivative gain for angle commands(3 elements for [roll pitch yaw])
                         kp_omega : np.ndarray
                            Position gain for omega commands (3 elements for [wx wy wz])
                         kd_omega : np.ndarray
                            Derivative gain for omega commands(3 elements for [wx wy wz])
                         omega_n : float
                            Second-order filter natural frequency (cutoff frequency) in rad/s
                         zeta : float
                            Second-order filter damping ratio
                         dt : float
                            Sampling time in seconds (used for the integral term and filter)
                         integral_sat_limit : np.ndarray
                            Saturation for post-gain integral term (4 elements for [x y z yaw])
                         stabilization_command_sat_limit : np.ndarray
                            Saturation limit for final stabilization command (4 elements for  [T roll pitch yaw])
                         angle_cmd_limit : np.ndarray
                            Saturation for angle commands (3 elements for [roll pitch yaw])
                         omega_cmd_limit : np.ndarray
                            Saturation limit for omega command (outer loop in stabilizer) (3 elements for [wx wy wz])
                         max_torque : np.ndarray
                            Maximum torque that can be generated (3 elements for [roll pitch yaw])
                         mass : float
                            Total mass of the UAV in kg
                         gravity_mag : float
                            Magnitude of gravitational acceleration in m/s^2
                         )pbdoc")
        .def("calculate_force",
             &PIDController::calculate_force,
             py::arg("state"),
             py::arg("state_dot"),
             py::arg("desired_state"),
             R"pbdoc(
            Calculate the control force for the UAV.
            Computes the abstract control force based on
            the current state, its derivative, and the desired state.
            Parameters
            ----------
            state : numpy.ndarray
                Current state vector (13 elements).
            state_dot : numpy.ndarray
                Time derivative of the state vector.
            desired_state : numpy.ndarray
                Target/desired state vector.
            Returns
            -------
            numpy.ndarray
                Control input force vector.
            )pbdoc")
        .def("calculate_moments",
             &PIDController::calculate_moments,
             py::arg("state"),
             py::arg("state_dot"),
             py::arg("desired_state"),
             py::arg("force"),
             R"pbdoc(
            Calculate the moments for the UAV.
            Computes the moments
            that should be applied to achieve the control of the UAV.
            Parameters
            ----------
            state : numpy.ndarray
                Current state vector (13 elements).
            state_dot : numpy.ndarray
                Time derivative of the state vector.
            desired_state : numpy.ndarray
                Target/desired state vector.
            force : numpy.ndarray
                Control input force vector.
            Returns
            -------
            numpy.ndarray
                moments vector (e.g., 3 moments in Nm).
            )pbdoc");
    py::class_<DRLServer, std::shared_ptr<DRLServer>>(m, "DRLServer", R"pbdoc(
        Main interface for controlling Gazebo simulations in RL environments.

        DRLServer provides a high-level API for deep reinforcement learning
        with the Gazebo physics simulator. It supports multiple models,
        various control modalities, and optional sensor integration.

        Parameters
        ----------
        partition : str
            Namespace for transport isolation (allows multiple servers).
        sdf_file : str
            Path to the SDF world file.
        model_names : list[str]
            List of model names to track and control.
        enable_sensors : bool
            Whether to enable sensor interfaces (cameras, LiDAR).
        config : DRLServerConfig, optional
            Server configuration for visualization and markers.

        Attributes
        ----------
        model_names : list[str]
            Names of models being tracked.
        control_states : dict
            Map of model names to their control state tuples.

        Notes
        -----
        - Commands are applied immediately (no queuing)
        - Serialize state queries with physics stepping when sharing a server
          between threads

        Example
        -------
        >>> import gzdrl
        >>> import numpy as np
        >>> 
        >>> # Create a server for the packaged hover world
        >>> server = gzdrl.DRLServer(
        ...     "training",
        ...     str(gzdrl.get_sdf_path("world_hover.sdf")),
        ...     ["quadrotor"],
        ...     False  # no sensors
        ... )
        >>> 
        >>> # Training loop
        >>> for episode in range(100):
        ...     # Reset position
        ...     server.reset_pos("quadrotor", np.array([0., 0., 1.]), np.zeros(3))
        ...     
        ...     for step in range(500):
        ...         # Get state
        ...         states = server.state_info("quadrotor")
        ...         obs = states["quadrotor/base_link"]
        ...         
        ...         # Compute and apply action
        ...         action = policy(obs)
        ...         server.set_rotor_velocity_cmd(
        ...             "quadrotor", "quadrotor/base_link", action
        ...         )
        ...         
        ...         # Step simulation
        ...         server.run_once()

        See Also
        --------
        AsyncDRLServerPool : For parallel environment training.
        DRLServerConfig : Configuration options.
    )pbdoc")
        .def(py::init<const std::string &, const std::string &,
                      const std::vector<std::string> &, bool, DRLServerConfig &>(),
             py::arg("partition"),
             py::arg("sdf_file"),
             py::arg("model_names"),
             py::arg("enable_sensors"),
             py::arg("config"),
             R"pbdoc(
            Create a DRLServer with custom configuration.

            Parameters
            ----------
            partition : str
                Namespace for transport isolation.
            sdf_file : str
                Path to the SDF world file.
            model_names : list[str]
                List of model names to track.
            enable_sensors : bool
                Whether to enable sensor interfaces.
            config : DRLServerConfig
                Custom server configuration.
            )pbdoc")
        .def(py::init<const std::string &,
                      const std::string &, const std::vector<std::string> &, bool>(),
             py::arg("partition"),
             py::arg("sdf_file"),
             py::arg("model_names"),
             py::arg("enable_sensors"),
             R"pbdoc(
            Create a DRLServer with default configuration.

            Parameters
            ----------
            partition : str
                Namespace for transport isolation.
            sdf_file : str
                Path to the SDF world file.
            model_names : list[str]
                List of model names to track.
            enable_sensors : bool
                Whether to enable sensor interfaces.
            )pbdoc")
        .def("run_once",
             &DRLServer::run_once,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Execute one physics simulation step.

            Advances the simulation by one timestep (typically 1ms).
            Commands set before this call take effect during the step.

            Warning
            -------
            This method is NOT thread-safe. Only call from one thread.

            See Also
            --------
            run_N : Execute multiple steps efficiently.
            step_size : Get the timestep duration.
          )pbdoc")
        .def("run_N",
             &DRLServer::run_N,
             py::arg("N"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Execute N physics simulation steps.

            Parameters
            ----------
            N : int
                Number of simulation steps to execute.

            Notes
            -----
            More efficient than calling run_once() N times.
            Total simulated time = N * step_size() seconds.

            Warning
            -------
            This method is NOT thread-safe. Only call from one thread.
          )pbdoc")
        .def("step_size",
             &DRLServer::step_size,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Get the physics simulation timestep duration.

            Returns
            -------
            float
                Time step duration in seconds (e.g., 0.001 for 1ms steps).
          )pbdoc")
        .def("state_info",
             &DRLServer::state_info,
             py::arg("model_name"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Get current state of all links in a model.

            Parameters
            ----------
            model_name : str
                Name of the model to query.

            Returns
            -------
            dict[str, numpy.ndarray]
                Map of link names to 19-element state vectors.
                State format: [pos(3), quat(4), vel(3), ang_vel(3), acc(3), ang_acc(3)]

            Example
            -------
            >>> states = server.state_info("quadrotor")
            >>> position = states["quadrotor/base_link"][:3]
            >>> velocity = states["quadrotor/base_link"][7:10]
            )pbdoc")
        .def("reset_pos",
             static_cast<void (DRLServer::*)(std::string, Eigen::Vector3d &&, Eigen::Vector3d &&)>(&DRLServer::reset_pos),
             py::arg("model_name"),
             py::arg("position"),
             py::arg("orientation"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Reset model to a new position and orientation.

            Triggers entity teleportation at the new pose.
            Runs three stabilization physics steps before returning.

            Parameters
            ----------
            model_name : str
                Name of the model to reset.
            position : numpy.ndarray
                New position [x, y, z] in meters.
            orientation : numpy.ndarray
                New orientation [roll, pitch, yaw] in radians.

            Example
            -------
            >>> server.reset_pos("quadrotor", np.array([0., 0., 2.]), np.zeros(3))
            >>> state = server.state_info("quadrotor")
            )pbdoc")
        .def("respawn_model",
             static_cast<void (DRLServer::*)(std::string, Eigen::Vector3d &&, Eigen::Vector3d &&)>(&DRLServer::respawn_model),
             py::arg("model_name"),
             py::arg("position"),
             py::arg("orientation"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Respawn model to a new position and orientation.

            Triggers entity removal and respawn at the new pose.
            Runs three stabilization physics steps before returning.

            Parameters
            ----------
            model_name : str
                Name of the model to reset.
            position : numpy.ndarray
                New position [x, y, z] in meters.
            orientation : numpy.ndarray
                New orientation [roll, pitch, yaw] in radians.

            Warning
            -------
            This is an expensive operation. Avoid calling every step.

            Example
            -------
            >>> server.respawn_model("quadrotor", np.array([0., 0., 2.]), np.zeros(3))
            >>> state = server.state_info("quadrotor")
            )pbdoc")
        .def("set_rotor_velocity_cmd",
             static_cast<void (DRLServer::*)(std::string, std::string, Eigen::VectorXd &&)>(&DRLServer::set_rotor_velocity_cmd),
             py::arg("model_name"),
             py::arg("link_name"),
             py::arg("cmd"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Set rotor velocities for a specific link.

            Parameters
            ----------
            model_name : str
                Name of the model.
            link_name : str
                Name of the base link.
            cmd : numpy.ndarray
                Rotor velocities in rad/s (one per rotor).

            Notes
            -----
            Command is applied immediately. Takes effect in next run_once().
            )pbdoc")
        .def("set_ctbr_cmd",
             static_cast<void (DRLServer::*)(std::string, std::string, Eigen::VectorXd &&)>(&DRLServer::set_ctbr_cmd),
             py::arg("model_name"),
             py::arg("link_name"),
             py::arg("cmd"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Set ctbr for a specific link.

            Parameters
            ----------
            model_name : str
                Name of the model.
            link_name : str
                Name of the link to apply thrusts and body rates.
            cmd : numpy.ndarray
                ctbr cmd [T, wx, wy, wz].

            Notes
            -----
            Command is applied immediately. Takes effect in next run_once().
            )pbdoc")
        .def("set_ctbr_cmd",
             static_cast<void (DRLServer::*)(std::string, std::string, Eigen::VectorXd &&, const Eigen::VectorXd &, const Eigen::VectorXd &)>(&DRLServer::set_ctbr_cmd),
             py::arg("model_name"),
             py::arg("link_name"),
             py::arg("cmd"),
             py::arg("kp"),
             py::arg("kd"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Set ctbr for a specific link.

            Parameters
            ----------
            model_name : str
                Name of the model.
            link_name : str
                Name of the base link.
            cmd : numpy.ndarray
                ctbr cmd [T, wx, wy, wz].
            kp : numpy.ndarray
                Proportional Gains for internal stabilizer, such that tau = kp*e_W + kd*e_A + cross(W, J*W)
            kd : numpy.ndarray
                Derivative Gains for internal stabilizer, such that tau = kp*e_W + kd*e_A + cross(W, J*W)
            Notes
            -----
            Command is applied immediately. Takes effect in next run_once().
            )pbdoc")
        .def("set_ctbt_cmd", static_cast<void (DRLServer::*)(std::string, std::string, Eigen::VectorXd &&)>(&DRLServer::set_ctbt_cmd),
             py::arg("model_name"),
             py::arg("link_name"),
             py::arg("cmd"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Set CTBT command for a specific link.

            Parameters
            ----------
            model_name : str
                Name of the model.
            link_name : str
                Name of the link to control.
            cmd : numpy.ndarray
                CTBT (F, taux, tauy, tauz)
          )pbdoc")
        .def("set_srt_cmd", static_cast<void (DRLServer::*)(std::string, std::string, const std::vector<std::string> &, const std::vector<int> &, Eigen::VectorXd &&, double)>(&DRLServer::set_srt_cmd),
             py::arg("model_name"),
             py::arg("link_name"),
             py::arg("rotor_links"),
             py::arg("tuning_directions"),
             py::arg("cmd"),
             py::arg("ktau"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Set SRT command for a specific link.

            Parameters
            ----------
            model_name : str
                Name of the model.
            link_name : str
                Name of the  canonical link.
            rotor_links : list[str] 
                Rotor link names (srt will be applied at geometric center)
            turning_directions : list[int]
                Turning directions of rotors   
            cmd : numpy.ndarray
                thrust for each rotor
            ktau : float
                Motor torque constant such that |Tau| = |Ktau*SRT|
          )pbdoc")
        .def("set_velocity_cmd", static_cast<void (DRLServer::*)(std::string, std::string, Eigen::Vector3d &&)>(&DRLServer::set_velocity_cmd),
             py::arg("model_name"),
             py::arg("link_name"),
             py::arg("cmd"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Set linear velocity command for a specific link.
            The command is expected to be in link-frame (a.k.a body-fixed frame)
            
            Parameters
            ----------
            model_name : str
                Name of the model.
            link_name : str
                Name of the link to control.
            cmd : numpy.ndarray
                Linear velocity [vx, vy, vz] in m/s (world frame).
          )pbdoc")

        .def("set_angular_velocity_cmd", static_cast<void (DRLServer::*)(std::string, std::string, Eigen::Vector3d &&)>(&DRLServer::set_angular_velocity_cmd),
             py::arg("model_name"),
             py::arg("link_name"),
             py::arg("cmd"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Set angular velocity command for a specific link.

            Parameters
            ----------
            model_name : str
                Name of the model.
            link_name : str
                Name of the link to control.
            cmd : numpy.ndarray
                Angular velocity [wx, wy, wz] in rad/s (body-fixed frame).

            Notes
            -----
            Kinematic control - the link immediately achieves the commanded
            angular velocity.
          )pbdoc")
        .def("set_ackermann_velocity_cmd", static_cast<void (DRLServer::*)(std::string, std::string, Eigen::Vector2d &&)>(&DRLServer::set_ackermann_velocity_cmd),
             py::arg("model_name"),
             py::arg("link_name"),
             py::arg("cmd"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Set Ackermann steering velocity command for a specific link.

            Parameters
            ----------
            model_name : str
                Name of the model.
            link_name : str
                Name of the link to control.
            cmd : numpy.ndarray
                [linear_velocity (m/s), angular velocity (rad/s)].
          )pbdoc")
        .def("set_joint_position_cmd", static_cast<void (DRLServer::*)(std::string, std::string, Eigen::Vector3d &)>(&DRLServer::set_joint_position_cmd),
             py::arg("model_name"),
             py::arg("joint_name"),
             py::arg("cmd"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Set joint position command.

            Parameters
            ----------
            model_name : str
                Name of the model.
            joint_name : str
                Name of the joint to control.
            cmd : numpy.ndarray
                Joint position target.
          )pbdoc")
        .def("set_joint_position_cmd", static_cast<void (DRLServer::*)(std::string, std::string, Eigen::Vector3d &&)>(&DRLServer::set_joint_position_cmd),
             py::arg("model_name"),
             py::arg("joint_name"),
             py::arg("cmd"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Set joint position command.

            Parameters
            ----------
            model_name : str
                Name of the model.
            joint_name : str
                Name of the joint to control.
            cmd : numpy.ndarray
                Joint position target (radians for revolute, meters for prismatic).
          )pbdoc")
        .def("get_contacts", &DRLServer::get_contacts,
             py::arg("model_name"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Get contact data for all links in a model.

            Parameters
            ----------
            model_name : str
                Name of the model to query.

            Returns
            -------
            dict[str, list[Contacts]]
                Map of link names to lists of contact batches.

            Notes
            -----
            Must call request_contact_data() first to enable contact sensing.

            Example
            -------
            >>> server.request_contact_data("quadrotor")
            >>> server.run_once()
            >>> contacts = server.get_contacts("quadrotor")
            >>> if contacts["quadrotor/base_link"]:
            ...     print("Contact detected!")
          )pbdoc")
        .def("set_wrench", static_cast<void (DRLServer::*)(std::string, std::string, Eigen::Vector3d &&, Eigen::Vector3d &&)>(&DRLServer::set_wrench),
             py::arg("model_name"),
             py::arg("link_name"),
             py::arg("force"),
             py::arg("moments"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Apply wrench (force + torque) to a link.

            Parameters
            ----------
            model_name : str
                Name of the model.
            link_name : str
                Name of the link to apply wrench to.
            force : numpy.ndarray
                Force vector [fx, fy, fz] in Newtons (world frame).
            moments : numpy.ndarray
                Torque vector [tx, ty, tz] in N·m (world frame).

            Example
            -------
            >>> # Apply 10N upward force
            >>> server.set_wrench(
            ...     "quadrotor", "quadrotor/base_link",
            ...     np.array([0., 0., 10.]), np.zeros(3),
            ... )
          )pbdoc")
        .def("set_controller", &DRLServer::set_controller,
             py::arg("model_name"),
             py::arg("link_name"),
             py::arg("controller"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Attach a controller to a specific link.

            Parameters
            ----------
            model_name : str
                Name of the model.
            link_name : str
                Name of the link to control.
            controller : UAVController
                Controller instance to use.

            See Also
            --------
            control_with_rotor_velocity : Use the controller.
          )pbdoc")
        .def_property_readonly("control_states", py::cpp_function([](DRLServer &self) -> const auto &
                                                                  { return self.control_states; }, py::call_guard<py::gil_scoped_acquire>()),
                               R"pbdoc(
            dict: Map of model names to their link control states.

            Each entry maps link names to (current_state, state_derivative) tuples.
          )pbdoc")
        .def("control_with_rotor_velocity", static_cast<void (DRLServer::*)(std::string, std::string, const Stated &&, int)>(&DRLServer::control_with_rotor_velocity), py::arg("model_name"), py::arg("link_name"), py::arg("desired_state"), py::arg("N"), py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Update link using controller and rotor velocity commands.

            The controller computes rotor velocity commands to reach the desired state,
            then runs N simulation steps while tracking progress.

            Parameters
            ----------
            model_name : str
                Name of the model.
            link_name : str
                Name of the link to control.
            desired_state : numpy.ndarray
                Target state vector.
            N : int
                Number of simulation steps to run.

            Notes
            -----
            Requires a controller to be set for the (model_name, link_name)
            pair via set_controller(), and the model must have the
            MultiRotorPlugin attached.

            See Also
            --------
            set_controller : Attach a controller to a link.
            control_with_wrench : Controller drives the link via direct wrench.
          )pbdoc")
        .def("control_with_wrench", static_cast<void (DRLServer::*)(std::string, std::string, const Stated &&, int)>(&DRLServer::control_with_wrench), py::arg("model_name"), py::arg("link_name"), py::arg("desired_state"), py::arg("N"), py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Update link using controller and wrench commands.

            The controller computes forces/torques instead of rotor velocity commands.

            Parameters
            ----------
            model_name : str
                Name of the model.
            link_name : str
                Name of the link to control.
            desired_state : numpy.ndarray
                Target state vector.
            N : int
                Number of simulation steps to run.

            Notes
            -----
            Requires a controller to be set for the (model_name, link_name)
            pair via set_controller(). Unlike control_with_rotor_velocity,
            the computed wrench is applied directly and does not require a
            MultiRotorPlugin.

            See Also
            --------
            set_controller : Attach a controller to a link.
            control_with_rotor_velocity : Controller drives the rotors.
          )pbdoc")
        .def("update_control_states", &DRLServer::update_control_states, py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Update all control state tuples with current simulation state.

            Refreshes current_state in all control_states entries.
            Call this before using controllers to get latest state.
          )pbdoc")
        .def_property_readonly("model_names", py::cpp_function([](DRLServer &self)
                                                               { return self.model_names; }, py::call_guard<py::gil_scoped_acquire>()),
                               "list[str]: Names of models being tracked.")
        .def("set_trajectory_trace", &DRLServer::set_trajectory_trace, py::arg("model_name"), py::arg("link_name"), py::arg("config"), py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Enable trajectory visualization markers for a link.

            Parameters
            ----------
            model_name : str
                Name of the model.
            link_name : str
                Name of the link to trace.
            config : DRLServerConfig or None
                Marker configuration (None for default).
          )pbdoc")
        .def("get_sensor_image", [](DRLServer &self, std::string name)
             {
            gz::msgs::Image message;
            {
              py::gil_scoped_release release;
              message = self.get_sensor_image(name);
            }
            return convert_image_msg_to_numpy_copy(message); }, py::arg("name"),
             R"pbdoc(
            Get image from a camera sensor.

            Parameters
            ----------
            name : str
                Sensor name as defined in SDF.

            Returns
            -------
            numpy.ndarray
                Image array with shape (height, width, channels).
            )pbdoc")
        .def("get_sensor_gpu_lidar", [](DRLServer &self, std::string name)
             {
            systems::custom_plugins::Sensors::LidarFrameView data_tup;
            {
              py::gil_scoped_release release;
              data_tup = self.get_sensor_gpu_lidar(name);
            }
            if (data_tup.data){
              return py::array_t<float>({data_tup.h, data_tup.w, data_tup.c}, data_tup.data);
            }
            return py::array_t<float>({0,0,0}); }, py::arg("name"),
             R"pbdoc(
            Get GPU LiDAR scan data.

            Parameters
            ----------
            name : str
                Sensor name as defined in SDF.

            Returns
            -------
            numpy.ndarray
                LiDAR data array with shape (height, width, channels).

            Warning
            -------
            Returned data pointer is valid only until next sensor update.
            )pbdoc")
        .def("print_sensor_names", &DRLServer::print_sensor_names, py::call_guard<py::gil_scoped_release>(), "Print names of all available sensors to console.")
        .def("camera_sensor_names", &DRLServer::camera_sensor_names, py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Get list of all camera sensor names.

            Returns
            -------
            list[str]
                Names of camera sensors found in the world.
          )pbdoc")
        .def("lidar_sensor_names", &DRLServer::lidar_sensor_names, py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Get list of all LiDAR sensor names.

            Returns
            -------
            list[str]
                Names of LiDAR sensors found in the world.
          )pbdoc")
        .def("bind_img_cb", [](DRLServer &self, py::function cb, const std::string &name)
             {
            // Keep Python callback alive by storing in shared_ptr
            auto cb_ptr = keep_python_callback(std::move(cb));
            py::gil_scoped_release release;
            self.bind_img_cb([cb_ptr](const gz::msgs::Image& img_msg){
                if (!Py_IsInitialized()) return;
                py::gil_scoped_acquire gil;
                auto img = convert_image_msg_to_numpy_copy(img_msg);
                (*cb_ptr)(img);
            }, name); }, py::arg("cb"), py::arg("name"),
             R"pbdoc(
            Bind a callback function for camera images.

            Parameters
            ----------
            cb : callable
                Callback function that takes a numpy array image.
            name : str
                Sensor name.
          )pbdoc")
        .def("bind_lidar_cb", [](DRLServer &self, py::function cb, const std::string &name)
             {
            // Keep Python callback alive by storing in shared_ptr
            auto cb_ptr = keep_python_callback(std::move(cb));
            py::gil_scoped_release release;
            self.bind_lidar_cb( [cb_ptr](const systems::custom_plugins::Sensors::LidarFrameView& lidar_data){
                if (!Py_IsInitialized()) return;
                py::gil_scoped_acquire gil;
                if (lidar_data.data){
                  auto array = py::array_t<float>({lidar_data.h, lidar_data.w, lidar_data.c}, lidar_data.data);
                  (*cb_ptr)(array);
                } else {
                  (*cb_ptr)(py::none());
                }
            }, name); }, py::arg("cb"), py::arg("name"),
             R"pbdoc(
            Bind a callback function for LiDAR data.

            Parameters
            ----------
            cb : callable
                Callback function that takes a numpy array or None.
            name : str
                Sensor name.
          )pbdoc")
        .def("reset_world", &DRLServer::reset_world, py::arg("model_poses"), py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Reset all models in the world to new poses.

            Parameters
            ----------
            model_poses : dict[str, tuple[numpy.ndarray, numpy.ndarray]]
                Map of model names to (position, orientation) tuples.
          )pbdoc")
        .def("set_mass", &DRLServer::set_mass, py::arg("model_name"), py::arg("link_name"), py::arg("mass"), py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Set link mass (SDF cache only, requires reset to apply).

            Parameters
            ----------
            model_name : str
                Name of the model.
            link_name : str
                Name of the link.
            mass : float
                New mass value in kg.

            Warning
            -------
            Changes take effect only after respawn_model() is called.
          )pbdoc")
        .def("set_inertia", static_cast<void (DRLServer::*)(std::string, std::string, Eigen::Matrix3d &)>(&DRLServer::set_inertia), py::arg("model_name"), py::arg("link_name"), py::arg("inertia"), py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Set link inertia tensor (SDF cache only, requires reset to apply).

            Parameters
            ----------
            model_name : str
                Name of the model.
            link_name : str
                Name of the link.
            inertia : numpy.ndarray
                3x3 inertia tensor in kg·m².

            Warning
            -------
            Changes take effect only after respawn_model() is called.
          )pbdoc")
        .def("get_inertia", static_cast<Eigen::Matrix3d (DRLServer::*)(std::string, std::string)>(&DRLServer::get_inertia), py::arg("model_name"), py::arg("link_name"), py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Get link inertia tensor.

            Parameters
            ----------
            model_name : str
                Name of the model.
            link_name : str
                Name of the link.

            Returns
            -------
            numpy.ndarray
                3x3 inertia tensor in kg·m².
          )pbdoc")
        .def("get_mass", &DRLServer::get_mass, py::arg("model_name"), py::arg("link_name"), py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Get link mass.

            Parameters
            ----------
            model_name : str
                Name of the model.
            link_name : str
                Name of the link.

            Returns
            -------
            float
                Mass in kg.
          )pbdoc")
        .def("set_rotor_parameters", &DRLServer::set_rotor_parameters, py::arg("model_name"), py::arg("rotor_params"), py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Set rotor parameters for domain randomization.

            Parameters
            ----------
            model_name : str
                Name of the model.
            rotor_params : RotorParameters
                New rotor physics parameters.

            See Also
            --------
            RotorParameters : Container for rotor physics parameters.
          )pbdoc")
        .def("get_rotor_parameters", &DRLServer::get_rotor_parameters, py::arg("model_name"), py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Get current rotor parameters.

            Parameters
            ----------
            model_name : str
                Name of the model.

            Returns
            -------
            RotorParameters
                Current rotor physics parameters.

            See Also
            --------
            RotorParameters : Container for rotor physics parameters.
          )pbdoc")
        .def("get_rotor_thrust_allocation_matrix", static_cast<Eigen::MatrixXd (DRLServer::*)(std::string)>(&DRLServer::get_rotor_thrust_allocation_matrix), py::arg("model_name"), py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Get rotor allocation matrix for a model, with MultiRotorPlugin

            Parameters
            ----------
            model_name : str
                Name of the model.

            Returns
            -------
            numpy.ndarray
                Rotor allocation matrix mapping rotor thrusts to forces/torques .
          )pbdoc")
        .def("get_rotor_thrust_allocation_matrix", static_cast<Eigen::MatrixXd (DRLServer::*)(std::string, const std::vector<std::string> &, const std::vector<int> &, double)>(&DRLServer::get_rotor_thrust_allocation_matrix), py::arg("model_name"), py::arg("rotor_links"), py::arg("turning_directions"), py::arg("ktau"), py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Get rotor allocation matrix for a model, without the MultiRotorPlugin. 

            Parameters
            ----------
            model_name : str
                Name of the model.
            rotor_links : list[str]
                Rotor link names
            turning_directions : list[int]
                Turning directions of rotors
            ktau : float
                Motor torque constant such that |Tau| = |ktau*SRT|
            Returns
            -------
            numpy.ndarray
                Rotor allocation matrix mapping rotor thrusts to forces/torques .
          )pbdoc")
        .def("get_inverse_rotor_thrust_allocation_matrix", static_cast<Eigen::MatrixXd (DRLServer::*)(std::string)>(&DRLServer::get_inverse_rotor_thrust_allocation_matrix), py::arg("model_name"), py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Get inverse rotor allocation matrix for a model.

            Parameters
            ----------
            model_name : str
                Name of the model.

            Returns
            -------
            numpy.ndarray
                Inverse rotor allocation matrix mapping forces/torques to rotor thrusts.
          )pbdoc")
        .def("get_inverse_rotor_thrust_allocation_matrix", static_cast<Eigen::MatrixXd (DRLServer::*)(std::string, const std::vector<std::string> &, const std::vector<int> &, double)>(&DRLServer::get_inverse_rotor_thrust_allocation_matrix), py::arg("model_name"), py::arg("rotor_links"), py::arg("turning_directions"), py::arg("ktau"), py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Get inverse rotor allocation matrix for a model, without the MultiRotorPlugin. 

            Parameters
            ----------
            model_name : str
                Name of the model.
            rotor_links : list[str]
                Rotor link names
            turning_directions : list[int]
                Turning directions of rotors
            ktau : float
                Motor torque constant such that |Tau| = |ktau*SRT|
            Returns
            -------
            numpy.ndarray
                Inverse Rotor allocation matrix mapping rotor thrusts to forces/torques .
          )pbdoc")
        // force and moment mapping function
        .def("get_thrust_moment_to_rotor_velocity_mapping_function", &DRLServer::get_thrust_moment_to_rotor_velocity_mapping_function, py::arg("model_name"), py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Get thrust and moment to rotor velocity mapping function for a model.

            Parameters
            ----------
            model_name : str
                Name of the model.

            Returns
            -------
            ForceMomentMappingFunction
                Function mapping thrust and moments to rotor speeds.
          )pbdoc")
        // force and moment mapping function
        .def("get_thrust_moment_to_rotor_thrust_mapping_function", static_cast<std::function<Eigen::VectorXd(const Eigen::Vector4d &)> (DRLServer::*)(std::string)>(&DRLServer::get_thrust_moment_to_rotor_thrust_mapping_function), py::arg("model_name"), py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Get thrust and moment to rotor thrust mapping function for a model.
            The model must have MultiRotorPlugin.

            Parameters
            ----------
            model_name : str
                Name of the model.

            Returns
            -------
            ForceMomentMappingFunction
                Function mapping thrust and moments to rotor thrust.
          )pbdoc")
        .def("get_thrust_moment_to_rotor_thrust_mapping_function", static_cast<std::function<Eigen::VectorXd(const Eigen::Vector4d &)> (DRLServer::*)(std::string, const std::vector<std::string> &, std::vector<int> &, double)>(&DRLServer::get_thrust_moment_to_rotor_thrust_mapping_function), py::arg("model_name"), py::arg("rotor_links"), py::arg("turning_directions"), py::arg("ktau"), py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Get thrust and moment to rotor thrust mapping function for a model.
            The model must have MultiRotorPlugin.

            Parameters
            ----------
            model_name : str
                Name of the model.
            rotor_links : list[str]
                Rotor link names
            turning_directions : list[int]
                Turning directions of rotors
            ktau : float
                Motor torque constant such that |Tau| = |ktau*SRT|
            Returns
            -------
            ForceMomentMappingFunction
                Function mapping thrust and moments to rotor thrust.
          )pbdoc")
        .def("request_contact_data", &DRLServer::request_contact_data, py::arg("model_name"), py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Request contact data collection for a model.

            Parameters
            ----------
            model_name : str
                Name of the model to enable contact sensing.

            Notes
            -----
            Has performance impact - only enable for models that need it.
            Contact data can then be retrieved via get_contacts().
          )pbdoc")
        .def("set_srt_rate_limiter_time_constants", &DRLServer::set_srt_rate_limiter_time_constants, py::arg("model_name"), py::arg("link_name"), py::arg("tau_up"), py::arg("tau_down"), py::arg("initial_value"), py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Set the first-order rate limiter time constants.

            Parameters
            ----------
            model_name : str
                Name of the model for which the srt rate limiter will be applied
            link name : str
                Name of the link on the model for which srt rate limiter will be applied
            tau_up : float
                Rising time constant
            tau_down : float
                Falling time constant
            initial_value : numpy.ndarray
                Initial value
          )pbdoc")
        .def("set_ctbr_rate_limiter_time_constants", &DRLServer::set_ctbr_rate_limiter_time_constants, py::arg("model_name"), py::arg("link_name"), py::arg("tau_up"), py::arg("tau_down"), py::arg("initial_value"), py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Set the first-order rate limiter time constants.

            Parameters
            ----------
            model_name : str
                Name of the model for which the ctbr rate limiter will be applied
            link name : str
                Name of the link on the model for which ctbr rate limiter will be applied
            tau_up : float
                Rising time constant
            tau_down : float
                Falling time constant
            initial_value : numpy.ndarray
                Initial value
          )pbdoc")
        .def("set_ctbt_rate_limiter_time_constants", &DRLServer::set_ctbt_rate_limiter_time_constants, py::arg("model_name"), py::arg("link_name"), py::arg("tau_up"), py::arg("tau_down"), py::arg("initial_value"), py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Set the first-order rate limiter time constants.

            Parameters
            ----------
            model_name : str
                Name of the model for which the ctbt rate limiter will be applied
            link name : str
                Name of the link on the model for which ctbt rate limiter will be applied
            tau_up : float
                Rising time constant
            tau_down : float
                Falling time constant
            initial_value : numpy.ndarray
                Initial value
          )pbdoc")
        .def("reset_srt_rate_limiter", &DRLServer::reset_srt_rate_limiter, py::arg("model_name"), py::arg("link_name"), py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Reset the  srt first-order rate limiter, the time constants will not be changed, but the state
            will be reset to initil state.

            Parameters
            ----------
            model_name : str
                Name of the model for which the ctbt rate limiter will be applied
            link name : str
                Name of the link on the model for which ctbt rate limiter will be applied
          )pbdoc")
        .def("reset_ctbr_rate_limiter", &DRLServer::reset_ctbr_rate_limiter, py::arg("model_name"), py::arg("link_name"), py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Reset the ctbr first-order rate limiter, the time constants will not be changed, but the state
            will be reset to initil state.

            Parameters
            ----------
            model_name : str
                Name of the model for which the ctbt rate limiter will be applied
            link name : str
                Name of the link on the model for which ctbt rate limiter will be applied
          )pbdoc")
        .def("reset_ctbt_rate_limiter", &DRLServer::reset_ctbt_rate_limiter, py::arg("model_name"), py::arg("link_name"), py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Reset the ctbt first-order rate limiter, the time constants will not be changed, but the state
            will be reset to initil state.

            Parameters
            ----------
            model_name : str
                Name of the model for which the ctbt rate limiter will be applied
            link name : str
                Name of the link on the model for which ctbt rate limiter will be applied
          )pbdoc")
        .def("sim_iterations", &DRLServer::sim_iterations, py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Get the current simulation iteration count.

            Returns
            -------
            int
                Number of simulation steps executed since the server started.
        )pbdoc")
        .def("set_marker", &DRLServer::set_marker, py::arg("model_name"), py::arg("link_name"), py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Set the marker for a link on a model.

            Parameters
            ----------
            model_name : str
                Name of the model for which the marker will be set
            link name : str
                Name of the link on the model for which marker will be set
          )pbdoc")
        .def("start_camera_recording", &DRLServer::start_camera_recording, py::arg("cam_name"), py::arg("height"), py::arg("width"), py::arg("fps"), py::arg("pos"), py::arg("ori"), py::arg("output_file"), py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Start a new camera recording.

            Spawns a recording camera at the given pose and streams rendered
            frames to a video file until stop_camera_recording() is called.

            Parameters
            ----------
            cam_name : str
                Unique camera name.
            height : int
                Height of each frame in pixels.
            width : int
                Width of each frame in pixels.
            fps : int
                Frames per second of the output video.
            pos : numpy.ndarray
                Initial camera position [x, y, z] in meters.
            ori : numpy.ndarray
                Initial camera orientation [roll, pitch, yaw] in radians.
            output_file : str
                Output video file path (.mp4, .avi, or .ogv).

            Notes
            -----
            Blocks until the recording camera is live, stepping the
            simulation as needed. In a world with no other rendering
            sensors, the first recording also initializes the render
            scene, which can advance simulation time noticeably.

            See Also
            --------
            update_camera_recording_pose : Move the recording camera.
            stop_camera_recording : Finalize the video file.

            Example
            -------
            >>> server.start_camera_recording(
            ...     "chase_cam", 480, 640, 30,
            ...     np.array([0., 0., 2.]), np.zeros(3), "episode.mp4")
          )pbdoc")
        .def("update_camera_recording_pose", &DRLServer::update_camera_recording_pose, py::arg("cam_name"), py::arg("pos"), py::arg("ori"), py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Update the position and orientation of a recording camera.

            Parameters
            ----------
            cam_name : str
                Name of the camera (as passed to start_camera_recording).
            pos : numpy.ndarray
                New position [x, y, z] in meters.
            ori : numpy.ndarray
                New orientation [roll, pitch, yaw] in radians.

            See Also
            --------
            start_camera_recording : Start a recording.
          )pbdoc")
        .def("stop_camera_recording", &DRLServer::stop_camera_recording, py::arg("cam_name"), py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
            Stop recording and finalize the video file.

            Parameters
            ----------
            cam_name : str
                Name of the camera (as passed to start_camera_recording).

            Notes
            -----
            Blocks until the video file is fully written, stepping the
            simulation as needed (typically 1-2 steps).

            See Also
            --------
            start_camera_recording : Start a recording.
          )pbdoc");

    m.def("wrap", [](double angle)
          { return std::fmod(angle + M_PI, 2 * M_PI) - M_PI; }, py::arg("angle"), py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Wrap angle to [-π, π] range.

        Parameters
        ----------
        angle : float
            Angle in radians.

        Returns
        -------
        float
            Wrapped angle in range [-π, π].

        Example
        -------
        >>> gzdrl.wrap(4.0)  # Returns ~-2.28
      )pbdoc");

    bind_async_drl_server_pool(m);
};
