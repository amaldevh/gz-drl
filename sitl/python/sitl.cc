// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef SITL_CC
#define SITL_CC

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <string.h>
#include "second_order_lp_filter.hh"
#if ROS_VER ==1 || ROS_VER == 2
#include "ros_rl_server.hh"
#endif

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    py::class_<sitl::SecondOrderLPFilter<3, double>>(m, "SecondOrderLPFilter3d",
        R"pbdoc(
            Second order low-pass filter for 3-dimensional ``double`` signals.

            The filter is realised as a discrete-time biquad obtained by applying
            the bilinear (Tustin) transform to the continuous-time prototype

            .. math::

                H(s) = \frac{\omega_n^2}{s^2 + 2\,\eta\,\omega_n\,s + \omega_n^2}

            Coefficients are computed once at construction from the cutoff
            frequency, damping ratio and time step.
        )pbdoc")
        .def(py::init<double, double, double>(),
             py::arg("omega_n"),
             py::arg("eta"),
             py::arg("dt"),
             R"pbdoc(
                Construct the filter and compute its coefficients.

                Parameters
                ----------
                omega_n : float
                    Cutoff (natural) frequency in rad/s.
                eta : float
                    Damping ratio.
                dt : float
                    Sampling time step in seconds.
             )pbdoc")
        .def("reset", &sitl::SecondOrderLPFilter<3, double>::Reset,
            py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
                Reset the internal input/output state buffers to zero.
            )pbdoc")
        .def("update", &sitl::SecondOrderLPFilter<3, double>::Update,
             py::arg("input"),
            py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(
                Advance the filter by one sample.

                Parameters
                ----------
                input : numpy.ndarray
                    New 3-element input sample.

                Returns
                -------
                numpy.ndarray
                    The filtered 3-element output sample.
            )pbdoc");
        
    /** @brief ROS bindings if defined */
    #if ROS_VER ==1 || ROS_VER == 2
    /** @brief ROS RL Server bindings if defined */
    py::class_<RosDRLServer, std::shared_ptr<RosDRLServer>>(m,
        "RosDRLServer",
        R"pbdoc(
            ROS-backed Deep Reinforcement Learning server.

            Wraps a :class:`DRLServer` driving a Gazebo simulation and exposes
            its state and actuation over ROS topics. It manages the ROS node,
            publishers/subscribers and the simulation stepping loop, and can be
            used both for DRL training and for software-in-the-loop (SITL) runs.
        )pbdoc")
    .def(py::init<const std::string&,
                  const std::string&,
                  const std::vector<std::string>&,
                  bool, double >(),
         py::arg("partition"),
         py::arg("sdf_file"),
         py::arg("model_names"),
         py::arg("enable_sensors"),
         py::arg("rtf"),
         R"pbdoc(
            Construct a RosDRLServer.

            Parameters
            ----------
            partition : str
                Gazebo partition name.
            sdf_file : str
                Path to the world SDF file to load.
            model_names : list of str
                Names of the models to control.
            enable_sensors : bool
                Enable the sensor (camera/lidar/odom) interface.
            rtf : float
                Real-time factor. At ``rtf = 1.0`` the server is stepped
                ``1 / physics_dt`` times per second (hardware permitting).
         )pbdoc")
     .def(py::init<const std::string&,
                  const std::string&,
                  const std::vector<std::string>&,
                  bool, double,
                   const std::unordered_map<std::string, std::vector<std::string>> &>(),
         py::arg("partition"),
         py::arg("sdf_file"),
         py::arg("model_names"),
         py::arg("enable_sensors"),
         py::arg("rtf"),
        py::arg("link_map"),
         R"pbdoc(
            Construct a RosDRLServer with an explicit link map.

            Parameters
            ----------
            partition : str
                Gazebo partition name.
            sdf_file : str
                Path to the world SDF file to load.
            model_names : list of str
                Names of the models to control.
            enable_sensors : bool
                Enable the sensor (camera/lidar/odom) interface.
            rtf : float
                Real-time factor. At ``rtf = 1.0`` the server is stepped
                ``1 / physics_dt`` times per second (hardware permitting).
            link_map : dict of {str : list of str}
                Map of model name to the link names used for sensor attachment.
         )pbdoc")
    .def("spin",
         &RosDRLServer::Spin,
         py::call_guard<py::gil_scoped_release>(),
         R"pbdoc(
            Spin the ROS executor in the calling thread (blocking).
         )pbdoc")
    .def("spin_async",
         &RosDRLServer::SpinAsync,
         R"pbdoc(
            Start spinning the ROS executor in a background thread (non-blocking).
         )pbdoc")
    .def("run",
         &RosDRLServer::Run,
         R"pbdoc(
            Run the simulation at the real-time factor given at construction.
         )pbdoc")
    .def("pause",
         &RosDRLServer::Pause,
         R"pbdoc(
            Pause the simulation.
         )pbdoc")
    .def("get_published_topic_map",
         &RosDRLServer::GetPublishedTopicMap,
         R"pbdoc(
            Get the map of topics published by this server.

            Returns
            -------
            dict of {str : list of str}
                Map of model name to published topic names.
         )pbdoc")
    .def("get_subscribed_topic_map",
         &RosDRLServer::GetSubscribedTopicMap,
         R"pbdoc(
            Get the map of topics subscribed to by this server.

            Returns
            -------
            dict of {str : list of str}
                Map of model name to subscribed topic names.
         )pbdoc")
     .def("server", &RosDRLServer::Server,
          R"pbdoc(
            Get the underlying :class:`DRLServer` instance.
          )pbdoc");
    #endif
};
#endif // SITL_CC
