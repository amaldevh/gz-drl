// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef TUNED_GAINS_HH
#define TUNED_GAINS_HH
#include <unordered_map>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <atomic>

/**
 * @brief A struct holding typical parameters required for a controller
 *
 */
struct ControllerParameters
{
    Eigen::Matrix3d inertia;                                        ///< Inertia of the Model
    double mass;                                                    ///< Mass of the Model
    double gravity_mag;                                             ///< Magnitude of gravity
    Eigen::Vector3d gravity_vec;                                    ///< Gravity vector
    Eigen::Vector3d max_accel;                                      ///< Maximum acceleration of the Model
    std::unordered_map<std::string, Eigen::VectorXd> miscellaneous; ///< Miscellaneous parameters
};

namespace _gains
{
    static std::unordered_map<std::string, std::unordered_map<std::string, std::unordered_map<std::string, Eigen::VectorXd>>> GAIN_MAP;
    static std::unordered_map<std::string, std::unordered_map<std::string, ControllerParameters>> PARAMETER_MAP;
    static std::atomic<bool> _initialized_map_{false};

    /**
     * @brief Initializes the control gain and maps
     *
     */
    inline static auto _initialize_map()
    {
        // Geometric controller
        GAIN_MAP["qdrone2"]["geometric_controller"] = {{"kp", Eigen::Vector3d(2.0, 2.0, 15.0)},
                                                       {"kd", Eigen::Vector3d(2.0, 2.0, 11.5)},
                                                       {"kp_att", Eigen::Vector3d(25.0, 25.0, 1.0)},
                                                       {"kd_att", Eigen::Vector3d(1.2, 1.2, 0.2)}};
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        I(0, 0) = 0.0147209;
        I(1, 1) = 0.0169101;
        I(2, 2) = 0.029448;
        PARAMETER_MAP["qdrone2"]["geometric_controller"].inertia = I;
        PARAMETER_MAP["qdrone2"]["geometric_controller"].max_accel = Eigen::Vector3d(6.8, 6.8, 10);
        PARAMETER_MAP["qdrone2"]["geometric_controller"].mass = 1.2;
        PARAMETER_MAP["qdrone2"]["geometric_controller"].gravity_vec = Eigen::Vector3d(0.0, 0.0, -9.81);
        PARAMETER_MAP["qdrone2"]["geometric_controller"].gravity_mag = 9.81;

        // Inner loop controller
        GAIN_MAP["qdrone2"]["inner_loop_controller"] = {{"kp_angle", Eigen::Vector3d(12.0, 12.0, 5)},
                                                        {"kd_angle", Eigen::Vector3d(0.1, 0.1, 0.05)},
                                                        {"kp_angular_rate", Eigen::Vector3d(0.1876, 0.1544, 0.09)},
                                                        {"kd_angular_rate", Eigen::Vector3d(0.0032, 0.0026, 0.0001)}};
        PARAMETER_MAP["qdrone2"]["inner_loop_controller"].inertia = I;
        PARAMETER_MAP["qdrone2"]["inner_loop_controller"].max_accel = Eigen::Vector3d(20.0, 20.0, 35.0);
        PARAMETER_MAP["qdrone2"]["inner_loop_controller"].mass = 1.54;
        PARAMETER_MAP["qdrone2"]["inner_loop_controller"].gravity_vec = Eigen::Vector3d(0.0, 0.0, -9.82);
        PARAMETER_MAP["qdrone2"]["inner_loop_controller"].gravity_mag = 9.82;

        // SMC Controller
        GAIN_MAP["qdrone2"]["sliding_mode_controller"] = {{"lambda_pos", Eigen::Vector3d(2.5, 2.5, 3.0)},
                                                          {"kappa_pos", Eigen::Vector3d(4.0, 4.0, 5.0)},
                                                          {"lambda_att", Eigen::Vector3d(8.0, 8.0, 5.0)},
                                                          {"kappa_att", Eigen::Vector3d(0.8, 0.8, 0.4)},
                                                          {"boundary_pos", Eigen::Vector3d(0.5, 0.5, 0.5)},
                                                          {"boundary_att", Eigen::Vector3d(0.2, 0.2, 0.2)}};
        PARAMETER_MAP["qdrone2"]["sliding_mode_controller"].inertia = I;
        PARAMETER_MAP["qdrone2"]["sliding_mode_controller"].max_accel = Eigen::Vector3d(10.0, 10.0, 10.0);
        PARAMETER_MAP["qdrone2"]["sliding_mode_controller"].mass = 1.53;
        PARAMETER_MAP["qdrone2"]["sliding_mode_controller"].gravity_vec = Eigen::Vector3d(0.0, 0.0, -9.81);
        PARAMETER_MAP["qdrone2"]["sliding_mode_controller"].gravity_mag = 9.81;

        // Qunaser UAV controller
        GAIN_MAP["qdrone2"]["pid_controller"] = {{"kp", Eigen::Vector4d(0.4712, 0.4712, 36.0000, 7.5000)},
                                                         {"kd", Eigen::Vector4d(0.6283, 0.6283, 28.0000, 0.4000)},
                                                         {"ki", Eigen::Vector4d(0.0100, 0.0100, 8.0000, 0)},
                                                         {"kp_att", Eigen::Vector3d(12.0, 12.0, 2.0)},
                                                         {"kd_att", Eigen::Vector3d(0.1, 0.1, 0.0)},
                                                         {"kp_omega", Eigen::Vector3d(0.1876, 0.1544, 0.0790)},
                                                         {"kd_omega", Eigen::Vector3d(0.0032, 0.0026, 0.0100)},
                                                         {"integral_sat_limit", Eigen::Vector4d(0.1950, 0.2475, 11.2500, 0.0375)},
                                                         {"stabilization_command_sat_limit", Eigen::Vector4d(0.7854, 0.7854, 35.0000, 1.7453)},
                                                         {"angle_cmd_limit", Eigen::Vector3d(0.7854, 0.7854, 1.7453)},
                                                         {"omega_cmd_limit", Eigen::Vector3d(10.4720, 10.4720, 10.4720)},
                                                         {"max_torque", Eigen::Vector3d(1.0915, 0.8984, 0.0984)}};

        PARAMETER_MAP["qdrone2"]["pid_controller"].inertia = I;
        PARAMETER_MAP["qdrone2"]["pid_controller"].max_accel = Eigen::Vector3d(10.0, 10.0, 12.0);
        PARAMETER_MAP["qdrone2"]["pid_controller"].mass = 1.53;
        PARAMETER_MAP["qdrone2"]["pid_controller"].gravity_vec = Eigen::Vector3d(0.0, 0.0, -9.81);
        PARAMETER_MAP["qdrone2"]["pid_controller"].gravity_mag = 9.81;
        PARAMETER_MAP["qdrone2"]["pid_controller"].miscellaneous["omega_n"] = Eigen::Matrix<double, 1, 1>{100.0};
        PARAMETER_MAP["qdrone2"]["pid_controller"].miscellaneous["zeta"] = Eigen::Matrix<double, 1, 1>{1.0};
        PARAMETER_MAP["qdrone2"]["pid_controller"].miscellaneous["dt"] = Eigen::Matrix<double, 1, 1>{1e-3};
    };

    /**
     * @brief Get the gains object
     *
     * @return const auto& immutable reference to GAIN_MAP
     */
    inline const auto &get_gains()
    {
        if (!_initialized_map_.load())
        {
            _initialize_map();
            _initialized_map_.store(true);
            return GAIN_MAP;
        }
        return GAIN_MAP;
    }
    /**
     * @brief Get the parameters object
     *
     * @return const auto&  immutable reference to PARAMETER_MAP
     */
    inline const auto &get_parameters()
    {
        if (!_initialized_map_.load())
        {
            _initialize_map();
            _initialized_map_.store(true);
            return PARAMETER_MAP;
        }
        return PARAMETER_MAP;
    }
};

#define GAIN_MAP _gains::get_gains()
#define PARAMETER_MAP _gains::get_parameters()

#endif