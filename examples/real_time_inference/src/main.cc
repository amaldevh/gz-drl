// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#include "onnx.hh"
#include "ros_rl_server.hh"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <atomic>
#include <cmath>
#include "common/high_precision_timer.hh"
#include <map>
#include <tuple>
#include "controllers/geometric_controller.hh"
#include "controllers/tuned_gains.hh"
#include <functional>
#include "trajectory_planning_processor.hh"
#include "common/waypoint_generator.hh"

std::map<std::string, std::string> parse_args(int argc, char** argv){
    std::map<std::string, std::string> arg_map;
    auto help_func = [&](){
            std::cout<<"Usage: "<<argv[0]<<" [options]"<<std::endl;
            std::cout<<"Options:"<<std::endl;
            std::cout<<"  --help, -h: Show this help message"<<std::endl;
            std::cout<<"  --model_path <path>: Path to ONNX model file (default: model.onnx)"<<std::endl;
            std::cout<<"  --envid <id>: Environment ID for DRLServer (default: 0)"<<std::endl;
            std::cout<<"  --sdf_file <path>: Path to SDF file for DRLServer (default: world_simple.sdf)"<<std::endl;
            std::cout<<"  --model_name <name>: Model name in SDF for DRLServer (default: quadrotor)"<<std::endl;
            std::cout<<"  --base_link_name <name>: Base link name in SDF for DRLServer (default: base_link)"<<std::endl;
            std::cout<<"  --log_csv <path>: CSV log file path (default: uav_policy_log.csv)"<<std::endl;
            std::cout<<"  --waypoints_yaml <path>: YAML waypoint export path (default: waypoints.yaml)"<<std::endl;
            std::cout<<"  --duration_seconds <seconds>: Run duration (default: 240)"<<std::endl;
            exit(0);
    };
    if (argc == 1){
        help_func();
    }
    for (int i = 1; i < argc; ++i){
        if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h"){
            help_func();
        }
        if (i + 1 < argc && std::string(argv[i]) == "--model_path"){
            arg_map["model_path"] = std::string(argv[i + 1]);
            ++i;
        }
        if (i + 1 < argc && std::string(argv[i]) == "--envid"){
            arg_map["envid"] = std::string(argv[i + 1]);
            ++i;
        }
        if (i + 1 < argc && std::string(argv[i]) == "--sdf_file"){
            arg_map["sdf_file"] = std::string(argv[i + 1]);
            ++i;
        }
        if (i + 1 < argc && std::string(argv[i]) == "--model_name"){
            arg_map["model_name"] = std::string(argv[i + 1]);
            ++i;
        }
        if (i + 1 < argc && std::string(argv[i]) == "--base_link_name"){
            arg_map["base_link_name"] = std::string(argv[i + 1]);
            ++i;
        }
        if (i + 1 < argc && std::string(argv[i]) == "--log_csv"){
            arg_map["log_csv"] = std::string(argv[i + 1]);
            ++i;
        }
        if (i + 1 < argc && std::string(argv[i]) == "--waypoints_yaml"){
            arg_map["waypoints_yaml"] = std::string(argv[i + 1]);
            ++i;
        }
        if (i + 1 < argc && std::string(argv[i]) == "--duration_seconds"){
            arg_map["duration_seconds"] = std::string(argv[i + 1]);
            ++i;
        }
    }
    return arg_map;
}
decltype(auto) generate_waypoints(std::string yaml_path){
        const bool USE_XYZ  =true ;
        const bool USE_VXYZ  =true;
        const double Z_OFFSET  =1.0;
        const double MIN_DIST  =1.0;
        const double MAX_DIST  =1.0;
        const int ACTION_HISTORY_SIZE = 10;
        const int STATE_HISTORY_SIZE = 10;
        const double GATE_WIDTH = 0.4;
        const double GATE_HEIGHT = 0.4;
        const double GATE_CROSSING_RADIUS = 0.9;
        // const double M_PI_ = 3.141592;
        const std::vector<double> BASE_CURVE_AMPS{1.5, 1.5, 0.32};
        const std::vector<double> BASE_CURVE_FREQS{1.0, 2.0, 2.0};
        const std::vector<double> BASE_CURVE_PHASES {0.0, 0.0, M_PI/2.0};
        std::vector<std::vector<float>> waypoints;
        WaypointGenerator way_gen;
        std::cout<<"Curve amps-x: "<<BASE_CURVE_AMPS[0]<<"\n";
        std::cout<<"Curve amps-y: "<<BASE_CURVE_AMPS[1]<<"\n";
        std::cout<<"Curve amps-z: "<<BASE_CURVE_AMPS[2]<<"\n";
        std::cout<<"Curve freq-x: "<<BASE_CURVE_FREQS[0]<<"\n";
        std::cout<<"Curve freq-y: "<<BASE_CURVE_FREQS[1]<<"\n";
        std::cout<<"Curve freq-z: "<<BASE_CURVE_FREQS[2]<<"\n";
        std::cout<<"Curve phase-x: "<<BASE_CURVE_PHASES[0]<<"\n";
        std::cout<<"Curve phase-y: "<<BASE_CURVE_PHASES[1]<<"\n";
        std::cout<<"Curve phase-z: "<<BASE_CURVE_PHASES[2]<<"\n";
        std::cout<<"Curve min dist: "<< MIN_DIST<<"\n";
        std::cout<<"Curve max dist: "<<MAX_DIST<<"\n";
        waypoints = way_gen.GenerateLissajousWaypoints(
            BASE_CURVE_AMPS[0],
            BASE_CURVE_AMPS[1],
            BASE_CURVE_AMPS[2],
            BASE_CURVE_FREQS[0],
            BASE_CURVE_FREQS[1],
            BASE_CURVE_FREQS[2],
            BASE_CURVE_PHASES[0],
            BASE_CURVE_PHASES[1],
            BASE_CURVE_PHASES[2],
            MIN_DIST,
            MAX_DIST,
            BASE_CURVE_AMPS[2]+0.5
        );
        for (size_t i =0; i< waypoints.size(); ++i){
            std::cout<<"Waypoint: "<<i<<"\n";
            for (const auto elem: waypoints[i])
                std::cout<<elem<<",";
            std::cout<<"\n";
        }
        Eigen::Vector3f xyz_scaling;
        Eigen::Vector3f vxyz_scaling;
        xyz_scaling << 0.22, 0.22 , 0.22;
        vxyz_scaling<<0.06, 0.06, 0.06;
        print_info("Updating processor with " + std::to_string(waypoints.size()) + " waypoints.\n");
        auto processor_ = std::make_shared<TrajectoryPlanningProcessor>(ACTION_HISTORY_SIZE, STATE_HISTORY_SIZE,
            USE_XYZ, USE_VXYZ,
            waypoints, GATE_WIDTH, GATE_HEIGHT, GATE_CROSSING_RADIUS,
            xyz_scaling, vxyz_scaling);
        processor_->Reset(0);
        // write yaml
        std::ofstream yaml_file(yaml_path, std::ios::out | std::ios::trunc);
        if (!yaml_file.is_open()) {
            std::cerr << "[Error] Failed to open waypoints YAML file: " << yaml_path << std::endl;
            return std::shared_ptr<TrajectoryPlanningProcessor>(nullptr);
        }

        yaml_file << std::fixed << std::setprecision(6);
        yaml_file << "num_waypoints: " << waypoints.size() << "\n";
        yaml_file << "waypoints:\n";

        for (size_t waypoint_index = 0; waypoint_index < waypoints.size(); ++waypoint_index) {
            const auto& corners = processor_->gate_corners_.row(waypoint_index);
            const auto& center = processor_->gate_centers_.row(waypoint_index);
            yaml_file << "  - id: " << waypoint_index << "\n";
            yaml_file << "    center: ["
                      << center.x() << ", " << center.y() << ", " << center.z() << "]\n";
            yaml_file << "    corners:\n";
            for (size_t corner_idx = 0; corner_idx < 4; ++corner_idx) {
                const auto& corner = corners.segment<3>(corner_idx * 3);
                yaml_file << "      - ["
                          << corner.x() << ", " << corner.y() << ", " << corner.z() << "]\n";
            }
        }


        return processor_;
    }



int main(int argc, char** argv){
    auto args = parse_args(argc, argv);
    auto model_path = args["model_path"];
    auto envid = args.count("envid") ? std::stoi(args["envid"]) : 0;
    auto sdf_file = args.count("sdf_file") ? args["sdf_file"]
        : "world_simple.sdf";
    auto model_name = args.count("model_name") ? args["model_name"] : "quadrotor";
    auto base_link_name = args.count("base_link_name") ? args["base_link_name"] : "quadrotor/base_link";
    auto log_csv_path = args.count("log_csv") ? args["log_csv"] : "uav_policy_log.csv";
    auto waypoints_yaml_path = args.count("waypoints_yaml") ? args["waypoints_yaml"] : "waypoints.yaml";
    const double duration_seconds = args.count("duration_seconds")
        ? std::stod(args["duration_seconds"]) : 240.0;
    if (!std::isfinite(duration_seconds) || duration_seconds <= 0.0) {
        std::cerr << "[Error] --duration_seconds must be finite and greater than zero." << std::endl;
        return 1;
    }

    std::ofstream log_file(log_csv_path, std::ios::out | std::ios::trunc);
    if (!log_file.is_open()) {
        std::cerr << "[Error] Failed to open log CSV file: " << log_csv_path << std::endl;
        return 1;
    }
    log_file << std::fixed << std::setprecision(6);
    log_file << "step";
    for (int i = 0; i < 13; ++i) log_file << ",state_" << i;
    for (int i = 0; i < 6; ++i) log_file << ",action_raw_" << i;
    for (int i = 0; i < 6; ++i) log_file << ",action_scaled_" << i;
    for (int i = 0; i < 13; ++i) log_file << ",reference_" << i;
    log_file << ",reward,total_reward";
    log_file << "\n";
    std::atomic<uint64_t> log_step{0};
    RosDRLServer ros_server("real_time" + std::to_string(envid), sdf_file, {model_name}, true, 1.0);
    // set a controller
    const auto &geometric_controller_gains = GAIN_MAP.at("qdrone2").at("geometric_controller");
    const auto& controller_params = PARAMETER_MAP.at("qdrone2").at("geometric_controller");

    auto controller = std::make_shared<GeometricController>(geometric_controller_gains.at("kp"),
                                            geometric_controller_gains.at("kd"),
                                            geometric_controller_gains.at("kp_att"),
                                            geometric_controller_gains.at("kd_att"),
                                            controller_params.max_accel,
                                            controller_params.gravity_vec,
                                            controller_params.mass,
                                            controller_params.inertia);
    ros_server.WithServerLocked([&](DRLServer &server) {
        server.set_trajectory_trace(model_name, base_link_name);
        server.run_N(5);
        Eigen::Vector3d position(0, 0, 0.15);
        Eigen::Vector3d orientation(0, 0, 0);
        server.reset_pos(model_name, position, orientation);
        server.set_controller(model_name, base_link_name, controller);
        server.update_control_states();
    });
    // Spin only after configuration. All raw server access below uses the
    // wrapper's shared lock, the same lock used by ROS callbacks.
    ros_server.SpinAsync();

    if (model_path.empty()) std::cerr<<"[Warning] No model path provided. Using default: model.onnx"<<std::endl;
    
    ONNXModel model(model_path.empty() ? "model.onnx" : model_path);
    Eigen::Matrix<float, OBSERVATION_DIM, 1> observation; // Example input size
    observation.setZero(); // Fill with random data for testing

    // Main inference pipeline
    // create timer lambdas
    Eigen::Matrix<double, 13, 1> des_state;
    des_state.setZero();
    des_state(6) = 1.0; // Set desired orientation as identity quaternion
    // start waypoint
    int waypoint_idx = 0;
    int laps = 0;
    std::shared_ptr<TrajectoryPlanningProcessor> processor = generate_waypoints(waypoints_yaml_path);
    if (!processor) {
        std::cerr << "[Error] Failed to generate waypoints and processor->" << std::endl;
        return 1;
    }
    processor->Reset(0);
    
    std::cout << "[Info] Wrote waypoint YAML to: " << waypoints_yaml_path << std::endl;

    // max and min vel and omega, for safety checks
    double max_vel = -1.0; // m/s
    double max_omega = -1.0; // rad/s
    double min_vel = 100.0;
    double min_omega = 100.0;
    bool initialized = false;
    float total_reward = 0.0f;

    Stated last_control_state = Stated::Zero();
    Stated last_control_state_dot = Stated::Zero();
    ros_server.WithServerLocked([&](DRLServer &server) {
        const auto &states = server.control_states.at(model_name).at(base_link_name);
        last_control_state = std::get<0>(states);
        last_control_state_dot = std::get<1>(states);
    });

    auto server_run_func = [&ros_server, &model_name,
        &base_link_name, &des_state](){
        ros_server.WithServerLocked([&](DRLServer &server) {
            server.control_with_rotor_velocity(model_name, base_link_name, des_state, 1);
        });
    };
    
    auto inference_run_func = [&model, &observation, &ros_server,
         &model_name, &base_link_name, &last_control_state, &last_control_state_dot,
         &waypoint_idx, &processor,
        &des_state, &max_vel, &max_omega, &min_vel, &min_omega,
        &log_file, &log_step, &laps, &initialized, &total_reward](){
        Stated control_state;
        Stated control_state_dot;
        ros_server.WithServerLocked([&](DRLServer &server) {
            const auto &states = server.control_states.at(model_name).at(base_link_name);
            control_state = std::get<0>(states);
            control_state_dot = std::get<1>(states);
        });
        // prepare obs 
        const auto &curr_waypoint = processor->gate_centers_.row(waypoint_idx);
        if (!initialized) {
            ros_server.WithServerLocked([&](DRLServer &) {
                des_state.head<3>() = curr_waypoint.cast<double>();
                des_state(6) = 1.0;
            });
            if ((control_state.head<3>().cast<float>() - des_state.head<3>().cast<float>()).norm() <0.4f) {
                initialized = true;
                std::cout<<"Initialized! Starting inference loop..."<<std::endl;
            } else {
                std::cout<<"Waiting to initialize... Current position: "<<control_state.head<3>().transpose()<<std::endl;
                std::cout<<"Desired position: "<<des_state.head<3>().transpose()<<std::endl;
                std::cout<<"Distance to desired: "<<(control_state.head<3>().cast<float>() - des_state.head<3>().cast<float>()).norm()<<std::endl;
            }
            last_control_state = control_state;
            last_control_state_dot = control_state_dot;
            return; // Skip the first inference to allow server to initialize states
        }
        Eigen::Matrix<double, 13, 1> control_state_snapshot;
        Eigen::VectorXf raw_action_snapshot;
        Eigen::VectorXf scaled_action_snapshot;
        Eigen::Matrix<double, 13, 1> des_state_snapshot;
       
        processor->ProcessObservation(control_state.cast<float>(),
                                      control_state_dot.cast<float>(),
                                      last_control_state.cast<float>(),
                                      last_control_state_dot.cast<float>(),
                                      observation);
        Eigen::VectorXf obs_tmp = observation;
        Eigen::VectorXf action_tmp = Eigen::VectorXf::Zero(ACTION_DIM);
        model.RunInference(obs_tmp, action_tmp);
        float reward = 0.0;
        Eigen::VectorXf scaled_action = action_tmp;
        processor->ProcessAction(action_tmp, scaled_action);
        ros_server.WithServerLocked([&](DRLServer &) {
            double desired_yaw = 0.0;
            des_state(6) = std::cos(desired_yaw / 2.0);
            des_state(9) = std::sin(desired_yaw / 2.0);
            des_state.head<6>() = control_state.head<6>() + scaled_action.cast<double>();
        });
        processor->ComputeReward(control_state.cast<float>(),
             control_state_dot.cast<float>(),
             last_control_state.cast<float>(),
             last_control_state_dot.cast<float>(),
             scaled_action, reward);
        total_reward += reward;

            // Check done
            if (std::abs(control_state(0)) > 3.5f || std::abs(control_state(1)) > 3.5f || control_state(2) > 3.5f) {
            std::cout<<"Crash detected: Out of bounds: "<<control_state.head<3>().transpose()<<std::endl;
                }
            Eigen::Matrix3d R = Eigen::Quaterniond(control_state(6), control_state(7), control_state(8), control_state(9)).toRotationMatrix();
            double yaw_curr = std::atan2(R(1,0), R(0,0));
            // roll/pitch angle from body z vs world z
            double dot = std::clamp(static_cast<double>(R.col(2).dot(Eigen::Vector3d::UnitZ())), -1.0, 1.0);
            if (std::abs(std::acos(dot)) > static_cast<float>(169.0 * M_PI / 180.0f)) {
                std::cout<<"Crash detected: excessive roll/pitch angle."<<std::endl;
            }
            if (std::abs(yaw_curr) > static_cast<float>(M_PI/2.0)){ // yaw angle check
                std::cout<<"Crash detected: excessive yaw angle."<<std::endl;
            }

        control_state_snapshot = control_state;
        raw_action_snapshot = action_tmp;
        scaled_action_snapshot = scaled_action;
        ros_server.WithServerLocked([&](DRLServer &) {
            des_state_snapshot = des_state;
        });
        last_control_state = control_state;
        last_control_state_dot = control_state_dot;

        const auto current_step = log_step.fetch_add(1, std::memory_order_relaxed);
        log_file << current_step;
        for (int i = 0; i < 13; ++i) log_file << "," << control_state_snapshot(i);
        for (int i = 0; i < 6; ++i) log_file << "," << raw_action_snapshot(i);
        for (int i = 0; i < 6; ++i) log_file << "," << scaled_action_snapshot(i);
        for (int i = 0; i < 13; ++i) log_file << "," << des_state_snapshot(i);
        log_file<<","<<reward<<","<<total_reward;
        log_file << "\n";
        // update desired state

    };
    // create timers
    HighPrecisionTimer server_timer( 1000.0, server_run_func); // 1000 Hz Physics
    HighPrecisionTimer inference_timer(100.0, inference_run_func); // 100 Hz Inference
    bool run = true;
    auto start_time = std::chrono::system_clock::now();
    while (run) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Sleep to prevent busy waiting
        auto tf = std::chrono::system_clock::now();
        if (std::chrono::duration<double>(tf - start_time).count() >= duration_seconds){
            run = false;
        }
    }
    server_timer.stop();
    inference_timer.stop();
    return 0;
}
