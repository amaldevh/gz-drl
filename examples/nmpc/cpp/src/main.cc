// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#include "nmpc_controller.hh"
#include "rl_server.hh"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>

// Trajectory generator function
struct TrajectoryPoint {
  Eigen::Vector3d position;
  Eigen::Vector3d velocity;
};

TrajectoryPoint generate_trajectory(double t) {
  TrajectoryPoint point;
  
  // Lissajous curve trajectory
  double x = std::cos(2.0 * M_PI / 4.0 * t) * 2.5;
  double y = std::sin(3.0 * M_PI / 4.0 * t) * 2.5;
  double z = 2.0 + 0.5 * std::sin(1.0 * M_PI / 4.0 * t);
  
  double vx = -std::sin(2.0 * M_PI / 4.0 * t) * (2.0 * M_PI / 4.0) * 2.5;
  double vy = std::cos(3.0 * M_PI / 4.0 * t) * (3.0 * M_PI / 4.0) * 2.5;
  double vz = 0.5 * std::cos(1.0 * M_PI / 4.0 * t) * (1.0 * M_PI / 4.0);
  
  point.position << x, y, z;
  point.velocity << vx, vy, vz;
  
  return point;
}

void test_trajectory_tracking(DRLServer& server,
                              NMPCController& controller,
                              const std::string& uav_model = "quadrotor",
                              const std::string& uav_canonical_link = "quadrotor/base_link") {
  
  std::cout << "Starting trajectory tracking test..." << std::endl;
  
  const int control_type = (QUADROTOR_NX==13)? 0 : 1;
  // Reset quadrotor to initial position
  Eigen::Vector3d initial_pos(0.0, 0.0, 0.15);
  Eigen::Vector3d initial_rpy(0.0, 0.0, 0.0);
  server.reset_pos(uav_model, initial_pos, initial_rpy);
  server.run_N(10);
  
  // Storage for logging
  std::vector<uav_controllers::X> states;
  std::vector<uav_controllers::X> desired_states;
  states.reserve(10000);
  desired_states.reserve(10000);
  
  // Get thrust-moment to rotor velocity mapping
  auto mapping_func = server.get_thrust_moment_to_rotor_velocity_mapping_function(uav_model);
  
  // Desired state vector
  uav_controllers::X des_state = uav_controllers::X::Zero();
  des_state(6) = 1.0;  // quaternion w component
  
  auto start_time = std::chrono::high_resolution_clock::now();
  
  // Run simulation for 10 seconds (1000 iterations at 100 Hz)
  for (int i = 0; i < 1000; i++) {
    double t = i * 0.01;  // 10 ms timestep
    
    // Generate desired trajectory
    TrajectoryPoint traj = generate_trajectory(t);
    des_state.setZero();
    des_state.segment<3>(0) = traj.position;
    des_state.segment<3>(3) = traj.velocity;
    des_state(6) = 1.0;  // keep level orientation
    
    // Run 10 simulation steps per control update (1 kHz sim, 100 Hz control)
    for (int j = 0; j < 10; j++) {
      // Update control states
      server.update_control_states();
      
      // Get current state
      auto [state, state_dot] = server.control_states[uav_model][uav_canonical_link];
      
      // Log data
      states.push_back(state);
      desired_states.push_back(des_state);
      
      if (control_type == 0){
        // Calculate thrust and moments
        Eigen::Vector4d TM = controller.calculate_thrust_moments(state, state_dot, des_state);
        // Set rotor velocities
        server.set_rotor_velocity_cmd(uav_model, uav_canonical_link, mapping_func(TM));
      }
      else{
        // Calculate thrust and moments
        Eigen::Vector4d TB = controller.calculate_thrust_bodyrates(state, state_dot, des_state);
        // Set wrench and angular velocity
        Eigen::Quaterniond quat(state(6), state(7), state(8), state(9));
        Eigen::Matrix3d rot = quat.normalized().toRotationMatrix();
        Eigen::Vector3d T_i = rot*Eigen::Vector3d(0.0, 0.0, TB(0));
        Eigen::Vector3d W_i =  Eigen::Vector3d::Zero();
        server.set_wrench(uav_model, uav_canonical_link, T_i, W_i);
        server.set_angular_velocity_cmd(uav_model, uav_canonical_link, TB.segment<3>(1));
      }
      // Step simulation
      server.run_N(1);
    }
    
    // Print progress every second
    if (i % 100 == 0) {
      std::cout << "Progress: " << i / 10 << " / 100 seconds" << std::endl;
    }
  }
  
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
  
  std::cout << "Test completed!" << std::endl;
  std::cout << "Test duration: " << duration.count() / 1000000000.0 << " seconds" << std::endl;
  std::cout << "Average FPS: " << 10000.0 / (duration.count() / 1000000000.0) << std::endl;
  
  // Save results to CSV file
  std::ofstream csv_file("trajectory_results.csv");
  csv_file << "time,x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz,"
           << "des_x,des_y,des_z,des_vx,des_vy,des_vz" << std::endl;
  
  for (size_t i = 0; i < states.size(); i++) {
    double t = i * 0.001;  // 1 ms timestep
    csv_file << t;
    for (int j = 0; j < 13; j++) {
      csv_file << "," << states[i](j);
    }
    for (int j = 0; j < 6; j++) {
      csv_file << "," << desired_states[i](j);
    }
    csv_file << std::endl;
  }
  csv_file.close();
  
  std::cout << "Results saved to trajectory_results.csv" << std::endl;
  
  // Calculate and print tracking errors
  double pos_error_sum = 0.0;
  double vel_error_sum = 0.0;
  for (size_t i = 0; i < states.size(); i++) {
    Eigen::Vector3d pos_error = states[i].segment<3>(0) - desired_states[i].segment<3>(0);
    Eigen::Vector3d vel_error = states[i].segment<3>(3) - desired_states[i].segment<3>(3);
    pos_error_sum += pos_error.norm();
    vel_error_sum += vel_error.norm();
  }
  
  std::cout << "Average position error: " << pos_error_sum / states.size() << " m" << std::endl;
  std::cout << "Average velocity error: " << vel_error_sum / states.size() << " m/s" << std::endl;
}

int main(int argc, char** argv) {
  try {
    const std::string sdf_file = argc > 1 ? argv[1] : "world_simple.sdf";

    // Create NMPC controller
    auto controller = NMPCController::tuned_nmpc();
    
    // Create DRL server
    std::vector<std::string> models = {"quadrotor"};
    DRLServer server("0", sdf_file, models, false);
    
    std::cout << "Server initialized successfully" << std::endl;
    
    // Run trajectory tracking test
    test_trajectory_tracking(server, *controller, "quadrotor", "quadrotor/base_link");
    
    std::cout << "All tests completed successfully!" << std::endl;
    
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  
  return 0;
}
