// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#include "rl_server.hh"
#include "env_specs/gazebo_envpool_spec.hh"
#include <iostream>
#include "controllers/uav_controllers.hh"
#include "controllers/geometric_controller.hh"
#include <chrono>
#include "print_utils.hh"
#include <atomic>
#include <gz/common/WorkerPool.hh>
#include "async_rl_server.hh"
#include <chrono>
#include "controllers/tuned_gains.hh"

using gz::common::WorkerPool;
using uav_controllers::X;
static std::atomic<bool> env_set_(false);
static std::vector<std::shared_ptr<DRLServer>> servers_;
static std::atomic<int> server_count_(0);
static int max_servers_ = 1;
static X des_state{1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.9238795, 0.0, 0.0, 0.3826834, 0.0, 0.0, 0.0};
static WorkerPool pool(max_servers_);
static int max_steps_ = 1000;

/** @brief Sets environment variables for Gazebo simulation paths */
static inline void SetupEnv() noexcept{
  const std::string resources_path = GZDRL_TEST_RESOURCE_DIR;
  const std::string plugins_path = GZDRL_TEST_PLUGIN_DIR;

  if (setenv("GZ_SIM_RESOURCE_PATH", resources_path.c_str(), 1) != 0)
    print_err("failed to set GZ_SIM_RESOURCE_PATH");
  if (setenv("GZ_SIM_SYSTEM_PLUGIN_PATH", plugins_path.c_str(), 1) != 0)
    print_err("failed to set GZ_SIM_SYSTEM_PLUGIN_PATH");
  const char* max_servers_env = std::getenv("MAX_SERVERS");;
  if (max_servers_env != NULL){
    max_servers_ = std::atoi(max_servers_env);
  }
  print_info("Max servers set to: ", max_servers_);
  const char* max_steps_env = std::getenv("MAX_STEPS");;
  if (max_steps_env != NULL){
    max_steps_ = std::atoi(max_steps_env);
  }
  print_info("Max steps set to: ", max_steps_);
}

/** @brief  Creates servers*/
static void CreateServers(){
    if (!env_set_.load()){
        SetupEnv();
        env_set_.store(true);
    }
    if (servers_.empty()){
        servers_.resize(max_servers_);
        const std::string sdf_file = "world_simple.sdf";
        const std::string model_name = "quadrotor";
        const std::string link_name = "quadrotor/base_link";
        const std::vector<std::string> model_names{"quadrotor"};

        const auto &geometric_controller_gains = GAIN_MAP.at("qdrone2").at("geometric_controller");
        const auto& controller_params = PARAMETER_MAP.at("qdrone2").at("geometric_controller");
        for (int i =0; i< max_servers_; ++i){
            auto controller = std::make_shared<GeometricController>(geometric_controller_gains.at("kp"),
                                                  geometric_controller_gains.at("kd"),
                                                  geometric_controller_gains.at("kp_att"),
                                                  geometric_controller_gains.at("kd_att"),
                                                  controller_params.max_accel,
                                                  controller_params.gravity_vec,
                                                  controller_params.mass,
                                                  controller_params.inertia);
            servers_[i] = std::make_shared<DRLServer>(std::to_string(i), 
                  sdf_file, 
                  model_names,
                false);
            servers_[i]->run_N(5);
            servers_[i]->set_controller( model_name, link_name,controller);
        }
    }

}

struct Timer{
    Timer(const std::string& name_): name(name_){
        start = std::chrono::high_resolution_clock::now();
    }
    ~Timer(){
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::cout<<name<<" took "<<duration<<" ns"<<std::endl;
        dt = duration;
    }
    inline static double dt = 0.0;
    std::string name;
    std::chrono::high_resolution_clock::time_point start;

};
int main(int argc, char** argv){
    CreateServers();
    std::unique_ptr<Timer> timer;
    timer = std::make_unique<Timer>("Total Time");
    for (int i =0; i< max_servers_; ++i){
            std::function<void()> func;
            func = [=](){
              for (int step = 0; step < max_steps_; ++step) {
                servers_[i]->control_with_rotor_velocity("quadrotor", 
                 "quadrotor/base_link", 
                 des_state, 
                 1);
                servers_[i]->update_control_states();
              }
            };
            pool.AddWork(func);
        }
        pool.WaitForResults();
    timer.reset();
    double dt = Timer::dt*1e-9;
    double rate = max_servers_*max_steps_/dt;
    double error = 0.0;
    Eigen::Vector3d final_pos;
    Eigen::Vector4d final_ori;
    for (const auto& server : servers_){
        error += (std::get<0>(server->control_states["quadrotor"]["quadrotor/base_link"])-des_state).norm();
        final_pos = std::get<0>(server->control_states["quadrotor"]["quadrotor/base_link"]).head<3>();
        final_ori = std::get<0>(server->control_states["quadrotor"]["quadrotor/base_link"]).segment<4>(6);
    }
    print_info("Desired state used for benchmark: ", "(",des_state.head<3>().transpose(), 
    ", ", des_state.segment<4>(6).transpose(), ")");
    print_info("Average final state error across all servers: ", error/static_cast<double>(max_servers_));
    print_info("Average final position across all servers: ", "(", final_pos.transpose()/static_cast<double>(max_servers_), ")");
    print_info("Average final orientation across all servers: ", "(", final_ori.transpose()/static_cast<double>(max_servers_), ")");
    print_info("Total time taken: ", dt, " s");
    print_info("Average time per server: ", dt/max_servers_, " s");
    print_info("Total step rate / s: ", rate, " Hz");
    print_info("Average step rate per server / s: ", rate/max_servers_, " Hz");
    
    return 0;
}
