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
#include <benchmark/benchmark.h>
#include "async_rl_server.hh"
#include "controllers/tuned_gains.hh"

using gz::common::WorkerPool;
using uav_controllers::X;

#define ADD_FREQUENCY(scale) state.counters["Wall-Hz"] = benchmark::Counter(\
        state.iterations()*scale,\
        benchmark::Counter::kIsRate\
    );
static std::atomic<bool> env_set_(false);
static std::vector<std::shared_ptr<DRLServer>> servers_;
static std::atomic<int> server_count_(0);
static int max_servers_ = 15;
static X des_state{1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.9238795, 0.0, 0.3826834, 0.0, 0.0, 0.0, 0.0};
static WorkerPool pool(max_servers_);

/** @brief a helper method to add custom metrics */
inline void AddCustomMetrics(benchmark::State& state){
    std::int64_t iters = state.iterations();

}

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
}

/** @brief  Creates servers*/
static void CreateServers(benchmark::State& state){
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
            auto controller =  std::make_shared<GeometricController>(geometric_controller_gains.at("kp"),
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
            servers_[i]->set_controller( model_name, link_name,controller);
        }
    }

}

/** @brief Benchmarks serial stepping of servers with a controller*/
static void BM_SerialControlStepping(benchmark::State& state) {
    CreateServers(state);
    std::cout<<"Total number of servers: "<<servers_.size()<<std::endl;
    for (auto _ : state) {
        for (int i =0; i< max_servers_; ++i){
            servers_[i]->control_with_rotor_velocity("quadrotor", 
             "quadrotor/base_link", 
             des_state, 
             1);
            servers_[i]->update_control_states();
        }
    }
    ADD_FREQUENCY(max_servers_);
    
}
BENCHMARK(BM_SerialControlStepping)->UseRealTime();

/** @brief Benchmarks void stepping of server */
static void BM_SerialVoidStepping(benchmark::State& state) {
    CreateServers(state);
    for (auto _ : state) {
        for (int i =0; i< max_servers_; ++i){
            servers_[i]->run_N(1);
        }
    }
    ADD_FREQUENCY(max_servers_);
}
BENCHMARK(BM_SerialVoidStepping)->UseRealTime();

/** @brief tests thread based stepping */
static void BM_ParallelControlStepping(benchmark::State& state) {
    CreateServers(state);
    for (auto _ : state) {
        for (int i =0; i< max_servers_; ++i){
            std::function<void()> func;
            func = [=](){
              servers_[i]->control_with_rotor_velocity("quadrotor", 
               "quadrotor/base_link", 
               des_state, 
               1);
              servers_[i]->update_control_states();
            };
            pool.AddWork(func);
        }
        pool.WaitForResults();
    }
    ADD_FREQUENCY(max_servers_);
    
}
BENCHMARK(BM_ParallelControlStepping)->UseRealTime();

/** @brief tests parallel void stepping */
static void BM_ParallelVoidStepping(benchmark::State& state) {
    CreateServers(state);
    for (auto _ : state) {
        for (int i =0; i< max_servers_; ++i){
            std::function<void()> func;
            func = [=](){
              servers_[i]->run_N(1);
            };
            pool.AddWork(func);
        }
        pool.WaitForResults();
    }
    ADD_FREQUENCY(max_servers_);
}
BENCHMARK(BM_ParallelVoidStepping)->UseRealTime();

/** @brief single server stepping with controller*/
static void BM_SingleServerControlStepping(benchmark::State& state) {
    CreateServers(state);
    auto server = servers_[0];
    for (auto _ : state) {
        server->control_with_rotor_velocity("quadrotor", 
         "quadrotor/base_link", 
         des_state, 
         1);
        server->update_control_states();
    }
    ADD_FREQUENCY(1);
}
BENCHMARK(BM_SingleServerControlStepping)->UseRealTime();
/** @brief single server void stepping*/
static void BM_SingleServerVoidStepping(benchmark::State& state) {

    CreateServers(state);
    auto server = servers_[0];
    for (auto _ : state) {
        server->run_N(1);
    }
    ADD_FREQUENCY(1);
}
BENCHMARK(BM_SingleServerVoidStepping)->UseRealTime();

/** @brief update control states benchmark */
static void BM_UpdateControlStates(benchmark::State& state) {
    CreateServers(state);
    auto server = servers_[0];
    for (auto _ : state) {
        server->update_control_states();
    }
    ADD_FREQUENCY(1);
}
BENCHMARK(BM_UpdateControlStates)->UseRealTime();

/** @brief benchmark performance using asyncdrl server */
static std::unique_ptr<async::AsyncDRLServerPool> async_pool_ptr = nullptr;
/** @brief async pool stepping with controller*/
static void BM_AsyncPoolControlStepping(benchmark::State& state) {
    CreateServers(state);
    if (async_pool_ptr == nullptr){
        async_pool_ptr = std::make_unique<async::AsyncDRLServerPool>(servers_);
    }
    auto& async_pool = *async_pool_ptr;
    using UpdateFn = void (DRLServer::*)(std::string, std::string, const Stated&, int);
    UpdateFn fn = &DRLServer::control_with_rotor_velocity;
  
    std::vector<decltype(async_pool.call(std::size_t{},
       fn, std::string{},
      std::string{}, des_state, int{}))> update_futs;
    update_futs.resize(max_servers_);
    std::vector<decltype(async_pool.call(std::size_t{}, &DRLServer::update_control_states))> update_cs_futs;
    update_cs_futs.resize(max_servers_);

    for (auto _ : state) {
      for (int i = 0; i < max_servers_; ++i){
        update_futs[i] = async_pool.call(i, fn,
            "quadrotor",
             "quadrotor/base_link",
             des_state,
            1
        );
    }
      for (int i = 0; i < max_servers_; ++i){
        update_futs[i].get();
        
      }
      for (int i = 0; i < max_servers_; ++i){
        update_cs_futs[i] = async_pool.call(i, &DRLServer::update_control_states);
      }
      for (int i = 0; i < max_servers_; ++i){
        update_cs_futs[i].get();
      }
  }
    ADD_FREQUENCY(max_servers_);
}
BENCHMARK(BM_AsyncPoolControlStepping)->UseRealTime();


/** @brief update control states benchmark */
static void BM_UpdateControlStatesAndAccess(benchmark::State& state) {
    CreateServers(state);
    auto server = servers_[0];
    for (auto _ : state) {
        server->update_control_states();
        const auto & ctrl_state = server->control_states["quadrotor"]["quadrotor/base_link"];
        const auto &state = std::get<0>(ctrl_state);
        const auto &state_dot = std::get<1>(ctrl_state);
    }
    ADD_FREQUENCY(1);
}
BENCHMARK(BM_UpdateControlStatesAndAccess)->UseRealTime();

BENCHMARK_MAIN();
