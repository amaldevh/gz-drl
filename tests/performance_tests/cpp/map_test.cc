// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#include <string>
#include <string_view>
#include <vector>
#include <tuple>
#include <iostream>
#include <benchmark/benchmark.h>
#include <map>
#include <Eigen/Dense>
#include <tsl/robin_map.h>
#include <algorithm>

#define ADD_FREQUENCY state.counters["Wall-Hz"] = benchmark::Counter(\
        state.iterations(),\
        benchmark::Counter::kIsRate\
    );

static void BM_StringCreation(benchmark::State& state) {
  for (auto _ : state)
    std::string empty_string;
}
// Register the function as a benchmark
BENCHMARK(BM_StringCreation)->UseRealTime();

// Define another benchmark
static void BM_StringCopy(benchmark::State& state) {
  std::string x = "hello";
  for (auto _ : state)
    std::string copy(x);
}
BENCHMARK(BM_StringCopy)->UseRealTime();

using State = Eigen::Matrix<float, 13, 1>;
using StateTup = std::tuple<State, State>;

using MapType = std::unordered_map<std::string, StateTup>;
//using MapType = tsl::robin_map
                  //<std::string, StateTup, std::hash<std::string>, 
                //    std::equal_to<std::string>,
                //    std::allocator<std::pair<std::string, StateTup>>,
                //    true>;
using StateMap = MapType;

static StateMap servers_states;
static int key_idx = 0;
static std::vector<std::string> key_map;
static std::unordered_map<int, StateTup> int_state_map;

/** @brief Benchmark for setting a key in an unordered map */
static void BM_UnorderedMapSet(benchmark::State& state) {
    std::string key = "quadrotor/base_link";
    State state1 = State::Random();
    State state2 = State::Random();
    for (auto _ : state) {
        servers_states[key + std::to_string(key_idx++)] = std::make_tuple(state1, state2);
    }
    ADD_FREQUENCY;
}
BENCHMARK(BM_UnorderedMapSet)->UseRealTime();
/** @brief Benchmark for getting a key in an unordered map */
static void BM_UnorderedMapGet(benchmark::State& state) {
    std::string key = "quadrotor/base_link";
    State state1 = State::Random();
    State state2 = State::Random();
    // Prepopulate the map
    for (int i = 0; i < 1000; ++i) {
        servers_states[key + std::to_string(i)] = std::make_tuple(state1, state2);
    }
    key_idx = 0;
    for (auto _ : state) {
        auto& val = servers_states[key + std::to_string(key_idx++)];
        const auto & vals1 = std::get<0>(val);
        const auto & vals2 = std::get<1>(val);
    }
    ADD_FREQUENCY;
}
BENCHMARK(BM_UnorderedMapGet)->UseRealTime();

/** @brief benchmark for using int map, we find the string index
on a one-to-one map of strings to ints */
static void BM_IntMapGet(benchmark::State& state) {
    std::string key = "quadrotor/base_link";
    State state1 = State::Random();
    State state2 = State::Random();
    // Prepopulate the map
    for (int i = 0; i < 1000; ++i) {
        key_map.push_back(key + std::to_string(i));
        int_state_map[i] = std::make_tuple(state1, state2);
    }
    key_idx = 999;
    for (auto _ : state) {
        int key_int = key_idx--;
        // int idx = std::find(key_map.begin(), key_map.end(), key + std::to_string(key_int)) - key_map.begin();
        auto& val = int_state_map[key_int];
        const auto & vals1 =  std::get<0>(val);
        const auto & vals2 =  std::get<1>(val);
    }
    ADD_FREQUENCY;
}
BENCHMARK(BM_IntMapGet)->UseRealTime();

static std::unordered_map<uint32_t, StateTup> uint_state_map;

uint32_t hash_string(const char * s)
{
    uint32_t hash = 0;

    for(; *s; ++s)
    {
        hash += *s;
        hash += (hash << 10);
        hash ^= (hash >> 6);
    }

    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);

    return hash;
}

/** @brief benchmark using custom hash */
static void BM_IntMapGetCustomHash(benchmark::State& state) {
    std::string key = "quadrotor/base_link";
    State state1 = State::Random();
    State state2 = State::Random();
    // Prepopulate the map
    for (int i = 0; i < 1000; ++i) {
        std::string key_ = (key + std::to_string(i));
        
        uint_state_map[hash_string(key_.c_str())] = std::make_tuple(state1, state2);
    }
    key_idx = 999;
    for (auto _ : state) {
        int key_int = key_idx--;
        // int idx = std::find(key_map.begin(), key_map.end(), key + std::to_string(key_int)) - key_map.begin();
        auto& val = uint_state_map[hash_string((key + std::to_string(key_int)).c_str())];
        const auto & vals1 =  std::get<0>(val);
        const auto & vals2 =  std::get<1>(val);
    }
    ADD_FREQUENCY;
}
BENCHMARK(BM_IntMapGetCustomHash)->UseRealTime();

BENCHMARK_MAIN();