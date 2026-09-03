# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

cmake_minimum_required(VERSION 3.10)

# Tests
include_directories(robin-map/include)

find_package(benchmark REQUIRED)
find_package(glog REQUIRED)
link_libraries(benchmark::benchmark
    benchmark::benchmark_main)

add_executable(map_test 
    performance_tests/cpp/map_test.cc)

add_executable(perf_test 
    performance_tests/cpp/test_rl_perf.cc)
target_link_libraries(perf_test PRIVATE
    rl_server
    glog::glog
    # System libcurl provides curl_*@CURL_OPENSSL_4 required by the
    # transitive gz-fuel_tools/gdal/hdf5 deps; conda's libcurl lacks the
    # versioned symbols, so link it explicitly to win over conda's -L path.
    /usr/lib/x86_64-linux-gnu/libcurl.so.4)

add_executable(raw_perf_test 
    performance_tests/cpp/raw_serv_step_perf.cc)
target_link_libraries(raw_perf_test PRIVATE
    rl_server
    glog::glog
    /usr/lib/x86_64-linux-gnu/libcurl.so.4)

foreach(performance_target perf_test raw_perf_test)
    target_compile_definitions(${performance_target} PRIVATE
        GZDRL_TEST_RESOURCE_DIR="${CMAKE_SOURCE_DIR}/resources/sdf"
        GZDRL_TEST_PLUGIN_DIR="$<TARGET_FILE_DIR:rotor_plugin>")
endforeach()
