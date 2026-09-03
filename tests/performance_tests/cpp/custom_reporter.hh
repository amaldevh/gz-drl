// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef CUSTOM_REPORTER_HH
#define CUSTOM_REPORTER_HH_

#include <benchmark/benchmark.h>
#include <iostream>

// ANSI color codes
#define ANSI_RESET   "\033[0m"
#define ANSI_RED     "\033[31m"
#define ANSI_GREEN   "\033[32m"
#define ANSI_YELLOW  "\033[33m"
#define ANSI_BLUE    "\033[34m"
#define ANSI_MAGENTA "\033[35m"
#define ANSI_CYAN    "\033[36m"
#define ANSI_WHITE   "\033[37m"

// Bold colors
#define ANSI_BOLD_RED     "\033[1;31m"
#define ANSI_BOLD_GREEN   "\033[1;32m"
#define ANSI_BOLD_YELLOW  "\033[1;33m"

// Background colors
#define ANSI_BG_RED    "\033[41m"
#define ANSI_BG_GREEN  "\033[42m"


class MyReporter : public benchmark::ConsoleReporter {
public:
    bool ReportContext(const Context& context) override {
        return true;
    }
    
    void ReportRuns(const std::vector<Run>& reports) override {
        benchmark::ConsoleReporter::ReportRuns(reports);
        for (const auto& run : reports) {
            std::cout << ANSI_GREEN << run.benchmark_name()<< " custom report" << ANSI_RESET << "\n";
            std::cout << ANSI_BLUE<<"Real time: " << run.real_accumulated_time << " seconds\n";
            std::cout << "CPU time: " << run.cpu_accumulated_time << " seconds\n";
            std::cout << "Iterations: " << run.iterations<<ANSI_RESET << "\n";
        }
    }
};

#define CUSTOM_BENCHMARK_MAIN int main(int argc, char** argv) {\
    MyReporter reporter;\
    benchmark::Initialize(&argc, argv);\
    benchmark::RunSpecifiedBenchmarks(&reporter);\
    return 0;\
}

#endif