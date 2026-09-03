// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef PRINT_UTILS_HH_
#define PRINT_UTILS_HH_

#include <cstring>
#include <iostream>

/** printing macro (for debug) */
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define ANSI_RED "\033[31m"
#define ANSI_GREEN "\033[32m"
#define ANSI_BLUE "\033[36m"
#define ANSI_RESET "\033[0m"

// Error prefix macro for formatted error messages with file location
#define ERROR_PREF std::string(ANSI_RED) + "[" + std::string(__FILENAME__) + ":" +     \
                       std::to_string(__LINE__) + "@" + std::string(__func__) + "] " + \
                       std::string(ANSI_RESET) + std::string(ANSI_BLUE)

#define INFO_PREF std::string(ANSI_GREEN) + "[" + std::string(__FILENAME__) + ":" +   \
                      std::to_string(__LINE__) + "@" + std::string(__func__) + "] " + \
                      std::string(ANSI_RESET) + std::string(ANSI_BLUE)
/**
 * @brief Print a single argument to stdout with color reset
 * @tparam Arg Type of the argument to print
 * @param arg The argument to print
 */
template <typename Arg>
inline void _print(Arg arg)
{
    std::cout << arg << std::string(ANSI_RESET) << "\n";
}

/**
 * @brief Print multiple arguments to stdout (variadic template)
 * @tparam Arg Type of the first argument
 * @tparam Args Types of remaining arguments
 * @param arg First argument to print
 * @param args Remaining arguments to print
 */
template <typename Arg, typename... Args>
inline void _print(Arg arg, Args... args)
{
    std::cout << arg;
    _print(args...);
}

/**
 * @brief Print error message with formatting
 * @tparam Args Types of arguments to print
 * @param args Arguments to print as error message
 */
template <typename... Args>
inline void _print_err(Args... args)
{
    _print(args...);
}

/// Macro for printing formatted error messages with file location
#define print_err(...) _print_err(ERROR_PREF, __VA_ARGS__)

// Macro for printing formatted info messages with file location
#define print_info(...) _print(INFO_PREF, __VA_ARGS__)

#endif