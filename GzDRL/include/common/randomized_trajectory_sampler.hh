// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef COMMON_RANDOMIZED_TRAJECTORY_SAMPLER_HH
#define COMMON_RANDOMIZED_TRAJECTORY_SAMPLER_HH

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/waypoint_generator.hh"

/** Samples one of four randomized, feasibility-checked training trajectories.
 *
 * Height convention:
 *   - nominal height / vertical offset: 0.8 m
 *   - vertical amplitude for YZ and 3D Lissajous: U[0.3, 0.4] m
 *   - XY Lissajous and circle remain at z = 0.8 m
 *
 * The YZ curve is checked at 1.0 m/s. With only 0.3--0.4 m of vertical
 * amplitude, a closed YZ Lissajous requires substantially more curvature
 * authority at 1.5 m/s than the conservative default limits permit.
 */
class RandomizedTrajectorySampler {
public:
    enum class TrajectoryType {
        XYLissajous = 0,
        YZLissajous = 1,
        Lissajous3D = 2,
        Circle = 3
    };

    struct Config {
        float nominal_height = 0.8f;
        float vertical_amplitude_min = 0.3f;
        float vertical_amplitude_max = 0.4f;
        float min_waypoint_distance = 0.65f;
        float max_waypoint_distance = 0.65f;
        int max_parameter_attempts = 64;

        // Conservative YZ-specific feasibility envelope. The other three
        // trajectories retain the caller-provided 1.0--1.5 m/s envelope.
        double yz_checked_speed = 1.0;
        double yz_max_normal_acceleration = 4.5;
    };

    struct Sample {
        std::vector<WaypointGenerator::Point> waypoints;
        TrajectoryType type = TrajectoryType::Circle;
        std::string description;
        double checked_max_speed = 0.0;
    };

    RandomizedTrajectorySampler() : rng_(std::random_device{}()) {}
    explicit RandomizedTrajectorySampler(std::uint32_t seed) : rng_(seed) {}

    void Seed(std::uint32_t seed) { rng_.seed(seed); }

    static const char* TypeName(TrajectoryType type) {
        switch (type) {
            case TrajectoryType::XYLissajous:
                return "XY planar Lissajous";
            case TrajectoryType::YZLissajous:
                return "YZ planar Lissajous";
            case TrajectoryType::Lissajous3D:
                return "3D Lissajous";
            case TrajectoryType::Circle:
                return "XY circle";
        }
        return "unknown";
    }

    Sample GenerateRandom(
        WaypointGenerator& generator,
        const WaypointGenerator::FeasibilityLimits& base_limits) {
        return GenerateRandom(generator, base_limits, Config{});
    }

    Sample GenerateRandom(
        WaypointGenerator& generator,
        const WaypointGenerator::FeasibilityLimits& base_limits,
        const Config& config) {
        std::uniform_int_distribution<int> type_distribution(0, 3);
        return GenerateOfType(
            static_cast<TrajectoryType>(type_distribution(rng_)),
            generator,
            base_limits,
            config);
    }

    Sample GenerateOfType(
        TrajectoryType type,
        WaypointGenerator& generator,
        const WaypointGenerator::FeasibilityLimits& base_limits) {
        return GenerateOfType(type, generator, base_limits, Config{});
    }

    Sample GenerateOfType(
        TrajectoryType type,
        WaypointGenerator& generator,
        const WaypointGenerator::FeasibilityLimits& base_limits,
        const Config& config) {
        ValidateConfig(config);

        std::string last_error;
        for (int attempt = 0; attempt < config.max_parameter_attempts; ++attempt) {
            try {
                Sample sample = GenerateCandidate(type, generator, base_limits, config);
                if (sample.waypoints.size() < 4U) {
                    throw std::runtime_error(
                        "Generated fewer than four waypoints.");
                }
                return sample;
            } catch (const std::exception& error) {
                last_error = error.what();
            }
        }

        // Deterministic fallbacks make training robust to an unlucky run of
        // rejected randomized parameters while preserving the selected type.
        try {
            return GenerateFallback(type, generator, base_limits, config);
        } catch (const std::exception& fallback_error) {
            throw std::runtime_error(
                std::string("Unable to generate ") + TypeName(type) +
                " after randomized retries. Last randomized error: " +
                last_error + ". Fallback error: " + fallback_error.what());
        }
    }

private:
    std::mt19937 rng_;

    static void ValidateConfig(const Config& config) {
        if (!(config.nominal_height > 0.0f)) {
            throw std::invalid_argument("nominal_height must be positive.");
        }
        if (!(config.vertical_amplitude_min > 0.0f) ||
            config.vertical_amplitude_min > config.vertical_amplitude_max) {
            throw std::invalid_argument(
                "Require 0 < vertical_amplitude_min <= vertical_amplitude_max.");
        }
        if (config.nominal_height - config.vertical_amplitude_max < 0.0f) {
            throw std::invalid_argument(
                "The minimum trajectory height must not be below zero.");
        }
        if (!(config.min_waypoint_distance > 0.0f) ||
            config.min_waypoint_distance > config.max_waypoint_distance) {
            throw std::invalid_argument(
                "Require 0 < min_waypoint_distance <= max_waypoint_distance.");
        }
        if (config.max_parameter_attempts <= 0) {
            throw std::invalid_argument(
                "max_parameter_attempts must be positive.");
        }
        if (!(config.yz_checked_speed > 0.0) ||
            !(config.yz_max_normal_acceleration > 0.0)) {
            throw std::invalid_argument(
                "YZ feasibility speed and acceleration must be positive.");
        }
    }

    float Uniform(float lower, float upper) {
        std::uniform_real_distribution<float> distribution(lower, upper);
        return distribution(rng_);
    }

    static float TwoPi() {
        return static_cast<float>(2.0 * WaypointGenerator::PI());
    }

    static WaypointGenerator::FeasibilityLimits YZLimits(
        const WaypointGenerator::FeasibilityLimits& base_limits,
        const Config& config) {
        auto limits = base_limits;
        limits.min_speed = std::min(limits.min_speed, config.yz_checked_speed);
        limits.max_speed = config.yz_checked_speed;
        limits.max_normal_acceleration = std::max(
            limits.max_normal_acceleration,
            config.yz_max_normal_acceleration);
        return limits;
    }

    Sample GenerateCandidate(
        TrajectoryType type,
        WaypointGenerator& generator,
        const WaypointGenerator::FeasibilityLimits& base_limits,
        const Config& config) {
        const float phase = Uniform(0.0f, TwoPi());
        const float min_dist = config.min_waypoint_distance;
        const float max_dist = config.max_waypoint_distance;

        switch (type) {
            case TrajectoryType::XYLissajous: {
                const float mean_amplitude = Uniform(1.35f, 1.65f);
                const float aspect = Uniform(0.94f, 1.06f);
                const float A = mean_amplitude * std::sqrt(aspect);
                const float B = mean_amplitude / std::sqrt(aspect);
                const float phase_difference = Uniform(
                    static_cast<float>(70.0 * WaypointGenerator::PI() / 180.0),
                    static_cast<float>(110.0 * WaypointGenerator::PI() / 180.0));
                auto points = generator.GenerateLissajousWaypoints(
                    A, B,
                    1.0f, 1.0f,
                    phase, phase + phase_difference,
                    min_dist, max_dist,
                    config.nominal_height,
                    base_limits);
                return MakeSample(
                    type,
                    std::move(points),
                    base_limits.max_speed,
                    "A=" + Format(A) + ", B=" + Format(B) +
                        ", z=" + Format(config.nominal_height));
            }

            case TrajectoryType::YZLissajous: {
                const float vertical_amplitude = Uniform(
                    config.vertical_amplitude_min,
                    config.vertical_amplitude_max);
                // Near the low-curvature region for y=A sin(u),
                // z=z0+C sin(2u), while respecting |y| < 2 m.
                const float y_amplitude =
                    4.07f * vertical_amplitude * Uniform(0.97f, 1.03f);
                const float x_offset = Uniform(-0.15f, 0.15f);
                const auto limits = YZLimits(base_limits, config);
                auto points = generator.GenerateYZLissajousWaypoints(
                    y_amplitude,
                    vertical_amplitude,
                    1.0f, 2.0f,
                    phase, 2.0f * phase,
                    min_dist, max_dist,
                    x_offset,
                    config.nominal_height,
                    limits);
                return MakeSample(
                    type,
                    std::move(points),
                    limits.max_speed,
                    "Ay=" + Format(y_amplitude) +
                        ", Az=" + Format(vertical_amplitude) +
                        ", z0=" + Format(config.nominal_height));
            }

            case TrajectoryType::Lissajous3D: {
                const float mean_amplitude = Uniform(1.68f, 1.78f);
                const float aspect = Uniform(0.96f, 1.04f);
                const float A = mean_amplitude * std::sqrt(aspect);
                const float B = mean_amplitude / std::sqrt(aspect);
                const float C = Uniform(
                    config.vertical_amplitude_min,
                    config.vertical_amplitude_max);
                const float xy_phase_difference = Uniform(
                    static_cast<float>(82.0 * WaypointGenerator::PI() / 180.0),
                    static_cast<float>(98.0 * WaypointGenerator::PI() / 180.0));
                const float z_phase = Uniform(0.0f, TwoPi());
                auto points = generator.GenerateLissajousWaypoints(
                    A, B, C,
                    1.0f, 1.0f, 2.0f,
                    phase,
                    phase + xy_phase_difference,
                    z_phase,
                    min_dist, max_dist,
                    config.nominal_height,
                    base_limits);
                return MakeSample(
                    type,
                    std::move(points),
                    base_limits.max_speed,
                    "A=" + Format(A) + ", B=" + Format(B) +
                        ", C=" + Format(C) +
                        ", z0=" + Format(config.nominal_height));
            }

            case TrajectoryType::Circle: {
                const float radius = Uniform(1.30f, 1.75f);
                const float center_limit = std::max(0.0f, 1.90f - radius);
                const float center_x = Uniform(-center_limit, center_limit);
                const float center_y = Uniform(-center_limit, center_limit);
                auto points = generator.GenerateCircleWaypoints(
                    radius,
                    center_x,
                    center_y,
                    min_dist,
                    max_dist,
                    config.nominal_height,
                    base_limits);
                return MakeSample(
                    type,
                    std::move(points),
                    base_limits.max_speed,
                    "r=" + Format(radius) +
                        ", center=(" + Format(center_x) + "," +
                        Format(center_y) + ")" +
                        ", z=" + Format(config.nominal_height));
            }
        }

        throw std::logic_error("Unhandled trajectory type.");
    }

    Sample GenerateFallback(
        TrajectoryType type,
        WaypointGenerator& generator,
        const WaypointGenerator::FeasibilityLimits& base_limits,
        const Config& config) {
        const float min_dist = config.min_waypoint_distance;
        const float max_dist = config.max_waypoint_distance;
        const float half_pi = static_cast<float>(WaypointGenerator::PI() / 2.0);

        switch (type) {
            case TrajectoryType::XYLissajous:
                return MakeSample(
                    type,
                    generator.GenerateLissajousWaypoints(
                        1.5f, 1.5f, 1.0f, 1.0f,
                        0.0f, half_pi,
                        min_dist, max_dist,
                        config.nominal_height,
                        base_limits),
                    base_limits.max_speed,
                    "fallback A=B=1.5, z=" + Format(config.nominal_height));

            case TrajectoryType::YZLissajous: {
                const auto limits = YZLimits(base_limits, config);
                return MakeSample(
                    type,
                    generator.GenerateYZLissajousWaypoints(
                        1.63f, 0.4f,
                        1.0f, 2.0f,
                        0.0f, 0.0f,
                        min_dist, max_dist,
                        0.0f,
                        config.nominal_height,
                        limits),
                    limits.max_speed,
                    "fallback Ay=1.63, Az=0.4, z0=" +
                        Format(config.nominal_height));
            }

            case TrajectoryType::Lissajous3D:
                return MakeSample(
                    type,
                    generator.GenerateLissajousWaypoints(
                        1.75f, 1.75f, 0.35f,
                        1.0f, 1.0f, 2.0f,
                        0.0f, half_pi, 0.0f,
                        min_dist, max_dist,
                        config.nominal_height,
                        base_limits),
                    base_limits.max_speed,
                    "fallback A=B=1.75, C=0.35, z0=" +
                        Format(config.nominal_height));

            case TrajectoryType::Circle:
                return MakeSample(
                    type,
                    generator.GenerateCircleWaypoints(
                        1.5f, 0.0f, 0.0f,
                        min_dist, max_dist,
                        config.nominal_height,
                        base_limits),
                    base_limits.max_speed,
                    "fallback r=1.5, z=" + Format(config.nominal_height));
        }

        throw std::logic_error("Unhandled trajectory type.");
    }

    static Sample MakeSample(
        TrajectoryType type,
        std::vector<WaypointGenerator::Point> points,
        double checked_max_speed,
        const std::string& parameters) {
        Sample sample;
        sample.type = type;
        sample.waypoints = std::move(points);
        sample.checked_max_speed = checked_max_speed;
        sample.description = std::string(TypeName(type)) + ": " + parameters;
        return sample;
    }

    static std::string Format(float value) {
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(3) << value;
        return stream.str();
    }
};

#endif  // COMMON_RANDOMIZED_TRAJECTORY_SAMPLER_HH
