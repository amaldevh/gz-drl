// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef COMMON_WAYPOINT_GENERATOR_HH
#define COMMON_WAYPOINT_GENERATOR_HH

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>


/** @class WaypointGenerator
 * Generates curvature-aware waypoints along circular and Lissajous curves.
 *
 * The feasibility checks assume constant-speed traversal of the underlying
 * smooth parametric curve.  The returned points are still geometric waypoints;
 * use a smooth trajectory interpolator / time-parameterizer downstream.
 */
class WaypointGenerator {
public:
    using Point = std::vector<float>;  // [x, y, z, nx, ny, nz]

    static constexpr double PI() { return 3.14159265358979323846; }
    static constexpr double EPSILON() { return 1e-9; }

    struct FeasibilityLimits {
        // The whole curve is checked at max_speed. Therefore it is also valid
        // at every lower constant speed, including min_speed.
        double min_speed = 1.0;                  // m/s
        double max_speed = 1.5;                  // m/s

        // Translational normal-acceleration budget. 2 m/s^2 matches a common
        // conservative outer-loop acceleration limit; replace with your own.
        double max_normal_acceleration = 2.0;    // m/s^2

        // Optional quadrotor-specific checks based on
        //   specific_thrust = desired_acceleration + gravity * e_z.
        double gravity = 9.81;                   // m/s^2
        double max_specific_thrust = 2.0 * 9.81; // N/kg == m/s^2
        double max_tilt_rad = 35.0 * PI() / 180.0;

        // Reserve model/controller authority. The usable limits are multiplied
        // by this number. Use 1.0 in ideal simulation, <1.0 on hardware.
        double safety_factor = 0.90;

        // Limit tangent rotation represented by one waypoint interval. This
        // may intentionally reduce spacing below min_dist in high curvature.
        double max_tangent_change_rad = 10.0 * PI() / 180.0;

        // Numerical settings for Lissajous validation and arc-length stepping.
        int feasibility_samples = 20000;
        double integration_arc_step = 0.005;     // m
        double max_parameter_step = 0.001;       // curve-parameter units
        double min_parameter_speed = 1e-6;       // detects cusps/singularities

        // Optional full 3D tangent-heading rate check. Since |dT/dt| = v*kappa,
        // this remains well-defined even for vertical tangents.
        double max_heading_rate_rad_s =
            std::numeric_limits<double>::infinity();

        // Optional legacy XY-projected yaw-rate check. Infinity disables it.
        double max_yaw_rate_rad_s = std::numeric_limits<double>::infinity();
    };

private:
    struct Vec3 {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;

        Vec3() = default;
        Vec3(double x_value, double y_value, double z_value)
            : x(x_value), y(y_value), z(z_value) {}

        Vec3 operator+(const Vec3& other) const {
            return {x + other.x, y + other.y, z + other.z};
        }
        Vec3 operator-(const Vec3& other) const {
            return {x - other.x, y - other.y, z - other.z};
        }
        Vec3 operator*(double scalar) const {
            return {x * scalar, y * scalar, z * scalar};
        }
        Vec3 operator/(double scalar) const {
            return {x / scalar, y / scalar, z / scalar};
        }
        double dot(const Vec3& other) const {
            return x * other.x + y * other.y + z * other.z;
        }
        double squaredNorm() const { return dot(*this); }
        double norm() const { return std::sqrt(squaredNorm()); }
        double xyNorm() const { return std::sqrt(x * x + y * y); }
    };

    struct CurveState {
        Vec3 position;
        Vec3 first_derivative;
        Vec3 second_derivative;
    };

    struct FeasibilityMetrics {
        double curvature = 0.0;
        double normal_acceleration = 0.0;
        double specific_thrust = 0.0;
        double tilt_rad = 0.0;
        double heading_rate_rad_s = 0.0;
        double yaw_rate_rad_s = 0.0;
    };

    std::mt19937 rng_;

    static bool IsFiniteLimit(double value) {
        return std::isfinite(value);
    }

    static void ValidateFrequency(double frequency, const char* name) {
        if (!(frequency > 0.0)) {
            throw std::invalid_argument(std::string(name) +
                                        " must be positive.");
        }
        if (std::abs(frequency - std::round(frequency)) > 1e-5) {
            throw std::invalid_argument(
                std::string(name) +
                " must be an integer frequency multiplier so the curve has "
                "a well-defined closed period.");
        }
    }

    static void ValidateSpacing(double min_dist, double max_dist) {
        if (!(min_dist > 0.0) || !(max_dist > 0.0)) {
            throw std::invalid_argument("min_dist and max_dist must be positive.");
        }
        if (min_dist > max_dist) {
            throw std::invalid_argument("min_dist must be <= max_dist.");
        }
    }

    static void ValidateLimits(const FeasibilityLimits& limits) {
        if (!(limits.min_speed > 0.0) || !(limits.max_speed > 0.0) ||
            limits.min_speed > limits.max_speed) {
            throw std::invalid_argument(
                "Require 0 < min_speed <= max_speed in FeasibilityLimits.");
        }
        if (!(limits.safety_factor > 0.0) || limits.safety_factor > 1.0) {
            throw std::invalid_argument("safety_factor must lie in (0, 1].");
        }
        if (!(limits.gravity > 0.0)) {
            throw std::invalid_argument("gravity must be positive.");
        }
        if (IsFiniteLimit(limits.max_normal_acceleration) &&
            !(limits.max_normal_acceleration > 0.0)) {
            throw std::invalid_argument(
                "max_normal_acceleration must be positive or infinity.");
        }
        if (IsFiniteLimit(limits.max_specific_thrust) &&
            !(limits.max_specific_thrust > limits.gravity)) {
            throw std::invalid_argument(
                "max_specific_thrust must exceed gravity or be infinity.");
        }
        if (IsFiniteLimit(limits.max_tilt_rad) &&
            (!(limits.max_tilt_rad > 0.0) || limits.max_tilt_rad >= PI() / 2.0)) {
            throw std::invalid_argument(
                "max_tilt_rad must lie in (0, pi/2) or be infinity.");
        }
        if (IsFiniteLimit(limits.max_heading_rate_rad_s) &&
            !(limits.max_heading_rate_rad_s > 0.0)) {
            throw std::invalid_argument(
                "max_heading_rate_rad_s must be positive or infinity.");
        }
        if (IsFiniteLimit(limits.max_yaw_rate_rad_s) &&
            !(limits.max_yaw_rate_rad_s > 0.0)) {
            throw std::invalid_argument(
                "max_yaw_rate_rad_s must be positive or infinity.");
        }
        if (!(limits.max_tangent_change_rad > 0.0) ||
            limits.max_tangent_change_rad >= PI()) {
            throw std::invalid_argument(
                "max_tangent_change_rad must lie in (0, pi).");
        }
        if (limits.feasibility_samples < 1000) {
            throw std::invalid_argument("feasibility_samples must be >= 1000.");
        }
        if (!(limits.integration_arc_step > 0.0) ||
            !(limits.max_parameter_step > 0.0) ||
            !(limits.min_parameter_speed > 0.0)) {
            throw std::invalid_argument(
                "Numerical integration settings must be positive.");
        }
    }

    float GetRandomDistance(float min_dist, float max_dist) {
        std::uniform_real_distribution<float> dist(min_dist, max_dist);
        return dist(rng_);
    }

    static Vec3 CurvatureVector(const CurveState& state,
                                           double min_parameter_speed) {
        const double q2 = state.first_derivative.squaredNorm();
        const double q_min2 = min_parameter_speed * min_parameter_speed;
        if (q2 <= q_min2) {
            throw std::runtime_error(
                "Curve contains a cusp/stationary parameter point: |r'(u)| is "
                "too small for non-zero-speed traversal.");
        }

        // dT/ds for an arbitrary parameter u, where T = r'/|r'|.
        return state.second_derivative / q2 -
               state.first_derivative *
                   (state.first_derivative.dot(state.second_derivative) /
                    (q2 * q2));
    }

    static FeasibilityMetrics ComputeMetrics(
        const CurveState& state,
        const FeasibilityLimits& limits) {
        const Vec3 curvature_vector =
            CurvatureVector(state, limits.min_parameter_speed);

        FeasibilityMetrics metrics;
        metrics.curvature = curvature_vector.norm();

        const Vec3 desired_acceleration =
            curvature_vector * (limits.max_speed * limits.max_speed);
        metrics.normal_acceleration = desired_acceleration.norm();
        metrics.heading_rate_rad_s =
            limits.max_speed * metrics.curvature;

        const Vec3 specific_thrust_vector =
            desired_acceleration +
            Vec3(0.0, 0.0, limits.gravity);
        metrics.specific_thrust = specific_thrust_vector.norm();

        if (metrics.specific_thrust <= EPSILON() ||
            specific_thrust_vector.z <= 0.0) {
            metrics.tilt_rad = std::numeric_limits<double>::infinity();
        } else {
            metrics.tilt_rad = std::atan2(
                specific_thrust_vector.xyNorm(),
                specific_thrust_vector.z);
        }

        const double dx = state.first_derivative.x;
        const double dy = state.first_derivative.y;
        const double ddx = state.second_derivative.x;
        const double ddy = state.second_derivative.y;
        const double xy_speed2 = dx * dx + dy * dy;
        const double parameter_speed = state.first_derivative.norm();
        if (xy_speed2 > limits.min_parameter_speed * limits.min_parameter_speed &&
            parameter_speed > limits.min_parameter_speed) {
            const double yaw_per_parameter =
                (dx * ddy - dy * ddx) / xy_speed2;
            const double parameter_rate = limits.max_speed / parameter_speed;
            metrics.yaw_rate_rad_s =
                std::abs(yaw_per_parameter * parameter_rate);
        }

        return metrics;
    }

    static bool MetricsAreFeasible(const FeasibilityMetrics& metrics,
                                   const FeasibilityLimits& limits,
                                   std::string* reason = nullptr) {
        const double factor = limits.safety_factor;

        if (IsFiniteLimit(limits.max_normal_acceleration) &&
            metrics.normal_acceleration >
                factor * limits.max_normal_acceleration) {
            if (reason) {
                std::ostringstream oss;
                oss << "normal acceleration " << metrics.normal_acceleration
                    << " m/s^2 exceeds usable limit "
                    << factor * limits.max_normal_acceleration << " m/s^2";
                *reason = oss.str();
            }
            return false;
        }

        if (IsFiniteLimit(limits.max_specific_thrust) &&
            metrics.specific_thrust > factor * limits.max_specific_thrust) {
            if (reason) {
                std::ostringstream oss;
                oss << "specific thrust " << metrics.specific_thrust
                    << " m/s^2 exceeds usable limit "
                    << factor * limits.max_specific_thrust << " m/s^2";
                *reason = oss.str();
            }
            return false;
        }

        if (IsFiniteLimit(limits.max_tilt_rad) &&
            metrics.tilt_rad > factor * limits.max_tilt_rad) {
            if (reason) {
                std::ostringstream oss;
                oss << "required tilt " << metrics.tilt_rad * 180.0 / PI()
                    << " deg exceeds usable limit "
                    << factor * limits.max_tilt_rad * 180.0 / PI()
                    << " deg";
                *reason = oss.str();
            }
            return false;
        }

        if (IsFiniteLimit(limits.max_heading_rate_rad_s) &&
            metrics.heading_rate_rad_s >
                factor * limits.max_heading_rate_rad_s) {
            if (reason) {
                std::ostringstream oss;
                oss << "3D tangent-heading rate "
                    << metrics.heading_rate_rad_s
                    << " rad/s exceeds usable limit "
                    << factor * limits.max_heading_rate_rad_s << " rad/s";
                *reason = oss.str();
            }
            return false;
        }

        if (IsFiniteLimit(limits.max_yaw_rate_rad_s) &&
            metrics.yaw_rate_rad_s > factor * limits.max_yaw_rate_rad_s) {
            if (reason) {
                std::ostringstream oss;
                oss << "tangent-yaw rate " << metrics.yaw_rate_rad_s
                    << " rad/s exceeds usable limit "
                    << factor * limits.max_yaw_rate_rad_s << " rad/s";
                *reason = oss.str();
            }
            return false;
        }

        return true;
    }

    static void ThrowIfInfeasible(const CurveState& state,
                                  const FeasibilityLimits& limits,
                                  const std::string& curve_name,
                                  double parameter) {
        const FeasibilityMetrics metrics = ComputeMetrics(state, limits);
        std::string reason;
        if (!MetricsAreFeasible(metrics, limits, &reason)) {
            std::ostringstream oss;
            oss << curve_name << " is infeasible at parameter " << parameter
                << " for max_speed=" << limits.max_speed << " m/s: "
                << reason << ". Curvature=" << metrics.curvature
                << " 1/m (local radius=";
            if (metrics.curvature > EPSILON()) {
                oss << 1.0 / metrics.curvature;
            } else {
                oss << "infinity";
            }
            oss << " m). Increase the curve radius/amplitudes, reduce the "
                   "frequency ratios, lower max_speed, or raise the verified "
                   "vehicle/controller limits.";
            throw std::runtime_error(oss.str());
        }
    }

    template <typename CurveEvaluator>
    static void ValidateCurve(const CurveEvaluator& evaluate,
                              double period,
                              double max_frequency,
                              const FeasibilityLimits& limits,
                              const std::string& curve_name) {
        if (!(period > 0.0) || !std::isfinite(period)) {
            throw std::invalid_argument("Curve period must be finite and positive.");
        }

        // At least 2048 samples per oscillation of the highest-frequency axis.
        const int frequency_scaled_samples = static_cast<int>(std::ceil(
            2048.0 * period * max_frequency / (2.0 * PI())));
        const int samples =
            std::max(limits.feasibility_samples, frequency_scaled_samples);

        for (int i = 0; i < samples; ++i) {
            const double u = period * static_cast<double>(i) /
                             static_cast<double>(samples);
            ThrowIfInfeasible(evaluate(u), limits, curve_name, u);
        }
    }

    template <typename CurveEvaluator>
    static bool AdvanceByArcLengthAndTurning(
        const CurveEvaluator& evaluate,
        double period,
        double requested_distance,
        const FeasibilityLimits& limits,
        double& parameter) {
        double accumulated_distance = 0.0;
        double accumulated_turn = 0.0;

        while (parameter < period - EPSILON()) {
            const CurveState current = evaluate(parameter);
            const double parameter_speed = current.first_derivative.norm();
            if (parameter_speed <= limits.min_parameter_speed) {
                throw std::runtime_error(
                    "Encountered a curve cusp while integrating arc length.");
            }

            double du = std::min(
                limits.max_parameter_step,
                limits.integration_arc_step / parameter_speed);
            du = std::min(du, period - parameter);
            if (du <= EPSILON()) {
                return false;
            }

            const CurveState next = evaluate(parameter + du);
            const double next_parameter_speed = next.first_derivative.norm();
            if (next_parameter_speed <= limits.min_parameter_speed) {
                throw std::runtime_error(
                    "Encountered a curve cusp while integrating arc length.");
            }

            const double ds =
                0.5 * (parameter_speed + next_parameter_speed) * du;
            const double kappa_current =
                CurvatureVector(current, limits.min_parameter_speed).norm();
            const double kappa_next =
                CurvatureVector(next, limits.min_parameter_speed).norm();
            const double dturn = 0.5 * (kappa_current + kappa_next) * ds;

            double fraction = 1.0;
            if (accumulated_distance + ds > requested_distance) {
                fraction = std::min(
                    fraction,
                    (requested_distance - accumulated_distance) /
                        std::max(ds, EPSILON()));
            }
            if (accumulated_turn + dturn >
                limits.max_tangent_change_rad) {
                fraction = std::min(
                    fraction,
                    (limits.max_tangent_change_rad - accumulated_turn) /
                        std::max(dturn, EPSILON()));
            }

            if (fraction < 1.0) {
                // Avoid a zero-progress loop from round-off at a threshold.
                fraction = std::max(fraction, 1e-6);
                parameter += fraction * du;
                return true;
            }

            parameter += du;
            accumulated_distance += ds;
            accumulated_turn += dturn;

            if (accumulated_distance >= requested_distance - EPSILON() ||
                accumulated_turn >=
                    limits.max_tangent_change_rad - EPSILON()) {
                return true;
            }
        }

        return false;
    }

    static Point MakePoint(const CurveState& state) {
        const double tangent_norm = state.first_derivative.norm();
        if (!std::isfinite(tangent_norm) || tangent_norm <= EPSILON()) {
            throw std::runtime_error(
                "Cannot create a 3D heading from a zero curve tangent.");
        }
        const Vec3 tangent = state.first_derivative / tangent_norm;
        return {
            static_cast<float>(state.position.x),
            static_cast<float>(state.position.y),
            static_cast<float>(state.position.z),
            static_cast<float>(tangent.x),
            static_cast<float>(tangent.y),
            static_cast<float>(tangent.z)};
    }

public:
    WaypointGenerator() {
        std::random_device rd;
        rng_.seed(rd());
    }

    explicit WaypointGenerator(std::uint32_t seed) : rng_(seed) {}

    /** Greatest Common Divisor for positive rational-like frequencies. */
    inline double gcd_double(double a, double b) const {
        a = std::abs(a);
        b = std::abs(b);

        while (b > 1e-7) {
            double remainder = std::fmod(a, b);
            if (std::abs(b - remainder) < 1e-7) {
                remainder = 0.0;
            }
            a = b;
            b = remainder;
        }
        return a;
    }

    inline double calculate_period(double freq_a, double freq_b) const {
        const double common_freq = gcd_double(freq_a, freq_b);
        if (common_freq < 1e-7) {
            return 0.0;
        }
        return 2.0 * PI() / common_freq;
    }

    inline double calculate_period(double freq_a,
                                   double freq_b,
                                   double freq_c) const {
        const double common_freq =
            gcd_double(freq_a, gcd_double(freq_b, freq_c));
        if (common_freq < 1e-7) {
            return 0.0;
        }
        return 2.0 * PI() / common_freq;
    }

    /** Validate a closed 3D Lissajous curve without generating waypoints.
     *
     * This uses the same curvature and vehicle-limit checks as
     * GenerateLissajousWaypoints, but avoids the arc-length integration when
     * callers only need to accept or reject sampled curve parameters.
     */
    void ValidateLissajousCurve(
        float A,
        float B,
        float C,
        float a,
        float b,
        float c,
        float delta_x,
        float delta_y,
        float delta_z,
        const FeasibilityLimits& limits) const {
        ValidateLimits(limits);
        if (!(A > 0.0f) || !(B > 0.0f) || !(C > 0.0f)) {
            throw std::invalid_argument(
                "Lissajous amplitudes A, B, and C must be positive.");
        }
        ValidateFrequency(a, "Frequency a");
        ValidateFrequency(b, "Frequency b");
        ValidateFrequency(c, "Frequency c");

        const auto evaluate = [=](double u) -> CurveState {
            const double ax = a * u + delta_x;
            const double by = b * u + delta_y;
            const double cz = c * u + delta_z;
            return {
                Vec3(
                    A * std::sin(ax),
                    B * std::sin(by),
                    C * std::sin(cz)),
                Vec3(
                    A * a * std::cos(ax),
                    B * b * std::cos(by),
                    C * c * std::cos(cz)),
                Vec3(
                    -A * a * a * std::sin(ax),
                    -B * b * b * std::sin(by),
                    -C * c * c * std::sin(cz))};
        };

        const double period = calculate_period(a, b, c);
        ValidateCurve(
            evaluate,
            period,
            std::max({static_cast<double>(a),
                      static_cast<double>(b),
                      static_cast<double>(c)}),
            limits,
            "3D Lissajous");
    }

    std::vector<Point> GenerateCircleWaypoints(
        float radius,
        float center_x,
        float center_y,
        float min_dist,
        float max_dist,
        float z,
        const FeasibilityLimits& limits) {
        ValidateSpacing(min_dist, max_dist);
        ValidateLimits(limits);
        if (!(radius > 0.0f)) {
            throw std::invalid_argument("Circle radius must be positive.");
        }

        const auto evaluate = [=](double theta) -> CurveState {
            return {
                Vec3(
                    center_x + radius * std::cos(theta),
                    center_y + radius * std::sin(theta),
                    z),
                Vec3(
                    -radius * std::sin(theta),
                    radius * std::cos(theta),
                    0.0),
                Vec3(
                    -radius * std::cos(theta),
                    -radius * std::sin(theta),
                    0.0)};
        };

        // Circle curvature is constant, so one analytic check is sufficient.
        ThrowIfInfeasible(evaluate(0.0), limits, "Circle", 0.0);

        const double curvature_spacing_cap =
            radius * limits.max_tangent_change_rad;
        const double average_requested = 0.5 * (min_dist + max_dist);
        const double average_effective =
            std::min(average_requested, curvature_spacing_cap);

        std::vector<Point> waypoints;
        const int estimate = static_cast<int>(std::ceil(
            2.0 * PI() * radius /
            std::max(average_effective, static_cast<double>(EPSILON()))));
        waypoints.reserve(std::max(estimate, 1));

        double theta = 0.0;
        while (theta < 2.0 * PI() - EPSILON()) {
            const CurveState state = evaluate(theta);
            waypoints.push_back(MakePoint(state));

            const double requested = GetRandomDistance(min_dist, max_dist);
            const double effective_distance =
                std::min(requested, curvature_spacing_cap);
            const double delta_theta = effective_distance / radius;
            if (delta_theta <= EPSILON()) {
                throw std::runtime_error(
                    "Circle waypoint integration made no progress.");
            }
            theta += delta_theta;
        }

        return waypoints;
    }

    /** 2D Lissajous: x=A sin(a u+dx), y=B sin(b u+dy), z=constant. */
    std::vector<Point> GenerateLissajousWaypoints(
        float A,
        float B,
        float a,
        float b,
        float delta_x,
        float delta_y,
        float min_dist,
        float max_dist,
        float z,
        const FeasibilityLimits& limits) {
        ValidateSpacing(min_dist, max_dist);
        ValidateLimits(limits);
        if (!(A > 0.0f) || !(B > 0.0f)) {
            throw std::invalid_argument("Lissajous amplitudes A and B must be positive.");
        }
        ValidateFrequency(a, "Frequency a");
        ValidateFrequency(b, "Frequency b");

        const auto evaluate = [=](double u) -> CurveState {
            const double ax = a * u + delta_x;
            const double by = b * u + delta_y;
            return {
                Vec3(A * std::sin(ax), B * std::sin(by), z),
                Vec3(
                    A * a * std::cos(ax),
                    B * b * std::cos(by),
                    0.0),
                Vec3(
                    -A * a * a * std::sin(ax),
                    -B * b * b * std::sin(by),
                    0.0)};
        };

        const double period = calculate_period(a, b);
        ValidateCurve(
            evaluate,
            period,
            std::max(static_cast<double>(a), static_cast<double>(b)),
            limits,
            "2D Lissajous");

        std::vector<Point> waypoints;
        double u = 0.0;
        CurveState initial = evaluate(u);
        waypoints.push_back(MakePoint(initial));

        while (u < period - EPSILON()) {
            const double requested = GetRandomDistance(min_dist, max_dist);
            const double previous_u = u;
            if (!AdvanceByArcLengthAndTurning(
                    evaluate, period, requested, limits, u)) {
                break;
            }
            if (u >= period - EPSILON()) {
                break;  // Do not duplicate the first point at the closed endpoint.
            }
            if (u <= previous_u + EPSILON()) {
                throw std::runtime_error(
                    "2D Lissajous waypoint integration made no progress.");
            }

            const CurveState state = evaluate(u);
            ThrowIfInfeasible(state, limits, "2D Lissajous", u);
            waypoints.push_back(MakePoint(state));
        }

        return waypoints;
    }

    /** YZ-planar Lissajous:
     *  x=x_offset, y=A sin(a u+dy), z=z_offset+C sin(c u+dz).
     */
    std::vector<Point> GenerateYZLissajousWaypoints(
        float A,
        float C,
        float a,
        float c,
        float delta_y,
        float delta_z,
        float min_dist,
        float max_dist,
        float x_offset,
        float z_offset,
        const FeasibilityLimits& limits) {
        ValidateSpacing(min_dist, max_dist);
        ValidateLimits(limits);
        if (!(A > 0.0f) || !(C > 0.0f)) {
            throw std::invalid_argument(
                "YZ Lissajous amplitudes A and C must be positive.");
        }
        ValidateFrequency(a, "Frequency a");
        ValidateFrequency(c, "Frequency c");

        const auto evaluate = [=](double u) -> CurveState {
            const double ay = a * u + delta_y;
            const double cz = c * u + delta_z;
            return {
                Vec3(
                    x_offset,
                    A * std::sin(ay),
                    z_offset + C * std::sin(cz)),
                Vec3(
                    0.0,
                    A * a * std::cos(ay),
                    C * c * std::cos(cz)),
                Vec3(
                    0.0,
                    -A * a * a * std::sin(ay),
                    -C * c * c * std::sin(cz))};
        };

        const double period = calculate_period(a, c);
        ValidateCurve(
            evaluate,
            period,
            std::max(static_cast<double>(a), static_cast<double>(c)),
            limits,
            "YZ Lissajous");

        std::vector<Point> waypoints;
        double u = 0.0;
        waypoints.push_back(MakePoint(evaluate(u)));

        while (u < period - EPSILON()) {
            const double requested = GetRandomDistance(min_dist, max_dist);
            const double previous_u = u;
            if (!AdvanceByArcLengthAndTurning(
                    evaluate, period, requested, limits, u)) {
                break;
            }
            if (u >= period - EPSILON()) {
                break;
            }
            if (u <= previous_u + EPSILON()) {
                throw std::runtime_error(
                    "YZ Lissajous waypoint integration made no progress.");
            }

            const CurveState state = evaluate(u);
            ThrowIfInfeasible(state, limits, "YZ Lissajous", u);
            waypoints.push_back(MakePoint(state));
        }

        return waypoints;
    }

    /** 3D Lissajous with a normalized full 3D tangent heading. */
    std::vector<Point> GenerateLissajousWaypoints(
        float A,
        float B,
        float C,
        float a,
        float b,
        float c,
        float delta_x,
        float delta_y,
        float delta_z,
        float min_dist,
        float max_dist,
        float z_offset,
        const FeasibilityLimits& limits) {
        ValidateSpacing(min_dist, max_dist);
        ValidateLimits(limits);
        if (!(A > 0.0f) || !(B > 0.0f) || !(C > 0.0f)) {
            throw std::invalid_argument(
                "Lissajous amplitudes A, B, and C must be positive.");
        }
        ValidateFrequency(a, "Frequency a");
        ValidateFrequency(b, "Frequency b");
        ValidateFrequency(c, "Frequency c");

        const auto evaluate = [=](double u) -> CurveState {
            const double ax = a * u + delta_x;
            const double by = b * u + delta_y;
            const double cz = c * u + delta_z;
            return {
                Vec3(
                    A * std::sin(ax),
                    B * std::sin(by),
                    z_offset + C * std::sin(cz)),
                Vec3(
                    A * a * std::cos(ax),
                    B * b * std::cos(by),
                    C * c * std::cos(cz)),
                Vec3(
                    -A * a * a * std::sin(ax),
                    -B * b * b * std::sin(by),
                    -C * c * c * std::sin(cz))};
        };

        const double period = calculate_period(a, b, c);
        ValidateCurve(
            evaluate,
            period,
            std::max({static_cast<double>(a),
                      static_cast<double>(b),
                      static_cast<double>(c)}),
            limits,
            "3D Lissajous");

        std::vector<Point> waypoints;
        double u = 0.0;
        CurveState initial = evaluate(u);
        waypoints.push_back(MakePoint(initial));

        while (u < period - EPSILON()) {
            const double requested = GetRandomDistance(min_dist, max_dist);
            const double previous_u = u;
            if (!AdvanceByArcLengthAndTurning(
                    evaluate, period, requested, limits, u)) {
                break;
            }
            if (u >= period - EPSILON()) {
                break;
            }
            if (u <= previous_u + EPSILON()) {
                throw std::runtime_error(
                    "3D Lissajous waypoint integration made no progress.");
            }

            const CurveState state = evaluate(u);
            ThrowIfInfeasible(state, limits, "3D Lissajous", u);
            waypoints.push_back(MakePoint(state));
        }

        return waypoints;
    }

    // Convenience overload: explicit limits with zero z-offset.
    std::vector<Point> GenerateLissajousWaypoints(
        float A,
        float B,
        float C,
        float a,
        float b,
        float c,
        float delta_x,
        float delta_y,
        float delta_z,
        float min_dist,
        float max_dist,
        const FeasibilityLimits& limits) {
        return GenerateLissajousWaypoints(
            A, B, C, a, b, c, delta_x, delta_y, delta_z,
            min_dist, max_dist, 0.0f, limits);
    }

    // Backward-compatible overloads using the default feasibility limits.
    std::vector<Point> GenerateCircleWaypoints(
        float radius,
        float center_x,
        float center_y,
        float min_dist,
        float max_dist,
        float z) {
        return GenerateCircleWaypoints(
            radius, center_x, center_y, min_dist, max_dist, z,
            FeasibilityLimits{});
    }

    std::vector<Point> GenerateLissajousWaypoints(
        float A,
        float B,
        float a,
        float b,
        float delta_x,
        float delta_y,
        float min_dist,
        float max_dist,
        float z) {
        return GenerateLissajousWaypoints(
            A, B, a, b, delta_x, delta_y, min_dist, max_dist, z,
            FeasibilityLimits{});
    }

    std::vector<Point> GenerateLissajousWaypoints(
        float A,
        float B,
        float C,
        float a,
        float b,
        float c,
        float delta_x,
        float delta_y,
        float delta_z,
        float min_dist,
        float max_dist,
        float z_offset = 0.0f) {
        return GenerateLissajousWaypoints(
            A, B, C, a, b, c, delta_x, delta_y, delta_z,
            min_dist, max_dist, z_offset, FeasibilityLimits{});
    }
};

#endif  // COMMON_WAYPOINT_GENERATOR_HH
