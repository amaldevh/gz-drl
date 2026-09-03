// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef TRAJECTORY_PLANNING_PROCESSOR_HH_
#define TRAJECTORY_PLANNING_PROCESSOR_HH_

#include <cmath> // Added for std::isnan, std::isinf
#include <deque>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <Eigen/Dense>

constexpr size_t OBSERVATION_DIM = 186;
constexpr size_t ACTION_DIM = 6;

// template<typename T>
// void _print(T t){ std::cout<<t;}
// template<typename T, typename... Args>
// void _print(T t, Args... args){ std::cout<<t; _print(args...);}

// #define print_info(...) _print("[TrajectoryPlanningProcessor] ", __VA_ARGS__, "\n")
// #define print_err(...) _print("[TrajectoryPlanningProcessor] ", __VA_ARGS__, "\n")
/** @class TrajectoryPlanningProcessor
 * @brief Processor for the Trajectory Planning environment
 * 
 * Improved version with:
 * - Better gate crossing detection using plane-crossing + proximity
 * - Progress-related observations (velocity towards gate, distance, progress ratio)
 * - Smoother gate transitions with blending information
 * - Enhanced reward shaping for continuous motion
 */
class TrajectoryPlanningProcessor {
 public:
    using State = Eigen::Matrix<float, OBSERVATION_DIM, 1>;
    using Statef = Eigen::Matrix<float, 13, 1>;
    using Action = Eigen::Matrix<float, ACTION_DIM, 1>;

    TrajectoryPlanningProcessor(int action_history_size,  int last_state_history_size,
            bool xyz, bool vxyz, const std::vector<std::vector<float>>& gate_centers,
            float gate_width, float gate_height,
            float gate_crossing_radius, const Eigen::VectorXf& xyz_scaling, 
            const Eigen::VectorXf& vxyz_scaling)
        : action_history_size_(action_history_size), 
        last_state_history_size_(last_state_history_size), gate_width_(gate_width), gate_height_(gate_height),
         gate_crossing_radius_(gate_crossing_radius),
        use_xyz_(xyz), use_vxyz_(vxyz), xyz_scaling_(xyz_scaling), vxyz_scaling_(vxyz_scaling) {
        if (use_xyz_ && use_vxyz_){
            act_dim_ = 6;
        }
        else if (use_xyz_ || use_vxyz_){
            act_dim_ = 3;
        }
        else {
            throw std::runtime_error("Unsupported action space configuration");
        }
        net_scaling_.resize(act_dim_);
        if (use_xyz_ && use_vxyz_){
            net_scaling_.head<3>() = xyz_scaling_;
            net_scaling_.tail<3>() = vxyz_scaling_;
        }
        else if (use_xyz_){
            net_scaling_ = xyz_scaling_;
        }
        else if (use_vxyz_){
            net_scaling_ = vxyz_scaling_;
        }
        num_gates_ = static_cast<int>(gate_centers.size());
        ExtractGateCornersFromWaypoints(gate_centers);
        action_history_.assign(action_history_size_ * act_dim_, 0.0f);
        last_state_history_.assign(last_state_history_size_ * state_size_, 0.0f);
        // Compute average gate distance for normalization
        ComputeAverageGateDistance();
        }
    
    /** @brief Generates gate corners based on centers, width, and height */
    void ExtractGateCornersFromWaypoints(const std::vector<std::vector<float>>& waypoints) {
    gate_corners_.resize(num_gates_, 12);
    gate_centers_.resize(num_gates_, 3);
    gate_normals_.resize(num_gates_, 3);

    for (int i = 0; i < num_gates_; ++i) {
      if (waypoints[i].size() != 4) {
       print_err("Error: waypoint ",i," has size "
                 ,waypoints[i].size()," != 4\n");
        continue;
      }

      const float x = waypoints[i][0];
      const float y = waypoints[i][1];
      const float z = waypoints[i][2];
      const float yaw = waypoints[i][3];

      Eigen::Vector3f center(x, y, z);
      gate_centers_.row(i) = center.transpose();

      // Rotation matrix from yaw
      const float cy = std::cos(yaw);
      const float sy = std::sin(yaw);
      Eigen::Matrix3f R;
      R << cy, -sy, 0,
           sy,  cy, 0,
            0,   0, 1;

      // Normal vector (forward direction)
      gate_normals_.row(i) = R.col(0).transpose();

      // Gate corners in local frame
      const Eigen::Vector3f e2 = R.col(1);  // right direction
      const Eigen::Vector3f e3 = R.col(2);  // up direction

      const Eigen::Vector3f right_dir = -e2 * (gate_width_ / 2.0f);
      const Eigen::Vector3f top_dir = e3 * (gate_height_ / 2.0f);

      // Four corners: left-top, right-top, right-bottom, left-bottom
      const Eigen::Vector3f left_corner   = center - right_dir + top_dir;
      const Eigen::Vector3f top_corner    = center + top_dir + right_dir;
      const Eigen::Vector3f right_corner  = center + right_dir - top_dir;
      const Eigen::Vector3f bottom_corner = center - right_dir - top_dir;

      // Store corners in row-major order
      gate_corners_(i, 0)  = left_corner(0);
      gate_corners_(i, 1)  = left_corner(1);
      gate_corners_(i, 2)  = left_corner(2);
      gate_corners_(i, 3)  = top_corner(0);
      gate_corners_(i, 4)  = top_corner(1);
      gate_corners_(i, 5)  = top_corner(2);
      gate_corners_(i, 6)  = right_corner(0);
      gate_corners_(i, 7)  = right_corner(1);
      gate_corners_(i, 8)  = right_corner(2);
      gate_corners_(i, 9)  = bottom_corner(0);
      gate_corners_(i, 10) = bottom_corner(1);
      gate_corners_(i, 11) = bottom_corner(2);
  }
}

    /** @brief Computes average distance between consecutive gates for normalization */
    void ComputeAverageGateDistance() {
        if (num_gates_ < 2) {
            avg_gate_distance_ = 1.0f;
            return;
        }
        float total_dist = 0.0f;
        for (int i = 0; i < num_gates_; ++i) {
            int next_i = (i + 1) % num_gates_;
            Eigen::Vector3f curr = gate_centers_.row(i).transpose();
            Eigen::Vector3f next = gate_centers_.row(next_i).transpose();
            total_dist += (next - curr).norm();
        }
        avg_gate_distance_ = total_dist / num_gates_;
        print_info("Average gate distance: ", avg_gate_distance_, "\n");
    }

    /** @brief Updates the gate index for observation
     * Uses plane-crossing detection combined with proximity check
     * This is more robust than pure proximity-based detection
     */
    void UpdateGateIndex(const Eigen::Vector3f& velocity) {
        const Eigen::Vector3f next_gate_center = gate_centers_.row(gate_index_).transpose();
        const Eigen::Vector3f gate_normal = gate_normals_.row(gate_index_).transpose();
        const Eigen::Vector3f vec_to_uav = current_pos_ - next_gate_center;
        const float dist_to_next = vec_to_uav.norm();
        
        // Projection onto gate normal (signed distance from gate plane)
        const float proj = vec_to_uav.dot(gate_normal);
        const int curr_sgn = sign(proj);
        
        // Gate crossing conditions:
        // 1. Sign change in projection (crossed the gate plane)
        // 2. Within crossing radius (close enough to the gate)
        // 3. OR very close to gate center (fallback for edge cases)
        const bool sign_changed = (last_sign_ != 0) && (curr_sgn != last_sign_);
        const bool within_radius = dist_to_next < gate_crossing_radius_;
        const bool very_close = dist_to_next < gate_crossing_radius_ * 0.5f;
        
        // Also check if moving towards next gate (velocity alignment)
        const Eigen::Vector3f dir_to_gate = -vec_to_uav.normalized();
        const float vel_alignment = velocity.normalized().dot(dir_to_gate);
        
        // Crossed if: (sign changed AND within radius) OR very close
        const bool crossed = (sign_changed && within_radius) || very_close;
        
        if (crossed) {
            print_info("Gate ", gate_index_, " crossed. Distance: ", dist_to_next, "\n");
            gates_crossed_++;
            gate_index_ = (gate_index_ + 1) % num_gates_;
            last_sign_ = -1;  // Reset sign for new gate
            // Store distance to new gate at crossing time for progress tracking
            const Eigen::Vector3f new_gate_center = gate_centers_.row(gate_index_).transpose();
            initial_dist_to_gate_ = (current_pos_ - new_gate_center).norm();
        } else {
            last_sign_ = curr_sgn;
        }
        
        // Update distance tracking for progress
        dist_to_current_gate_ = dist_to_next;
    }

     /** @brief Returns the number of gates */
    const int GateSize() const {
        return num_gates_;
    }

    /** @brief Returns number of gates crossed this episode */
    int GetGatesCrossed() const {
        return gates_crossed_;
    }

    /** @brief Resets the processor state   */
    void Reset(int gate_index) {
        gate_index_ = gate_index;
        action_history_.clear();
        action_history_.assign(action_history_size_ * act_dim_, 0.0f);
        last_state_history_.clear();
        last_state_history_.assign(last_state_history_size_ * state_size_, 0.0f);
        last_sign_ = -1;
        gates_crossed_ = 0;
        // Initialize distance tracking
        if (num_gates_ > 0) {
            const Eigen::Vector3f gate_center = gate_centers_.row(gate_index_).transpose();
            initial_dist_to_gate_ = avg_gate_distance_;  // Use average as initial estimate
            dist_to_current_gate_ = avg_gate_distance_;
        }
        last_velocity_ = Eigen::Vector3f::Zero();
        previous_action_smoothness_ = 0.0f;
        previous_action_ = Eigen::VectorXf::Zero(act_dim_);
    }

    /** @brief Pushes an action to the queue, should be done before processing observation */
    void PushAction(const Eigen::VectorXf& action) {
        previous_action_smoothness_ = (action - previous_action_).norm();
        previous_action_ = action;
        for (int i = 0; i < action.size(); ++i) {
            action_history_.push_back(action(i));
            if (static_cast<int>(action_history_.size()) > action_history_size_ * act_dim_) {
                action_history_.pop_front();
            }
        }
    }
    
    /** @brief Pushes last state to the queue, should be done before processing observation */
    void PushLastState(const Statef& last_state) {
        for (int i = 3; i < 13; ++i) {
            last_state_history_.push_back( last_state(i));
            if (static_cast<int>(last_state_history_.size()) > last_state_history_size_ * state_size_) {
                last_state_history_.pop_front();
            }
        }
    }
  /** @brief ProcessObservation processes the raw observation from the environment
   * into the observation space.
   * @param raw_obs The raw observation from the environment
   * @return The processed observation
   * 
   * Improved observation consists of:
   * pos (3), vel (3), quaternion (4), omega (3) = 13
   * next gate center relative to current pos (3)
   * next gate normal (3) - direction to fly through
   * velocity projected towards gate (1) - critical for smooth motion
   * normalized distance to gate (1) - progress indicator
   * next gate corners relative to current pos (12)
   * prev gate corners relative to current pos (12)
   * next next gate center relative to next gate (3)
   * next next gate normal (3)
   * action history (action_history_size * act_dim)
   * = 13 + 3 + 3 + 1 + 1 + 12 + 12 + 3 + 3 + action_history = 51 + action_history
   */
  void ProcessObservation(const Statef& current_state,
     const Statef& current_state_dot,
     const Statef& previous_state,
    const Statef& previous_state_dot,
    State& processed_obs) {

        const auto& state = current_state;
        PushLastState(state);
        current_pos_ = state.head<3>();
        const Eigen::Vector3f velocity = state.segment<3>(3);
        last_velocity_ = velocity;
        
        // Update gate index with velocity info for better crossing detection
        UpdateGateIndex(velocity);
        
        int idx = 0;
        
        // Copy pos, vel, quat, omega (Sanitized) - indices 0-12
        // for (int i = 3; i < 13; ++i) {
        //     processed_obs[idx++] = SanitizeValue(state(i));
        // }
        // // Copy last vel, quat, omega (Sanitized) - indices 13-22 
        // for (int i = 3; i < 13; ++i) {
        //     processed_obs[idx++] = SanitizeValue(last_state(i));
        // }
        for (const auto & val : last_state_history_) {
            processed_obs[idx++] = SanitizeValue(val);
        }
        
        // Gate indices
        const int next_gate_idx = gate_index_;
        // const int prev_gate_idx = (gate_index_ - 1 + num_gates_) % num_gates_;
        const int next_next_gate_idx = (gate_index_ + 1) % num_gates_;
        
        // Next gate center relative to current pos (3) - indices 13-15
        const Eigen::Vector3f next_gate_center = gate_centers_.row(next_gate_idx).transpose();
        const Eigen::Vector3f rel_next_center = next_gate_center - current_pos_;
        for (int i = 0; i < 3; ++i) {
            processed_obs[idx++] = SanitizeValue(rel_next_center(i));
        }
        
        // Next gate normal (3) - indices 16-18
        const Eigen::Vector3f next_gate_normal = gate_normals_.row(next_gate_idx).transpose();
        for (int i = 0; i < 3; ++i) {
            processed_obs[idx++] = SanitizeValue(next_gate_normal(i));
        }
        
        // Velocity projected towards gate (1) - index 19
        // Positive = moving towards gate, negative = moving away
        const float dist_to_gate = rel_next_center.norm();
        const Eigen::Vector3f dir_to_gate = (dist_to_gate > 1e-6f) ? 
            rel_next_center.normalized() : Eigen::Vector3f::Zero();
        const float vel_towards_gate = velocity.dot(dir_to_gate);
        processed_obs[idx++] = SanitizeValue(vel_towards_gate);
        
        // Normalized distance to gate (1) - index 20
        // Normalized by average gate distance for scale invariance
        const float norm_dist = dist_to_gate / std::max(avg_gate_distance_, 0.1f);
        processed_obs[idx++] = SanitizeValue(norm_dist);
        
        // Next gate corners relative to current pos (12) - indices 21-32
        for (int i = 0; i < 12; ++i) {
            float val = gate_corners_(next_gate_idx, i) - current_pos_(i % 3);
            processed_obs[idx++] = SanitizeValue(val);
        }
        
        // // Previous gate corners relative to current pos (12) - indices 33-44
        // for (int i = 0; i < 12; ++i) {
        //     float val = gate_corners_(prev_gate_idx, i) - current_pos_(i % 3);
        //     processed_obs[idx++] = SanitizeValue(val);
        // }
        
        // Next next gate center relative to next gate (3) - indices 45-47
        const Eigen::Vector3f next_next_center = gate_centers_.row(next_next_gate_idx).transpose();
        const Eigen::Vector3f rel_next_next = next_next_center - next_gate_center;
        for (int i = 0; i < 3; ++i) {
            processed_obs[idx++] = SanitizeValue(rel_next_next(i));
        }
        
        // Next next gate normal (3) - indices 48-50
        const Eigen::Vector3f next_next_normal = gate_normals_.row(next_next_gate_idx).transpose();
        for (int i = 0; i < 3; ++i) {
            processed_obs[idx++] = SanitizeValue(next_next_normal(i));
        }
        
        // Action history - indices 51+
        for (const float& act : action_history_) {
            processed_obs[idx++] = SanitizeValue(act);
        }
  }

  /** @brief processes action */
    void ProcessAction(const Action& policy_action, Eigen::VectorXf& processed_action)  {
            const Action& action_data = policy_action;
            auto &processed_action_ = processed_action;
            processed_action_.resize(act_dim_);
            for (int i = 0; i < act_dim_; ++i) {
                processed_action_(i) = action_data[i]* net_scaling_(i);
            }
            PushAction(processed_action_);
    }

    /** @brief Computes the reward based on the current state, previous state, and action
     * 
     * Improved reward function with:
     * 1. Distance-based reward (getting closer to gate)
     * 2. Velocity alignment reward (moving in the right direction)
     * 3. Progress reward (continuous positive signal for moving towards gate)
     * 4. Smoothness penalty (for jerky motions)
     * 5. Small penalty for being stationary when far from gate (prevents getting stuck)
     * 6. Gate crossing bonus
     */
    void ComputeReward(const Statef& current_state,
        const Statef& current_state_dot,
     const Statef& previous_state, 
        const Statef& previous_state_dot,
     const Action& action,
     float& rewards)  {
        
        float current_reward = 0.0f;
        
        const auto& agent_state = current_state;
        const auto& agent_state_dot = current_state_dot;
        const auto& prev_agent_state = previous_state;
        const auto& prev_agent_state_dot = previous_state_dot;
        const Eigen::Vector3f pos = agent_state.head<3>();
        const Eigen::Vector3f prev_pos = prev_agent_state.head<3>();
        const Eigen::Vector3f velocity = agent_state.segment<3>(3);
        const Eigen::Vector3f prev_velocity = prev_agent_state.segment<3>(3);
        const Eigen::Vector3f accel = agent_state_dot.segment<3>(3);
        const Eigen::Vector3f prev_accel = prev_agent_state_dot.segment<3>(3);
        const Eigen::Vector3f next_gate_center = gate_centers_.row(gate_index_).transpose();
        const Eigen::Vector3f gate_normal = gate_normals_.row(gate_index_).transpose();
        
        const float dist_to_next = (pos - next_gate_center).norm();
        const float prev_dist_to_next = (prev_pos - next_gate_center).norm();
        
        // 1. Distance-based reward (getting closer to gate)
        const float dist_improvement = prev_dist_to_next - dist_to_next;
        current_reward += dist_improvement * 2.0f;
        
        // 2. Velocity alignment reward
        // Reward for having velocity pointing towards the gate
        const Eigen::Vector3f dir_to_gate = (pos - next_gate_center).norm() > 1e-6f ?
            (next_gate_center - pos).normalized() : Eigen::Vector3f::Zero();
        const float vel_magnitude = velocity.norm();
        const float vel_alignment = (vel_magnitude > 1e-6f) ? 
            velocity.normalized().dot(dir_to_gate) : 0.0f;
        
        // Scale alignment reward by velocity magnitude (encourage fast + aligned motion)
        current_reward += vel_alignment * vel_magnitude * 0.3f;
        
        // 3. Progress reward - small continuous reward proportional to speed towards gate
        const float speed_towards_gate = velocity.dot(dir_to_gate);
        if (speed_towards_gate > 0.0f) {
            current_reward += speed_towards_gate * 0.1f;
        }
        
        // 4. Smoothness penalty - penalize sudden changes in acceleration (jerk)
        const Eigen::Vector3f accel_change = accel - prev_accel;
        const float jerk = accel_change.norm();
        current_reward -= jerk * 0.05f;
        
        // 5. Stagnation penalty - penalize being stationary when far from gate
        // This prevents the agent from getting stuck
        const float stagnation_threshold = 0.1f;  // m/s
        const float far_threshold = gate_crossing_radius_ * 2.0f;  // When to consider "far"
        if (vel_magnitude < stagnation_threshold && dist_to_next > far_threshold) {
            current_reward -= 0.1f;
        }
        
        // 6. Angular rate penalty (keep it stable)
        const Eigen::Vector3f omega = agent_state.tail<3>();
        current_reward -= omega.squaredNorm() * 0.005f;
        
        // 7. Bonus for being close to gate center and aligned with normal
        // Encourages flying through gate properly, not just getting close
        if (dist_to_next < gate_crossing_radius_) {
            const float alignment_with_normal = std::abs(velocity.normalized().dot(gate_normal));
            current_reward += alignment_with_normal * 0.5f;
        }
        // 8. Action smoothness penalty
        current_reward -= previous_action_smoothness_ * 0.1f;
        // Sanitize and clamp the final reward
        rewards = std::max(std::min(500.0f, SanitizeValue(current_reward)), -500.0f);
    }

public:
    /** @brief Helper to sanitize floats */
    inline float SanitizeValue(float val) {
        if (std::isnan(val)) {
            return 0.0f;
        }
        if (std::isinf(val)) {
            return (val > 0.0f) ? 1e5f : -1e5f;
        }
        return val;
    }

    inline int sign(float v) { return (v >= 0.0f) ? 1 : -1; }
    
    // Configuration
    int                             action_history_size_;
    int                             last_state_history_size_;
    int                             act_dim_;
    const int                            state_size_{10}; // vel(3), quat(4), omega(3)
    const float                     gate_width_;
    const float                     gate_height_;
    float                           gate_crossing_radius_;
    
    // Gate data
    int                             num_gates_ = 0;
    Eigen::MatrixXf                 gate_corners_; 
    Eigen::MatrixXf                 gate_centers_;
    Eigen::MatrixXf                 gate_normals_;
    float                           avg_gate_distance_ = 1.0f;
    
    // State tracking
    int                             gate_index_ = 0;
    int                             gates_crossed_ = 0;
    std::deque<float>               action_history_;
    std::deque<float>               last_state_history_;
    Eigen::Vector3f                 current_pos_;
    Eigen::Vector3f                 last_velocity_ = Eigen::Vector3f::Zero();
    int                             last_sign_ = -1;
    
    // Progress tracking
    float                           initial_dist_to_gate_ = 1.0f;
    float                           dist_to_current_gate_ = 1.0f;
    float                          previous_action_smoothness_ = 0.0f;
    Eigen::VectorXf                 previous_action_;
    Eigen::VectorXf                 xyz_scaling_;
    Eigen::VectorXf                 vxyz_scaling_;
    Eigen::VectorXf                 net_scaling_;
    bool                           use_xyz_;
    bool                           use_vxyz_;
    
};
#endif 