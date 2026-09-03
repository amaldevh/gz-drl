// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef GAZEBO_ENVPOOL_HH
#define GAZEBO_ENVPOOL_HH

#include "rl_server.hh"
#include "common_utils.hh"
#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include <memory>
#include <map>
#include "env_specs/gazebo_envpool_spec.hh"
#include "processors/processor.hh"
#include <cstdlib>
#include "envpool/core/py_envpool.h"
#include <algorithm>
#include <mutex>

/// Process-wide one-time setup for Gazebo resource and plugin search paths.
inline std::once_flag gazebo_envpool_env_vars_once;

/**
 * @brief A structure that holds randomized inertial parameters: Mass, Inertia
 *
 */
struct RandomizedParams
{
    double mass;             ///< Mass
    Eigen::Matrix3d inertia; ///< Inertia
};

/**
 * @brief Base class for Gazebo Envpool environments
 * This class should be inherited by all Gazebo Envpool environments.
 * The GazeboEnvpool class provides common functionality for all Gazebo Envpool
 * by interfacing the DRLServer.
 * The template parameter is a processor, which should be a child of GazeboProcessor.
 * At each step of the environment, the processor is used to process the raw observation
 * from the environment into the observation space, and to process the action from
 * the policy into the action space of the environment.
 * The GazeboEnvpool class also manages the DRLServer, which is used to communicate
 * with the Gazebo simulator.
 *
 * @tparam SpecCls Spec class type, the specs are defined in env_specs. See gazebo_envpool_spec.hh
 * @tparam ProcessorType Type of processor, the processors are defined in processors. See processor.hh
 */
template <typename SpecCls, typename ProcessorType>
class GazeboEnvpool : public Env<EnvSpec<SpecCls>>
{
public:
    using Parent = Env<EnvSpec<SpecCls>>;
    using Spec = EnvSpec<SpecCls>;
    /**
     * @brief Construct a new Gazebo Envpool object
     *
     * @param spec Dict containing config values
     * @param envid  Unique envid, the actual envid will depend on the current env number in the process
     */
    GazeboEnvpool(const Spec &spec, int envid) : Parent(spec, envid),
                                                 spec_(spec), envid_(envid)
    {
        auto keys = SpecCls::DefaultConfig().AllKeys();
        for (const auto &key : required_keys_)
        {
            if (std::find(keys.begin(), keys.end(), key) == keys.end())
            {
                throw std::runtime_error("Missing required config key: " + key);
            }
        }
        gz_partition_offset_base_ = spec.config["gz_partition_offset"_];
        static_assert(is_gazebo_spec<SpecCls>::value,
                      "SpecCls must be a child of GazeboSpec");
        SetEnvVars(spec);

        // Ensure there is a unique envid for each env instance
    }

    /**
     * @brief Obtain a Unique envid
     *
     * @return int The unique envid
     */
    int UniqueEnvid() const
    {
        // Derive the Gazebo partition from EnvPool's stable logical ID.  An
        // atomic construction-order counter makes the logical-env-to-RNG and
        // logical-env-to-partition mappings scheduler dependent.
        return gz_partition_offset_base_ + envid_;
    }

    /** Acquire an exception-safe, process-wide Gazebo construction lock. */
    [[nodiscard]] static std::unique_lock<std::mutex> AcquireConstructionLock()
    {
        return std::unique_lock<std::mutex>(gzdrl_gazebo_construction_mutex);
    }

    /**
     * @brief Updates the states for models specified in model_names_
     * The updated states are stored in current_state_ map
     * The previous states are stored in last_state_ map
     * The states will be obtained from the model_link_names_
     * Every Model + Link has a unique state key defined as concatenation of
     * model name  + link name.
     */
    void UpdateControlStates()
    {
        drl_server_->update_control_states();
        for (size_t i = 0; i < model_names_.size(); ++i)
        {
            const auto &model_name = model_names_[i];
            const auto &link_names = model_link_names_[i];
            for (const auto &link_name : link_names)
            {
                const auto key = model_name + link_name;
                const auto &state_tuple = drl_server_->control_states[model_name][link_name];
                last_state_[key] = std::move(current_state_[key]);
                last_state_dot_[key] = std::move(current_state_dot_[key]);
                current_state_[key] = std::get<0>(state_tuple).cast<float>();
                current_state_dot_[key] = std::get<1>(state_tuple).cast<float>();
            }
        }
    }

protected:
    /**
     * @brief Set the Env Vars
     *
     * @param spec  Env spec which also holds the config.
     */
    static void SetEnvVars(const Spec &spec)
    {
        std::call_once(gazebo_envpool_env_vars_once, [&spec]() {
            const auto merge_path = [](const std::string &configured,
                                       const char *existing)
            {
                if (configured.empty())
                    return existing ? std::string(existing) : std::string{};
                if (existing == nullptr || *existing == '\0' || configured == existing)
                    return configured;
                return configured + ":" + existing;
            };

            const auto resource_path = merge_path(
                spec.config["resources_path"_], std::getenv("GZ_SIM_RESOURCE_PATH"));
            const auto plugin_path = merge_path(
                spec.config["plugins_path"_], std::getenv("GZ_SIM_SYSTEM_PLUGIN_PATH"));
            if (!resource_path.empty())
                setenv("GZ_SIM_RESOURCE_PATH", resource_path.c_str(), 1);
            if (!plugin_path.empty())
                setenv("GZ_SIM_SYSTEM_PLUGIN_PATH", plugin_path.c_str(), 1);
        });
    }
    const Spec spec_;
    const int envid_;
    const std::vector<std::string> required_keys_{"max_steps_per_episode"}; ///< Required keys in the dict config
    std::unordered_map<std::string, RandomizedParams> randomized_params_;   ///< Randomized parameters index by state key, state key is
                                                                            /// formed by concatenating model_name and link_name
    std::shared_ptr<DRLServer> drl_server_;                                 ///< The DRL server
    std::shared_ptr<ProcessorType> processor_;                              ///< The processor for observations, rewards and actions. See processors.hh
    std::vector<std::string> model_names_;                                  ///< The model names being managed by DRLServer
    std::unordered_map<int, std::vector<std::string>> model_link_names_;    ///< The link names being tracked, indexed by models
    int max_steps_per_episode_;                                             ///< Maximum steps per episode
    bool done_ = true;                                                      ///< Done flag
    int current_step_ = 0;                                                  ///< Current step count
    int current_episode_ = 0;                                               ///< Current episode count
    std::unique_ptr<RNG<double>> double_rng_;                               ///< RNG for double values
    std::unique_ptr<RNG<int>> int_rng_;                                     ///< RNG for int values
    std::unordered_map<std::string, float> reward_;                         ///< Reward for each unique link (accessed by state_key := model_name + link_name)
    bool domain_randomization_ = false;                                     ///< Whether to apply domain randomization
    using Statef = Eigen::Matrix<float, 13, 1>;                             ///< State type
    std::unordered_map<std::string, Statef> current_state_;                 ///< Current state for each unique link (accessed by state_key := model_name + link_name)
    std::unordered_map<std::string, Statef> current_state_dot_;             ///< Current state dot for each unique link (accessed by state_key := model_name + link_name)
    std::unordered_map<std::string, Statef> last_state_;                    ///< Last state for each unique link (accessed by state_key := model_name + link_name)
    std::unordered_map<std::string, Statef> last_state_dot_;                ///< Last state dot for each unique link (accessed by state_key := model_name + link_name)
    std::unordered_map<std::string, Eigen::VectorXf> processed_action_;     ///< Processed action for each unique link (accessed by state_key := model_name + link_name)
    Stated desired_state_;                                                  ///< Desired state
    int gz_partition_offset_base_;                                          ///< Gz partition offset
};

#ifdef USE_XLA
#define MAKE_PY_GAZEBO_ENVPOOL(m, EnvClass, SpecClass)                               \
    {                                                                                \
        using PyEnvPool_ = PyEnvPool<AsyncEnvPool<EnvClass>>;                        \
        using PyEnvSpec_ = PyEnvSpec<SpecClass>;                                     \
        py::class_<PyEnvSpec_>(m, "_GazeboSpec" #SpecClass, py::metaclass(abc_meta)) \
            .def(py::init<const typename PyEnvSpec_::ConfigValues &>())              \
            .def_readonly("_config_values", &PyEnvSpec_::py_config_values)           \
            .def_readonly("_state_spec", &PyEnvSpec_::py_state_spec)                 \
            .def_readonly("_action_spec", &PyEnvSpec_::py_action_spec)               \
            .def_readonly_static("_state_keys", &PyEnvSpec_::py_state_keys)          \
            .def_readonly_static("_action_keys", &PyEnvSpec_::py_action_keys)        \
            .def_readonly_static("_config_keys", &PyEnvSpec_::py_config_keys)        \
            .def_readonly_static("_default_config_values",                           \
                                 &PyEnvSpec_::py_default_config_values);             \
        py::class_<PyEnvPool_>(m, "_GazeboPool" #EnvClass, py::metaclass(abc_meta))  \
            .def(py::init<const PyEnvSpec_ &>())                                     \
            .def_readonly("_spec", &PyEnvPool_::py_spec)                             \
            .def("_recv", &PyEnvPool_::PyRecv)                                       \
            .def("_send", &PyEnvPool_::PySend)                                       \
            .def("_reset", &PyEnvPool_::PyReset)                                     \
            .def_readonly_static("_state_keys", &PyEnvPool_::py_state_keys)          \
            .def_readonly_static("_action_keys", &PyEnvPool_::py_action_keys)        \
            .def("_xla", &PyEnvPool_::Xla);                                          \
    }
#else
#define MAKE_PY_GAZEBO_ENVPOOL(m, EnvClass, SpecClass)                               \
    {                                                                                \
        using PyEnvPool_ = PyEnvPool<AsyncEnvPool<EnvClass>>;                        \
        using PyEnvSpec_ = PyEnvSpec<SpecClass>;                                     \
        py::class_<PyEnvSpec_>(m, "_GazeboSpec" #SpecClass, py::metaclass(abc_meta)) \
            .def(py::init<const typename PyEnvSpec_::ConfigValues &>())              \
            .def_readonly("_config_values", &PyEnvSpec_::py_config_values)           \
            .def_readonly("_state_spec", &PyEnvSpec_::py_state_spec)                 \
            .def_readonly("_action_spec", &PyEnvSpec_::py_action_spec)               \
            .def_readonly_static("_state_keys", &PyEnvSpec_::py_state_keys)          \
            .def_readonly_static("_action_keys", &PyEnvSpec_::py_action_keys)        \
            .def_readonly_static("_config_keys", &PyEnvSpec_::py_config_keys)        \
            .def_readonly_static("_default_config_values",                           \
                                 &PyEnvSpec_::py_default_config_values);             \
        py::class_<PyEnvPool_>(m, "_GazeboPool" #EnvClass, py::metaclass(abc_meta))  \
            .def(py::init<const PyEnvSpec_ &>())                                     \
            .def_readonly("_spec", &PyEnvPool_::py_spec)                             \
            .def("_recv", &PyEnvPool_::PyRecv)                                       \
            .def("_send", &PyEnvPool_::PySend)                                       \
            .def("_reset", &PyEnvPool_::PyReset)                                     \
            .def_readonly_static("_state_keys", &PyEnvPool_::py_state_keys)          \
            .def_readonly_static("_action_keys", &PyEnvPool_::py_action_keys);       \
    }
#endif // USE_XLA

#define REGISTER_PROCESSOR(ProcessorCls, SpecCls, ...)                         \
    {                                                                          \
        using ProcessorClsType = ProcessorCls<SpecCls>;                        \
        py::class_<ProcessorClsType>(m, #ProcessorCls)                         \
            .def(py::init<__VA_ARGS__>())                                      \
            .def("process_observation", &ProcessorClsType::ProcessObservation, \
                 py::call_guard<py::gil_scoped_release>())                     \
            .def("process_action", &ProcessorClsType::ProcessAction,           \
                 py::call_guard<py::gil_scoped_release>())                     \
            .def("compute_reward", &ProcessorClsType::ComputeReward,           \
                 py::call_guard<py::gil_scoped_release>())                     \
            .def("reset", &ProcessorClsType::Reset,                            \
                 py::call_guard<py::gil_scoped_release>());                    \
    }
#endif // GAZEBO_ENVPOOL_HH
