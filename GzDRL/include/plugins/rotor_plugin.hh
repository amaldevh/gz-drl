/*
 * Copyright (C) 2019 Open Source Robotics Foundation
 * Copyright (C) 2026 Amal Dev Haridevan, SDCNLab, York University, Toronto, ON
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
#ifndef ROTOR_PLUGIN_HH_
#define ROTOR_PLUGIN_HH_

#include <gz/sim/System.hh>
#include <memory>

namespace gz
{
  namespace sim
  {
    // Inline bracket to help doxygen filtering.
    inline namespace GZ_SIM_VERSION_NAMESPACE
    {
      namespace systems
      {
        // Forward declaration
        class MultiRotorPluginPrivate;

        /// \brief This system applies a thrust force to models with spinning
        /// propellers. See examples/worlds/quadcopter.sdf for a demonstration.
        class MultiRotorPlugin
            : public System,
              public ISystemConfigure,
              public ISystemPreUpdate
        {
        public:
          MultiRotorPlugin();

        public:
          ~MultiRotorPlugin() override = default;

          // Documentation inherited
        public:
          void Configure(const Entity &_entity,
                         const std::shared_ptr<const sdf::Element> &_sdf,
                         EntityComponentManager &_ecm,
                         EventManager &_eventMgr) override;

          // Documentation inherited
        public:
          void PreUpdate(const gz::sim::UpdateInfo &_info,
                         gz::sim::EntityComponentManager &_ecm) override;

        private:
          std::unique_ptr<MultiRotorPluginPrivate> dataPtr;
        };
      } // namespace systems
    } // namespace GZ_SIM_VERSION_NAMESPACE
  } // namespace sim
} // namespace gz

#endif // ROTOR_PLUGIN_HH_
