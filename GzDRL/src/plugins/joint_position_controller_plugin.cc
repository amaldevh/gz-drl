/*
 * Copyright (C) 2020 Open Source Robotics Foundation
 * Copyright (C) 2023 Benjamin Perseghetti, Rudis Laboratories
 * Copyright (C) 2026 Amaldev Haridevan, York University
 *
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

#include "plugins/joint_position_controller_plugin.hh"
#include "common/common.hh"

#include <gz/msgs/actuators.pb.h>
#include <gz/msgs/double.pb.h>

#include <chrono>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include <gz/common/Profiler.hh>
#include <gz/math/PID.hh>
#include <gz/plugin/Register.hh>
#include <gz/transport/Node.hh>
#include <gz/transport/parameters/Registry.hh>

#include "gz/sim/components/Actuators.hh"
#include "gz/sim/components/Joint.hh"
#include "gz/sim/components/JointForceCmd.hh"
#include "gz/sim/components/JointVelocityCmd.hh"
#include "gz/sim/components/JointPosition.hh"
#include "gz/sim/Model.hh"
#include "gz/sim/Util.hh"

using namespace gz;
using namespace sim;
using namespace systems;

//////////////////////////////////////////////////
//! \brief Utility to declare and update a parameter owned by another object
//////////////////////////////////////////////////
class gz::sim::systems::JointPositionControllerPrivate
{

  /// \brief Joint Entity
  public: std::vector<Entity> jointEntities;

  /// \brief Joint name
  public: std::vector<std::string> jointNames;

  /// \brief Commanded joint position
  public: double jointPosCmd{0.0};

  /// \brief Index of position actuator.
  public: int actuatorNumber = 0;



  /// \brief Is the maximum PID gain set.
  public: bool isMaxSet {false};

  /// \brief Model interface
  public: Model model{kNullEntity};

  /// \brief True if using Actuator msg to control joint position.
  public: bool useActuatorMsg{false};

  /// \brief Position PID controller.
  public: math::PID posPid;

  /// \brief Joint index to be used.
  public: unsigned int jointIndex = 0u;

  /// \brief Operation modes
  enum OperationMode
  {
    /// \brief Use PID to achieve positional control
    PID,
    /// \brief Bypass PID completely. This means the joint will move to that
    /// position bypassing the physics engine.
    ABS
  };

  /// \brief Joint position mode
  public: OperationMode mode = OperationMode::PID;

  /// Dynamic parameters
  public: double pGain;//{"p_gain"};
  public: double iGain;//{"i_gain"};
  public: double dGain;//{"d_gain"};
  public: double iMax;//{"i_max"};
  public: double iMin;//{"i_min"};
  public: double cmdMax;//{"cmd_max"};
  public: double cmdMin;//{"cmd_min"};
  public: double cmdOffset;//{"cmd_offset"};
};

//////////////////////////////////////////////////
JointPositionController::JointPositionController()
  : dataPtr(std::make_unique<JointPositionControllerPrivate>())
{
}

//////////////////////////////////////////////////
void JointPositionController::Configure(const Entity &_entity,
    const std::shared_ptr<const sdf::Element> &_sdf,
    EntityComponentManager &_ecm,
    EventManager &/*_eventMgr*/)
{
  this->dataPtr->model = Model(_entity);

  if (!this->dataPtr->model.Valid(_ecm))
  {
    gzerr << "JointPositionController plugin should be attached to a model "
           << "entity. Failed to initialize." << std::endl;
    return;
  }

  // Get params from SDF
  auto sdfElem = _sdf->FindElement("joint_name");
  while (sdfElem)
  {
    if (!sdfElem->Get<std::string>().empty())
    {
      this->dataPtr->jointNames.push_back(sdfElem->Get<std::string>());
    }
    else
    {
      gzerr << "<joint_name> provided but is empty." << std::endl;
    }
    sdfElem = sdfElem->GetNextElement("joint_name");
  }
  if (this->dataPtr->jointNames.empty())
  {
    gzerr << "Failed to get any <joint_name>." << std::endl;
    return;
  }

  if (_sdf->HasElement("joint_index"))
  {
    this->dataPtr->jointIndex = _sdf->Get<unsigned int>("joint_index");
  }
  else{
     gzerr << "Failed to get any <joint_index>." << std::endl;
    return;
  }

  // PID parameters
  double p         =  1;
  double i         =  0.1;
  double d         =  0.01;
  double iMax      =  1;
  double iMin      = -1;
  double cmdMax    =  1000;
  double cmdMin    = -1000;
  double cmdOffset =  0;

  if (_sdf->HasElement("p_gain"))
  {
    p = _sdf->Get<double>("p_gain");
  }
  if (_sdf->HasElement("i_gain"))
  {
    i = _sdf->Get<double>("i_gain");
  }
  if (_sdf->HasElement("d_gain"))
  {
    d = _sdf->Get<double>("d_gain");
  }
  if (_sdf->HasElement("i_max"))
  {
    iMax = _sdf->Get<double>("i_max");
  }
  if (_sdf->HasElement("i_min"))
  {
    iMin = _sdf->Get<double>("i_min");
  }
  if (_sdf->HasElement("cmd_max"))
  {
    cmdMax = _sdf->Get<double>("cmd_max");
    this->dataPtr->isMaxSet = true;
  }
  if (_sdf->HasElement("cmd_min"))
  {
    cmdMin = _sdf->Get<double>("cmd_min");
  }
  if (_sdf->HasElement("cmd_offset"))
  {
    cmdOffset = _sdf->Get<double>("cmd_offset");
  }
  if (_sdf->HasElement("use_velocity_commands"))
  {
    auto useVelocityCommands = _sdf->Get<bool>("use_velocity_commands");
    if (useVelocityCommands)
    {
      this->dataPtr->mode =
        JointPositionControllerPrivate::OperationMode::ABS;
    }
  }

  this->dataPtr->posPid.Init(p, i, d, iMax, iMin, cmdMax, cmdMin, cmdOffset);


  if (_sdf->HasElement("initial_position"))
  {
    this->dataPtr->jointPosCmd = _sdf->Get<double>("initial_position");
  }

  if (_sdf->HasElement("use_actuator_msg") &&
    _sdf->Get<bool>("use_actuator_msg"))
  {
    if (_sdf->HasElement("actuator_number"))
    {
      this->dataPtr->actuatorNumber =
        _sdf->Get<int>("actuator_number");
      this->dataPtr->useActuatorMsg = true;
    }
    else
    {
      gzerr << "Please specify an actuator_number" <<
        "to use Actuator position message control." << std::endl;
    }
  }

}

//////////////////////////////////////////////////
void JointPositionController::ConfigureParameters(
    gz::transport::parameters::ParametersRegistry &_registry,
    gz::sim::EntityComponentManager &_ecm)
{

  // If the joints haven't been identified yet, look for them
  if (this->dataPtr->jointEntities.empty())
  {
    bool warned{false};
    for (const std::string &name : this->dataPtr->jointNames)
    {
      // First try to resolve by scoped name.
      Entity joint = kNullEntity;
      auto entities = entitiesFromScopedName(
          name, _ecm, this->dataPtr->model.Entity());

      if (!entities.empty())
      {
        if (entities.size() > 1)
        {
          gzwarn << "Multiple joint entities with name ["
                << name << "] found. "
                << "Using the first one.\n";
        }
        joint = *entities.begin();

        // Validate
        if (!_ecm.EntityHasComponentType(joint, components::Joint::typeId))
        {
          gzerr << "Entity with name[" << name
                << "] is not a joint\n";
          joint = kNullEntity;
        }
        else
        {
          gzdbg << "Identified joint [" << name
                << "] as Entity [" << joint << "]\n";
        }
      }

      if (joint != kNullEntity)
      {
        this->dataPtr->jointEntities.push_back(joint);
      }
      else if (!warned)
      {
        gzwarn << "Failed to find joint [" << name << "]\n";
        warned = true;
      }
    }
  }

  if (this->dataPtr->jointEntities.empty())
    return;

  // Use the first joint to create the scoped parameter names
  std::string scopedName = gz::sim::scopedName(
    this->dataPtr->jointEntities[0], _ecm, ".", false);
  std::string prefix = std::string("JointPositionController.") + scopedName
    + std::string(".");

  //! @note not using gz::msgs::PID because the message does not support all
  //!       fields available in gz::math::PID (cmd_max, cmd_min, cmd_offset)
}

//////////////////////////////////////////////////
void JointPositionController::PreUpdate(
    const UpdateInfo &_info,
    EntityComponentManager &_ecm)
{

  // \TODO(anyone) This is a temporary fix for
  // gazebosim/gz-sim#2165 until gazebosim/gz-sim#2217 is resolved.
  if (kNullEntity == this->dataPtr->model.Entity())
  {
    return;
  }

  if (!this->dataPtr->model.Valid(_ecm))
  {
    gzwarn << "JointPositionController model no longer valid. "
           << "Disabling plugin." << std::endl;
    this->dataPtr->model = Model(kNullEntity);
    return;
  }

  // \TODO(anyone) Support rewind
  if (_info.dt < std::chrono::steady_clock::duration::zero())
  {
    gzwarn << "Detected jump back in time ["
           << std::chrono::duration<double>(_info.dt).count()
           << "s]. System may not work properly." << std::endl;
  }

  if (this->dataPtr->jointEntities.empty())
    return;

  // Nothing left to do if paused.
  if (_info.paused)
    return;

  // Create joint position component if one doesn't exist
  auto jointPosComp = _ecm.Component<components::JointPosition>(
      this->dataPtr->jointEntities[0]);
  if (!jointPosComp)
  {
    _ecm.CreateComponent(this->dataPtr->jointEntities[0],
        components::JointPosition());
  }

  // We just created the joint position component, give one iteration for the
  // physics system to update its size
  if (jointPosComp == nullptr || jointPosComp->Data().empty())
    return;

  // Sanity check: Make sure that the joint index is valid.
  if (this->dataPtr->jointIndex >= jointPosComp->Data().size())
  {
    static std::unordered_set<Entity> reported;
    if (reported.find(this->dataPtr->jointEntities[0]) == reported.end())
    {
      gzerr << "[JointPositionController]: Detected an invalid <joint_index> "
             << "parameter. The index specified is ["
             << this->dataPtr->jointIndex << "] but joint ["
             << this->dataPtr->jointNames[0] << "] only has ["
             << jointPosComp->Data().size() << "] index[es]. "
             << "This controller will be ignored" << std::endl;
      reported.insert(this->dataPtr->jointEntities[0]);
    }
    return;
  }

  // Get error in position
  auto jointPosCmdComp = _ecm.Component<JointPositionCmdComp>( this->dataPtr->jointEntities[0]);
  if (jointPosCmdComp){this->dataPtr->jointPosCmd = jointPosCmdComp->Data().at(this->dataPtr->jointIndex);}
  // std::cout<<"cmd: "<<this->dataPtr->jointPosCmd<<"\n";
  double error;
  {
    error = jointPosComp->Data().at(this->dataPtr->jointIndex) -
            this->dataPtr->jointPosCmd;
  }

  // Check if the mode is ABS
  if (this->dataPtr->mode ==
    JointPositionControllerPrivate::OperationMode::ABS)
  {
    // Calculate target velcity
    double targetVel = 0;

    // Get time in seconds
    auto dt = std::chrono::duration<double>(_info.dt).count();

    // Get the maximum amount in m that this joint may move
    auto maxMovement = this->dataPtr->posPid.CmdMax() * dt;

    // Limit the maximum change to maxMovement
    if (abs(error) > maxMovement && this->dataPtr->isMaxSet)
    {
      targetVel = (error < 0) ? this->dataPtr->posPid.CmdMax() :
        -this->dataPtr->posPid.CmdMax();
    }
    else
    {
      targetVel = - error / dt;
    }
    for (Entity joint : this->dataPtr->jointEntities)
    {
      // Update velocity command.
      _ecm.SetComponentData<components::JointVelocityCmd>(joint, {targetVel});
    }
    return;
  }

  for (Entity joint : this->dataPtr->jointEntities)
  {
    // Update force command.
    double force = this->dataPtr->posPid.Update(error, _info.dt);

    auto forceComp =
        _ecm.Component<components::JointForceCmd>(joint);
    if (forceComp == nullptr)
    {
      _ecm.CreateComponent(joint,
                          components::JointForceCmd({force}));
    }
    else
    {
      *forceComp = components::JointForceCmd({force});
    }
  }
}


GZ_ADD_PLUGIN(JointPositionController,
                    System,
                    JointPositionController::ISystemConfigure,
                    JointPositionController::ISystemConfigureParameters,
                    JointPositionController::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(JointPositionController,
                          "gz::sim::systems::JointPositionController")