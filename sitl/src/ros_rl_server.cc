// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#include "ros_rl_server.hh"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;

RosDRLServer::RosDRLServer(const std::string &partition,
                           const std::string &sdf_file,
                           const std::vector<std::string> &model_names,
                           bool enable_sensors,
                           double rtf,
                           const std::unordered_map<std::string, std::vector<std::string>> &link_map)
    : sdf_file_(sdf_file),
      model_names_(model_names), partition_(partition), enable_sensors_(enable_sensors),
      rtf_(rtf), link_map_(link_map)
{
  if (!std::isfinite(rtf_) || rtf_ <= 0.0)
    throw std::invalid_argument("rtf must be finite and greater than zero");

  server_ = std::make_shared<DRLServer>(partition, sdf_file_, model_names_, enable_sensors);
  elapsed_sim_time_ = 0.0;
  phys_dt_ = server_->step_size();
  const double timer_period_ns = phys_dt_ * 1e9 / rtf_;
  const double max_period_ns = static_cast<double>(
      std::numeric_limits<std::chrono::nanoseconds::rep>::max());
  if (!std::isfinite(timer_period_ns) || timer_period_ns < 1.0 ||
      timer_period_ns > max_period_ns)
    throw std::invalid_argument("rtf produces an invalid ROS timer period");
  calibrated_sleep_t_ = std::chrono::nanoseconds(
      static_cast<std::chrono::nanoseconds::rep>(timer_period_ns));

  RosDRLServer::InitCtxt();
  try
  {
    MakeNode("drl_server", "/server_partition_" + partition);
// setup tf2 broadcaster
#if ROS_VER == 1
    tf2_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>();
#elif ROS_VER == 2
    tf2_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(node_);
#endif
    SetupDRLServer();
    server_->run_N(10);

  // std::vector<double> step_dts;
  // step_dts.reserve(20);
  // for (int i = 0; i < 20; ++i) {
  //   const auto ts = std::chrono::high_resolution_clock::now();
  //   server_->run_N(1);
  //   UpdateElapsedSimTime();
  //   const auto tf = std::chrono::high_resolution_clock::now();
  //   step_dts.push_back(
  //       static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(tf - ts).count()));
  // }

  // double avg_dt = 0.0;
  // for (double v : step_dts) avg_dt += v;
  // avg_dt /= 20.0;

  // double std_dev = 0.0;
  // for (double v : step_dts) std_dev += std::pow(v - avg_dt, 2.0);
  // std_dev = std::pow(std_dev / 20.0, 0.5);
  // const auto avg_dt_ns = std::chrono::nanoseconds(static_cast<int>(avg_dt));
  // const auto calibrated_sleep_t = std::max(min_ns, (sleep_t_ns - avg_dt_ns));

  // print_info("Physics step size: ",phys_step, ", Steps / sec: ", steps_per_sec,
  //   ", thread sleep time (ns): ", sleep_t_ns.count(), ", avg step time (ns): ", avg_dt_ns.count());
  // print_info("Avg server run time/step: ", avg_dt * 1e-9,
  //     ", Std dev for time/step: ", std_dev * 1e-9,
  //      ", Calibrated thread sleep time (s): ", calibrated_sleep_t.count() * 1e-9);
  }
  catch (...)
  {
    ResetNode();
    RosDRLServer::ReleaseCtxt();
    throw;
  }
}

void RosDRLServer::Spin()
{
  if (async_spin_t_ && async_spin_t_->joinable())
  {
    async_spin_t_->join();
    return;
  };
#if ROS_VER == 1
  if (async_spinner_)
    ros::waitForShutdown();
  else
    ros::spin();
  return;
#elif ROS_VER == 2
  if (exec_)
  {
    if (exec_->is_spinning())
      exec_->cancel();
    exec_->spin();
  }
#endif
}

void RosDRLServer::SpinAsync()
{
#if ROS_VER == 1
  if (!async_spinner_)
  {
    async_spinner_ = std::make_unique<ros::AsyncSpinner>(0);
    async_spinner_->start();
  }
  return;
#elif ROS_VER == 2
  if (async_spin_t_ && async_spin_t_->joinable())
    return;
  async_spin_t_.reset();
  async_spin_t_ = std::make_shared<std::thread>([this]()
                                                {
  if (exec_) {
    if (exec_->is_spinning()) exec_->cancel();
    exec_->spin();
  } });
#endif
}

void RosDRLServer::ResetNode()
{

#if ROS_VER == 1
  if (async_spinner_)
  {
    async_spinner_->stop();
    async_spinner_.reset();
  }
#elif ROS_VER == 2
  if (exec_)
    exec_->cancel();
  if (async_spin_t_ && async_spin_t_->joinable())
  {
    async_spin_t_->join();
  }
  if (exec_ && node_)
  {
    try
    {
      exec_->remove_node(node_);
    }
    catch (const std::exception &)
    {
      // The node may already have been removed during executor shutdown.
    }
  }
  async_spin_t_.reset();
#endif
  timers_.clear();
  run_timer_.reset();
  float_array_subs_.clear();
  pose_subs_.clear();
  twist_subs_.clear();
  wrench_subs_.clear();
  vector3_subs_.clear();
  inertia_subs_.clear();
  img_pubs_.clear();
  cam_info_pubs_.clear();
  pc_pubs_.clear();
  odom_pubs_.clear();
  internal_img_msgs_.clear();
  internal_cam_info_msgs_.clear();
  internal_pc_msgs_.clear();
  internal_odom_msgs_.clear();
  internal_tf2_msgs_.clear();
  internal_state_refs_.clear();
  internal_gz_pc_msgs_.clear();
  internal_gz_img_msgs_.clear();
  tf2_broadcaster_.reset();
  node_.reset();
#if ROS_VER == 2
  exec_.reset();
#endif
}

void RosDRLServer::MakeNode(std::string name, std::string ns)
{
  if (node_)
  {
    print_err("Node already exists!");
    throw std::runtime_error("");
  }
#if ROS_VER == 1
  node_ = std::make_shared<ros::NodeHandle>(ns);
#elif ROS_VER == 2
  node_ = std::make_shared<rclcpp::Node>(name, ns);
  exec_ = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
  exec_->add_node(node_);
#endif
}

void RosDRLServer::Pause()
{
  if (run_timer_)
  {
#if ROS_VER == 1
    run_timer_->stop();
#elif ROS_VER == 2
    run_timer_->cancel();
#endif
  }
}

void RosDRLServer::Run()
{
  Pause();
#if ROS_VER == 1
  auto timer_fnc = [this](const ros::WallTimerEvent &event)
  {
    {
      std::lock_guard<std::mutex> l(server_mutex_);
      server_->run_N(1);
      UpdateElapsedSimTime();
    }
  };
  run_timer_ = std::make_shared<ros::WallTimer>(node_->createWallTimer(
      ros::WallDuration(static_cast<double>(calibrated_sleep_t_.count()) * 1e-9), timer_fnc));
#elif ROS_VER == 2
  auto timer_fnc = [this]()
  {
    {
      std::lock_guard<std::mutex> l(server_mutex_);
      server_->run_N(1);
      UpdateElapsedSimTime();
    }
  };
  run_timer_ = node_->create_wall_timer(calibrated_sleep_t_, timer_fnc);
#endif
}

RosDRLServer::~RosDRLServer()
{
  ResetNode();
  RosDRLServer::ReleaseCtxt();
}

void RosDRLServer::SetupDRLServer()
{
  const std::string namespace_ = partition_.empty() ? "" : ("/server_partition_" + partition_);
  server_->run_N(10);
  std::vector<std::string> cam_sensors;
  std::vector<std::string> lidar_sensors;
  if (enable_sensors_)
  {
    cam_sensors = server_->camera_sensor_names();
    lidar_sensors = server_->lidar_sensor_names();
    // Need to reset models to ensure sensors are properly initialized
    // server_->update_control_states();
    // for (const auto &model_name : model_names_){
    //     const auto& cs = std::get<0>(server_->control_states[model_name].begin()->second);
    //     Eigen::Vector3d pos = cs.segment<3>(0);
    //     Eigen::Quaterniond quat(cs(6), cs(7), cs(8), cs(9));
    //     Eigen::Vector3d euler = quat.normalized().toRotationMatrix().eulerAngles(0, 1, 2);
    //     server_->reset_pos(model_name, std::move(pos), std::move(euler));
    // }
    // server_->run_N(100);
  }
  // For each model, we need to create subscribers for each command type
  // The subscribers will only be created for the Canonical Link (a.k.a base link),
  // Therfore, the user MUST ensure that any plugins
  // that process commands are attached to the Canonical Link (a.k.a base link)
  const auto world_sdf = server_->internal_sys->sdf_world_();
  for (const auto &model_name : model_names_)
  {
    auto model_ptr = world_sdf.ModelByName(model_name);
    if (!model_ptr)
    {
      print_err("Model sdf could not be found for model: ", model_name);
      throw std::runtime_error("");
    }
    auto link_ptr = model_ptr->CanonicalLink();
    if (!link_ptr)
    {
      print_err("Canonical link sdf could not be found for model: ", model_name);
      throw std::runtime_error("");
    }
    const std::string link_name = link_ptr->Name();
    const std::string base_topic = namespace_ + "/" + model_name + "/" + link_name;

    // Create subscriber for each command type
    // set rotor_velocity cmd
    {
      auto callback = [this, model_name, link_name](FloatArrayMsgPtr msg)
      {
        Eigen::VectorXd cmd;
        cmd.resize(msg->data.size());
        for (size_t i = 0; i < msg->data.size(); ++i)
          cmd(i) = static_cast<double>(msg->data[i]);
        {
          std::lock_guard<std::mutex> l(server_mutex_);
          server_->set_rotor_velocity_cmd(model_name, link_name, std::move(cmd));
        }
      };
      float_array_subs_[model_name + "::" + link_name + "_rotor_velocity_cmd"] =
          std::make_shared<Subscriber<FloatArrayMsg, FloatArrayMsgPtr>>(callback,
                                                                        node_,
                                                                        base_topic + "/rotor_velocity_cmd",
                                                                        1);
      subscribed_topic_map_[model_name].push_back(base_topic + "/rotor_velocity_cmd");
    }
    // set velocity cmd
    {
      auto callback = [this, model_name, link_name](Vector3MsgPtr msg)
      {
        Eigen::Vector3d cmd(msg->x, msg->y, msg->z);
        {
          std::lock_guard<std::mutex> l(server_mutex_);
          server_->set_velocity_cmd(model_name, link_name, std::move(cmd));
        }
      };
      vector3_subs_[model_name + "::" + link_name + "_velocity_cmd"] =
          std::make_shared<Subscriber<Vector3Msg, Vector3MsgPtr>>(callback,
                                                                  node_,
                                                                  base_topic + "/velocity_cmd",
                                                                  1);
      subscribed_topic_map_[model_name].push_back(base_topic + "/velocity_cmd");
    }
    // set angular velocity cmd
    {
      auto callback = [this, model_name, link_name](Vector3MsgPtr msg)
      {
        Eigen::Vector3d cmd(msg->x, msg->y, msg->z);
        {
          std::lock_guard<std::mutex> l(server_mutex_);
          server_->set_angular_velocity_cmd(model_name, link_name, std::move(cmd));
        }
      };
      vector3_subs_[model_name + "::" + link_name + "_angular_velocity_cmd"] =
          std::make_shared<Subscriber<Vector3Msg, Vector3MsgPtr>>(callback,
                                                                  node_,
                                                                  base_topic + "/angular_velocity_cmd",
                                                                  1);
      subscribed_topic_map_[model_name].push_back(base_topic + "/angular_velocity_cmd");
    }
    // set joint position cmd
    {
      auto callback = [this, model_name, link_name](Vector3MsgPtr msg)
      {
        Eigen::Vector3d cmd(msg->x, msg->y, msg->z);
        {
          std::lock_guard<std::mutex> l(server_mutex_);
          server_->set_joint_position_cmd(model_name, link_name, std::move(cmd));
        }
      };
      vector3_subs_[model_name + "::" + link_name + "_joint_position_cmd"] =
          std::make_shared<Subscriber<Vector3Msg, Vector3MsgPtr>>(callback,
                                                                  node_,
                                                                  base_topic + "/joint_position_cmd",
                                                                  1);
      subscribed_topic_map_[model_name].push_back(base_topic + "/joint_position_cmd");
    }
    // set ackermann velocity cmd
    {
      auto callback = [this, model_name, link_name](TwistMsgPtr msg)
      {
        Eigen::Vector2d cmd(msg->linear.x, msg->angular.z);
        {
          std::lock_guard<std::mutex> l(server_mutex_);
          server_->set_ackermann_velocity_cmd(model_name, link_name, std::move(cmd));
        }
      };
      twist_subs_[model_name + "::" + link_name + "_ackermann_velocity_cmd"] =
          std::make_shared<Subscriber<TwistMsg, TwistMsgPtr>>(callback,
                                                              node_,
                                                              base_topic + "/ackermann_velocity_cmd",
                                                              1);
      subscribed_topic_map_[model_name].push_back(base_topic + "/ackermann_velocity_cmd");
    }
    // set reset pose cmd
    {
      auto callback = [this, model_name](PoseMsgPtr msg)
      {
        Eigen::Vector3d pos{msg->position.x, msg->position.y, msg->position.z};
        Eigen::Quaterniond quat(msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z);
        Eigen::Vector3d euler = quat.normalized().toRotationMatrix().eulerAngles(2, 1, 0);
        euler << euler(2), euler(1), euler(0);
        {
          std::lock_guard<std::mutex> l(server_mutex_);
          server_->reset_pos(model_name, std::move(pos), std::move(euler));
          server_->run_N(10);
        }
      };
      pose_subs_[model_name + "::reset_pose"] =
          std::make_shared<Subscriber<PoseMsg, PoseMsgPtr>>(callback,
                                                            node_,
                                                            namespace_ + "/" + model_name + "/reset_pose",
                                                            1);
      subscribed_topic_map_[model_name].push_back(namespace_ + "/" + model_name + "/reset_pose");
    }
    // set mass and inertia
    {
      auto cb = [this, model_name, link_name](InertiaMsgPtr msg)
      {
        Eigen::Matrix3d inertia;
        inertia << msg->ixx, msg->ixy, msg->ixz,
            msg->ixy, msg->iyy, msg->iyz,
            msg->ixz, msg->iyz, msg->izz;
        {
          std::lock_guard<std::mutex> l(server_mutex_);
          server_->set_inertia(model_name, link_name, std::move(inertia));
          server_->set_mass(model_name, link_name, msg->m);
        }
      };
      inertia_subs_[model_name + "::" + link_name + "_set_inertia"] =
          std::make_shared<Subscriber<InertiaMsg, InertiaMsgPtr>>(cb, node_,
                                                                  base_topic + "/set_inertia", 1);
      subscribed_topic_map_[model_name].push_back(base_topic + "/set_inertia");
    }
    // set wrench
    {
      auto cb = [this, model_name, link_name](WrenchMsgPtr msg)
      {
        Eigen::Vector3d force(msg->force.x, msg->force.y, msg->force.z);
        Eigen::Vector3d moments(msg->torque.x, msg->torque.y, msg->torque.z);
        {
          std::lock_guard<std::mutex> l(server_mutex_);
          server_->set_wrench(model_name, link_name, std::move(force), std::move(moments));
        }
      };
      wrench_subs_[model_name + "::" + link_name + "_set_wrench"] =
          std::make_shared<Subscriber<WrenchMsg, WrenchMsgPtr>>(cb, node_,
                                                                base_topic + "/set_wrench", 1);
      subscribed_topic_map_[model_name].push_back(base_topic + "/set_wrench");
    }
    // publisher for odometry
    {
      const std::string index_key = model_name + "::" + link_name;
      internal_odom_msgs_[index_key] = OdomMsg();
      internal_state_refs_[index_key] = &std::get<0>(server_->control_states[model_name][link_name]);
      internal_odom_msgs_[index_key].child_frame_id = link_name;
      internal_odom_msgs_[index_key].header.frame_id = "map";

      auto pub_callback = [this, model_name, link_name, index_key]()
      {
        double dt = ElapsedSimTime();
        Stated cs;
        {
          std::lock_guard<std::mutex> l(server_mutex_);
          server_->update_control_states();
          cs = *internal_state_refs_[index_key];
        }
        double nano_sec = (std::fmod(dt, 1.0));
        double sec = dt - nano_sec;
        OdomMsg &msg = internal_odom_msgs_[index_key];
        msg.header.stamp.sec = static_cast<int>(sec);
#if ROS_VER == 1
        msg.header.stamp.nsec = static_cast<int>(nano_sec * 1e9);
#elif ROS_VER == 2
        msg.header.stamp.nanosec = static_cast<int>(nano_sec * 1e9);
#endif
        // position
        msg.pose.pose.position.x = cs(0);
        msg.pose.pose.position.y = cs(1);
        msg.pose.pose.position.z = cs(2);

        // orientation (cs has w,x,y,z but ROS wants x,y,z,w)
        msg.pose.pose.orientation.x = cs(7);
        msg.pose.pose.orientation.y = cs(8);
        msg.pose.pose.orientation.z = cs(9);
        msg.pose.pose.orientation.w = cs(6);

        // linear velocity
        msg.twist.twist.linear.x = cs(3);
        msg.twist.twist.linear.y = cs(4);
        msg.twist.twist.linear.z = cs(5);

        // angular velocity
        msg.twist.twist.angular.x = cs(10);
        msg.twist.twist.angular.y = cs(11);
        msg.twist.twist.angular.z = cs(12);
        return msg;
      };
      odom_pubs_[index_key] = std::make_shared<Publisher<OdomMsg>>(pub_callback,
                                                                   node_,
                                                                   base_topic + "/odom", 1,
                                                                   std::chrono::microseconds(static_cast<unsigned long>(phys_dt_ * 1e6 / rtf_)));
      published_topic_map_[model_name].push_back(base_topic + "/odom");
    }
    // publishers for each link in link_map
    {
      if (link_map_.find(model_name) != link_map_.end() && link_map_[model_name].size() != 0)
      {
        for (const auto &link_name_map : link_map_[model_name])
        {
          if (link_name_map == link_name)
            continue;
          auto link_ptr = model_ptr->LinkByName(link_name_map);
          if (!link_ptr)
          {
            print_err("link sdf for link name " + link_name_map + " could not be found, something went wrong");
            throw std::runtime_error("");
          }
          const std::string index_key = model_name + "::" + link_name_map;
          internal_odom_msgs_[index_key] = OdomMsg();
          internal_state_refs_[index_key] = &std::get<0>(server_->control_states[model_name][link_name_map]);
          internal_odom_msgs_[index_key].child_frame_id = link_name_map;
          internal_odom_msgs_[index_key].header.frame_id = "map";
          auto pub_callback = [this, model_name, link_name_map, index_key]()
          {
            double dt = ElapsedSimTime();
            Stated cs;
            {
              std::lock_guard<std::mutex> l(server_mutex_);
              server_->update_control_states();
              cs = *internal_state_refs_[index_key];
            }
            double nano_sec = (std::fmod(dt, 1.0));
            double sec = dt - nano_sec;
            OdomMsg &msg = internal_odom_msgs_[index_key];
            msg.header.stamp.sec = static_cast<int>(sec);
#if ROS_VER == 1
            msg.header.stamp.nsec = static_cast<int>(nano_sec * 1e9);
#elif ROS_VER == 2
            msg.header.stamp.nanosec = static_cast<int>(nano_sec * 1e9);
#endif
            // position
            msg.pose.pose.position.x = cs(0);
            msg.pose.pose.position.y = cs(1);
            msg.pose.pose.position.z = cs(2);
            // orientation (cs has w,x,y,z but ROS wants x,y,z,w)
            msg.pose.pose.orientation.x = cs(7);
            msg.pose.pose.orientation.y = cs(8);
            msg.pose.pose.orientation.z = cs(9);
            msg.pose.pose.orientation.w = cs(6);
            // linear velocity
            msg.twist.twist.linear.x = cs(3);
            msg.twist.twist.linear.y = cs(4);
            msg.twist.twist.linear.z = cs(5);
            // angular velocity
            msg.twist.twist.angular.x = cs(10);
            msg.twist.twist.angular.y = cs(11);
            msg.twist.twist.angular.z = cs(12);
            return msg;
          };
          odom_pubs_[index_key] = std::make_shared<Publisher<OdomMsg>>(pub_callback,
                                                                       node_,
                                                                       namespace_ + "/" + model_name + "/" + link_name_map + "/odom",
                                                                       1, std::chrono::microseconds(static_cast<unsigned long>(phys_dt_ * 1e6 / rtf_)));
          published_topic_map_[model_name].push_back(namespace_ + "/" + model_name + "/" + link_name_map + "/odom");
        }
      }
    }
    // Publisher for each sensor if enabled
    if (enable_sensors_)
    {
      for (auto &cam_name : cam_sensors)
      {
        auto cam_name_strip = namespace_ + "/" + ReplaceAll(cam_name, "::", "_");
        internal_img_msgs_[cam_name] = ImageMsg();
        internal_img_msgs_[cam_name].header.frame_id = cam_name;
        internal_gz_img_msgs_[cam_name] = gz::msgs::Image();
        img_pubs_[cam_name] = std::make_shared<Publisher<ImageMsg>>(node_,
                                                                    cam_name_strip + "/image",
                                                                    1);
        // camera info publisher
        internal_cam_info_msgs_[cam_name] = CameraInfoMsg();
        internal_cam_info_msgs_[cam_name].header.frame_id = cam_name;
        cam_info_pubs_[cam_name] = std::make_shared<Publisher<CameraInfoMsg>>(node_,
                                                                              cam_name_strip + "/camera_info",
                                                                              1);
        // tf2 broadcaster
        internal_tf2_msgs_[cam_name] = TransformStampedMsg();
        internal_tf2_msgs_[cam_name].header.frame_id = "map";
        internal_tf2_msgs_[cam_name].child_frame_id = cam_name;
        // Populate camera info msg
        gz::msgs::CameraInfo gz_cam_info_msg;
        {
          std::lock_guard<std::mutex> l(server_mutex_);
          gz_cam_info_msg = server_->get_camera_info(cam_name);
          auto pose3d = server_->get_camera_pose(cam_name);
          camera_rel_pos_[cam_name] = gz::math::eigen3::convert(pose3d.Pos());
          camera_rel_quat_[cam_name] = gz::math::eigen3::convert(pose3d.Rot());
        }
        std::copy(gz_cam_info_msg.intrinsics().k().begin(),
                  gz_cam_info_msg.intrinsics().k().end(), internal_cam_info_msgs_[cam_name].k.begin());
        internal_cam_info_msgs_[cam_name].d.resize(5);
        std::copy(gz_cam_info_msg.distortion().k().begin(),
                  gz_cam_info_msg.distortion().k().end(), internal_cam_info_msgs_[cam_name].d.begin());
        std::copy(gz_cam_info_msg.projection().p().begin(),
                  gz_cam_info_msg.projection().p().end(), internal_cam_info_msgs_[cam_name].p.begin());

        published_topic_map_[model_name].push_back(cam_name_strip + "/image");
        published_topic_map_[model_name].push_back(cam_name_strip + "/camera_info");

        std::shared_ptr<Publisher<ImageMsg>> pub_ptr = img_pubs_[cam_name];
        std::shared_ptr<Publisher<CameraInfoMsg>> info_pub_ptr = cam_info_pubs_[cam_name];

        auto pub_callback = [this, cam_name, pub_ptr, info_pub_ptr, model_name, link_name](const gz::msgs::Image &img_new)
        {
          double dt = ElapsedSimTime();
          internal_gz_img_msgs_[cam_name] = img_new;
          gz::msgs::Image &img = internal_gz_img_msgs_[cam_name];
          ImageMsg &msg = internal_img_msgs_[cam_name];
          CameraInfoMsg &info_msg = internal_cam_info_msgs_[cam_name];
          TransformStampedMsg &tf2_msg = internal_tf2_msgs_[cam_name];
          sensor_converters::convert_img(img, msg);
          double nano_sec = (std::fmod(dt, 1.0));
          double sec = dt - nano_sec;
          msg.header.stamp.sec = static_cast<int>(sec);
          info_msg.header.stamp.sec = static_cast<int>(sec);
          tf2_msg.header.stamp.sec = static_cast<int>(sec);
#if ROS_VER == 1
          msg.header.stamp.nsec = static_cast<int>(nano_sec * 1e9);
          info_msg.header.stamp.nsec = static_cast<int>(nano_sec * 1e9);
          tf2_msg.header.stamp.nsec = static_cast<int>(nano_sec * 1e9);
#elif ROS_VER == 2
          msg.header.stamp.nanosec = static_cast<int>(nano_sec * 1e9);
          info_msg.header.stamp.nanosec = static_cast<int>(nano_sec * 1e9);
          tf2_msg.header.stamp.nanosec = static_cast<int>(nano_sec * 1e9);
#endif
          pub_ptr->GetPublisher()->publish(msg);
          info_pub_ptr->GetPublisher()->publish(info_msg);
          // tf2 pub
          std::string index_key = model_name + "::" + link_name;
          const Stated &state = ((internal_state_refs_[index_key] == nullptr) ? Stated::Zero() : *internal_state_refs_[index_key]);
          Eigen::Vector3d &cam_rel_pos = camera_rel_pos_[cam_name];
          Eigen::Quaterniond &cam_rel_quat = camera_rel_quat_[cam_name];
          Eigen::Quaterniond quat(state(6), state(7), state(8), state(9));
          Eigen::Vector3d pos = state.segment<3>(0) + quat * cam_rel_pos;
          quat = quat * cam_rel_quat;
          internal_tf2_msgs_[cam_name].transform.translation.x = pos(0);
          internal_tf2_msgs_[cam_name].transform.translation.y = pos(1);
          internal_tf2_msgs_[cam_name].transform.translation.z = pos(2);
          internal_tf2_msgs_[cam_name].transform.rotation.x = quat.x();
          internal_tf2_msgs_[cam_name].transform.rotation.y = quat.y();
          internal_tf2_msgs_[cam_name].transform.rotation.z = quat.z();
          internal_tf2_msgs_[cam_name].transform.rotation.w = quat.w();
          tf2_broadcaster_->sendTransform(internal_tf2_msgs_[cam_name]);
        };
        {
          std::lock_guard<std::mutex> l(server_mutex_);
          server_->bind_img_cb(pub_callback, cam_name);
        }
      }
      for (auto &lidar_name : lidar_sensors)
      {
        auto lidar_name_strip = namespace_ + "/" + ReplaceAll(lidar_name, "::", "_");
        internal_pc_msgs_[lidar_name] = PointCloudMsg();
        internal_pc_msgs_[lidar_name].header.frame_id = lidar_name;
        internal_gz_pc_msgs_[lidar_name] = LidarFrameView{nullptr, 0, 0, 0};
        pc_pubs_[lidar_name] = std::make_shared<Publisher<PointCloudMsg>>(node_,
                                                                          lidar_name_strip,
                                                                          1);
        published_topic_map_[model_name].push_back(lidar_name_strip);
        std::shared_ptr<Publisher<PointCloudMsg>> pub_ptr = pc_pubs_[lidar_name];
        // setup tf2
        // tf2 broadcaster
        internal_tf2_msgs_[lidar_name] = TransformStampedMsg();
        internal_tf2_msgs_[lidar_name].header.frame_id = "map";
        internal_tf2_msgs_[lidar_name].child_frame_id = lidar_name;
        auto pose3d = server_->get_lidar_pose(lidar_name);
        lidar_rel_pos_[lidar_name] = gz::math::eigen3::convert(pose3d.Pos());
        lidar_rel_quat_[lidar_name] = gz::math::eigen3::convert(pose3d.Rot());

        auto pub_callback = [this, lidar_name, pub_ptr, model_name, link_name](const LidarFrameView &cloud_new)
        {
          double dt = ElapsedSimTime();
          internal_gz_pc_msgs_[lidar_name] = cloud_new;
          LidarFrameView &cloud = internal_gz_pc_msgs_[lidar_name];
          PointCloudMsg &msg = internal_pc_msgs_[lidar_name];
          TransformStampedMsg &tf2_msg = internal_tf2_msgs_[lidar_name];
          sensor_converters::convert_lidar(cloud, msg);
          double nano_sec = (std::fmod(dt, 1.0));
          double sec = dt - nano_sec;
          msg.header.stamp.sec = static_cast<int>(sec);
          tf2_msg.header.stamp.sec = static_cast<int>(sec);
#if ROS_VER == 1
          msg.header.stamp.nsec = static_cast<int>(nano_sec * 1e9);
          tf2_msg.header.stamp.nsec = static_cast<int>(nano_sec * 1e9);
#elif ROS_VER == 2
          msg.header.stamp.nanosec = static_cast<int>(nano_sec * 1e9);
          tf2_msg.header.stamp.nanosec = static_cast<int>(nano_sec * 1e9);
#endif
          pub_ptr->GetPublisher()->publish(msg);
          // tf2 pub
          std::string index_key = model_name + "::" + link_name;
          const Stated &state = ((internal_state_refs_[index_key] == nullptr) ? Stated::Zero() : *internal_state_refs_[index_key]);
          Eigen::Vector3d &lidar_rel_pos = lidar_rel_pos_[lidar_name];
          Eigen::Quaterniond &lidar_rel_quat = lidar_rel_quat_[lidar_name];
          Eigen::Quaterniond quat(state(6), state(7), state(8), state(9));
          Eigen::Vector3d pos = state.segment<3>(0) + quat * lidar_rel_pos;
          quat = quat * lidar_rel_quat;
          internal_tf2_msgs_[lidar_name].transform.translation.x = pos(0);
          internal_tf2_msgs_[lidar_name].transform.translation.y = pos(1);
          internal_tf2_msgs_[lidar_name].transform.translation.z = pos(2);
          internal_tf2_msgs_[lidar_name].transform.rotation.x = quat.x();
          internal_tf2_msgs_[lidar_name].transform.rotation.y = quat.y();
          internal_tf2_msgs_[lidar_name].transform.rotation.z = quat.z();
          internal_tf2_msgs_[lidar_name].transform.rotation.w = quat.w();
          tf2_broadcaster_->sendTransform(internal_tf2_msgs_[lidar_name]);
        };
        {
          std::lock_guard<std::mutex> l(server_mutex_);
          server_->bind_lidar_cb(pub_callback, lidar_name);
        }
      }
    }
  }
}
