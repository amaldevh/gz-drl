// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef ROS_RL_SERVER_HH_
#define ROS_RL_SERVER_HH_
#include "rl_server.hh"
#include <gz/math/eigen3/Conversions.hh>
using LidarFrameView = systems::custom_plugins::Sensors::LidarFrameView;
#include "print_utils.hh"
#if ROS_VER == 1
#include "ros/ros.h"
#include "ros/spinner.h"
#include "std_msgs/Float32MultiArray.h"
#include "ros/wall_timer.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Twist.h"
#include "geometry_msgs/Vector3.h"
#include "geometry_msgs/Wrench.h"
#include "geometry_msgs/Inertia.h"
#include "nav_msgs/Odometry.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/CameraInfo.h"
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

using FloatArrayMsg = std_msgs::Float32MultiArray;
using PoseMsg = geometry_msgs::Pose;
using TwistMsg = geometry_msgs::Twist;
using Vector3Msg = geometry_msgs::Vector3;
using WrenchMsg = geometry_msgs::Wrench;
using InertiaMsg = geometry_msgs::Inertia;
using ImageMsg = sensor_msgs::Image;
using CameraInfoMsg = sensor_msgs::CameraInfo;
using PointCloudMsg = sensor_msgs::PointCloud2;
using OdomMsg = nav_msgs::Odometry;
using TransformStampedMsg = geometry_msgs::TransformStamped;
using FloatArrayMsgPtr = const std_msgs::Float32MultiArray::ConstPtr &;
using OdomMsgPtr = const nav_msgs::Odometry::ConstPtr &;
using ImageMsgPtr = const sensor_msgs::Image::ConstPtr &;
using CameraInfoPtr = const sensor_msgs::CameraInfo::ConstPtr &;
using PointCloudMsgPtr = const sensor_msgs::PointCloud2::ConstPtr &;
using PoseMsgPtr = const geometry_msgs::Pose::ConstPtr &;
using TwistMsgPtr = const geometry_msgs::Twist::ConstPtr &;
using Vector3MsgPtr = const geometry_msgs::Vector3::ConstPtr &;
using WrenchMsgPtr = const geometry_msgs::Wrench::ConstPtr &;
using InertiaMsgPtr = const geometry_msgs::Inertia::ConstPtr &;
using TransformStampedMsgPtr = const geometry_msgs::TransformStamped::ConstPtr &;
#elif ROS_VER == 2
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/executor.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "geometry_msgs/msg/wrench.hpp"
#include "geometry_msgs/msg/inertia.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/timer.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2/LinearMath/Quaternion.h"

using FloatArrayMsg = std_msgs::msg::Float32MultiArray;
using PoseMsg = geometry_msgs::msg::Pose;
using TwistMsg = geometry_msgs::msg::Twist;
using Vector3Msg = geometry_msgs::msg::Vector3;
using WrenchMsg = geometry_msgs::msg::Wrench;
using InertiaMsg = geometry_msgs::msg::Inertia;
using ImageMsg = sensor_msgs::msg::Image;
using CameraInfoMsg = sensor_msgs::msg::CameraInfo;
using PointCloudMsg = sensor_msgs::msg::PointCloud2;
using OdomMsg = nav_msgs::msg::Odometry;
using TransformStampedMsg = geometry_msgs::msg::TransformStamped;
using FloatArrayMsgPtr = std::unique_ptr<std_msgs::msg::Float32MultiArray>;
using OdomMsgPtr = std::unique_ptr<nav_msgs::msg::Odometry>;
using ImageMsgPtr = std::unique_ptr<sensor_msgs::msg::Image>;
using CameraInfoPtr = std::unique_ptr<sensor_msgs::msg::CameraInfo>;
using PointCloudMsgPtr = std::unique_ptr<sensor_msgs::msg::PointCloud2>;
using PoseMsgPtr = std::unique_ptr<geometry_msgs::msg::Pose>;
using TwistMsgPtr = std::unique_ptr<geometry_msgs::msg::Twist>;
using Vector3MsgPtr = std::unique_ptr<geometry_msgs::msg::Vector3>;
using WrenchMsgPtr = std::unique_ptr<geometry_msgs::msg::Wrench>;
using InertiaMsgPtr = std::unique_ptr<geometry_msgs::msg::Inertia>;
using TransformStampedMsgPtr = std::unique_ptr<geometry_msgs::msg::TransformStamped>;
#endif
#include <atomic>
#include <memory>
#include <functional>
#include <mutex>
#include <type_traits>
#include <utility>
#include <thread>
#include <shared_mutex>
#include "converters.hh"

#define shared_ptr(T) std::shared_ptr<T>

using namespace std::chrono;

/** @brief A version independent wrapper for publisher

*/
#if ROS_VER == 2
/**
 * @brief A type invariant publisher wrapper
 *
 * @tparam Msg type of msg for the publisher
 */
template <typename Msg>
struct Publisher
{

    /**
     * @brief Constructor for Publisher
     *
     * @param func_ A function that returns the message to be published
     * @param node The rclcpp node
     * @param topic The topic to publish to
     * @param queue_sz The size of the publisher queue
     * @param dt The duration between publishes
     */
    Publisher(std::function<Msg()> func_,
              std::shared_ptr<rclcpp::Node> &node,
              std::string topic,
              size_t queue_sz,
              std::chrono::microseconds dt)
    {
        publisher_ = node->create_publisher<Msg>(topic, queue_sz);
        func = func_;
        timer_ = node->create_wall_timer(dt, [this]()
                                         { publisher_->publish(func()); });
    };
    /**
     * @brief Publisher just to store reference,
     * This wont run the timer, the publisher handle will be stored internally
     *
     * @param node rclcpp node
     * @param topic topic to publish on
     * @param queue_sz  queue size
     */
    Publisher(std::shared_ptr<rclcpp::Node> &node,
              std::string topic,
              size_t queue_sz)
    {
        publisher_ = node->create_publisher<Msg>(topic, queue_sz);
    };
    /**
     * @brief Get the Publisher object
     *
     * @return const std::shared_ptr<rclcpp::Publisher<Msg>>&
     */
    const std::shared_ptr<rclcpp::Publisher<Msg>> &GetPublisher() const
    {
        return publisher_;
    }
    /**
     * @brief Destroy the Publisher object
     *
     */
      ~Publisher()
      {
        if (timer_)
          timer_->cancel();
      }
    Publisher(const Publisher &) = delete;
    Publisher &operator=(const Publisher &) = delete;
    Publisher(Publisher &&) = default;
    Publisher &operator=(Publisher &&) = default;

private:
    rclcpp::TimerBase::SharedPtr timer_;                ///< timer
    std::shared_ptr<rclcpp::Publisher<Msg>> publisher_; ///< publisher
    std::function<Msg()> func;                          ///< publishing function
};
/**
 * @brief A version independent wrapper for subscriber
 *
 * @tparam Msg Msg type for subscription
 * @tparam MsgPtr the type recieved during callbacks
 */
template <typename Msg, typename MsgPtr>
struct Subscriber
{
    /**
     * @brief Constructor for Subscriber
     *
     * @param cb_ A function that will be called when a message is received
     * @param node The rclcpp node
     * @param topic The topic to subscribe to
     * @param queue_sz The size of the subscriber queue
     */
    Subscriber(std::function<void(MsgPtr)> cb_,
               std::shared_ptr<rclcpp::Node> &node,
               std::string topic,
               size_t queue_sz)
    {
        cb = cb_;
        subscription_ = node->create_subscription<Msg>(topic, queue_sz, cb);
    };

private:
    std::shared_ptr<rclcpp::Subscription<Msg>> subscription_; ///< subscriber
    std::function<void(MsgPtr)> cb;                           ///< callback
};
#elif ROS_VER == 1

template <typename Msg>
struct Publisher
{
    /** @brief Constructor for Publisher
     *
     * @param func_ A function that returns the message to be published
     * @param nh The ros node handle
     * @param topic The topic to publish to
     * @param queue_sz The size of the publisher queue
     * @param dt The duration between publishes
     */
    Publisher(std::function<Msg()> func_,
              std::shared_ptr<ros::NodeHandle> &nh,
              std::string topic,
              size_t queue_sz,
              std::chrono::microseconds dt)
    {
        publisher_ptr_ = std::make_shared<ros::Publisher>(
            nh->advertise<Msg>(topic, queue_sz));
        func = func_;
        ros::WallDuration duration(static_cast<double>(dt.count()) * 1e-6);
        timer_ = nh->createWallTimer(duration, [this](const ros::WallTimerEvent &event)
                                     { publisher_ptr_->publish(func()); });
    };

    /**
     * @brief Publisher just to store reference,
     * This wont run the timer, the publisher handle will be stored internally
     *
     * @param nh NodeHandle
     * @param topic topic for publication
     * @param queue_sz Queue size
     */
    Publisher(std::shared_ptr<ros::NodeHandle> &nh,
              std::string topic,
              size_t queue_sz)
    {
        publisher_ptr_ = std::make_shared<ros::Publisher>(
            nh->advertise<Msg>(topic, queue_sz));
    };

    /**
     * @brief Get the Publisher object
     *
     * @return const std::shared_ptr<ros::Publisher>&
     */
    const std::shared_ptr<ros::Publisher> &GetPublisher() const
    {
        return publisher_ptr_;
    }
    /**
     * @brief Destroy the Publisher object
     *
     */
    ~Publisher()
    {
        timer_.stop();
    }
    Publisher(const Publisher &) = delete;
    Publisher &operator=(const Publisher &) = delete;
    Publisher(Publisher &&) = default;
    Publisher &operator=(Publisher &&) = default;

private:
    ros::WallTimer timer_;                          ///< wall timer
    std::shared_ptr<ros::Publisher> publisher_ptr_; ///< pointer publisher, for API compat
    std::function<Msg()> func;                      ///< publishing function
};

/**
 * @brief A version independent wrapper for subscriber
 *
 * @tparam Msg Subscribed msg type
 * @tparam MsgPtr The msg type provided in callbacks
 */
template <typename Msg, typename MsgPtr>
struct Subscriber
{
    /**
     *  @brief Constructor for Subscriber
     *
     * @param cb_ A function that will be called when a message is received
     * @param nh The ros node handle
     * @param topic The topic to subscribe to
     * @param queue_sz The size of the subscriber queue
     */
    Subscriber(std::function<void(MsgPtr)> cb_,
               std::shared_ptr<ros::NodeHandle> &nh,
               std::string topic,
               size_t queue_sz)
    {
        cb = cb_;
        subscription_ = nh->subscribe<Msg>(topic, queue_sz, cb);
    };

private:
    ros::Subscriber subscription_;  ///< subscriber
    std::function<void(MsgPtr)> cb; ///< callback
};
#endif

/**
 *  @brief A ROS-based Deep Reinforcement Learning Server, which can also
be used for SITL applications.
*/
class RosDRLServer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    /**
     * @brief Constructor for RosDRLServer
     *
     * @param partition The Gazebo partition name
     * @param sdf_file The SDF file to load
     * @param model_names The names of the models to control
     * @param enable_sensors Flag to enable sensor interface
     * @param rtf The real-time factor for simulation
     * @param link_map The map of model names to their link names for sensor attachment
     */

    RosDRLServer(const std::string &partition,
                 const std::string &sdf_file,
                 const std::vector<std::string> &model_names,
                 bool enable_sensors, double rtf,
                 const std::unordered_map<std::string, std::vector<std::string>> &link_map = {});

    /** @brief Destructor for RosDRLServer
     */
    ~RosDRLServer();

    /** @brief Spin the ROS executor in the current thread
     */
    void Spin();
    /** @brief Start spinning the ROS executor in a background thread
     */
    void SpinAsync();

    /** @brief Run the simulation at the given real-time factor (specified at construction)
     *
     */
    void Run();

    /** @brief Pause the simulation
     */
    void Pause();

    /**
     * @brief Get the Published Topic Map object
     *
     * @return const std::unordered_map<std::string, std::vector<std::string>>&
     */
    const std::unordered_map<std::string, std::vector<std::string>> &GetPublishedTopicMap() const
    {
        return published_topic_map_;
    }
    /**
     * @brief Get the Subscribed Topic Map object
     *
     * @return const std::unordered_map<std::string, std::vector<std::string>>&
     */
    const std::unordered_map<std::string, std::vector<std::string>> &GetSubscribedTopicMap() const
    {
        return subscribed_topic_map_;
    }
    /**
     * @brief Get the underlying DRLServer instance
     *
     * @return std::shared_ptr<DRLServer>
     */
    std::shared_ptr<DRLServer> Server()
    {
        return server_;
    }

    /**
     * @brief Execute an operation while holding the same server mutex used by
     * ROS publishers and subscribers.
     *
     * Use this for native code that must access Server() while the ROS
     * executor is spinning. Returning references is intentionally forbidden
     * because they would outlive the lock.
     */
    template <typename Function>
    decltype(auto) WithServerLocked(Function &&function)
    {
        using Result = std::invoke_result_t<Function, DRLServer &>;
        static_assert(!std::is_reference_v<Result>,
                      "WithServerLocked callbacks must not return references");
        std::lock_guard<std::mutex> lock(server_mutex_);
        if constexpr (std::is_void_v<Result>)
        {
            std::invoke(std::forward<Function>(function), *server_);
        }
        else
        {
            return std::invoke(std::forward<Function>(function), *server_);
        }
    }

private:
#if ROS_VER == 2
    inline static std::mutex ctxt_mutex_;
    inline static std::size_t ctxt_users_{0};
    inline static bool owns_ctxt_{false};

    /**
     * @brief Initialize rclcpp context
     *
     */
    static void InitCtxt()
    {
        std::lock_guard<std::mutex> lock(ctxt_mutex_);
        int argc = 0;
        char **argv = nullptr;
        if (ctxt_users_ == 0 && !rclcpp::ok())
        {
            rclcpp::init(argc, argv);
            owns_ctxt_ = true;
        }
        ++ctxt_users_;
    };

    /** @brief Release this instance's reference to the global ROS 2 context. */
    static void ReleaseCtxt()
    {
        std::lock_guard<std::mutex> lock(ctxt_mutex_);
        if (ctxt_users_ == 0)
            return;
        --ctxt_users_;
        if (ctxt_users_ == 0)
        {
            if (owns_ctxt_ && rclcpp::ok())
                rclcpp::shutdown();
            owns_ctxt_ = false;
        }
    };

#elif ROS_VER == 1
    /** @brief Initialize ros context
     */
    static void InitCtxt()
    {
        int argc = 0;
        char **argv = nullptr;
        if (!ros::isInitialized())
            ros::init(argc, argv, "drl_server",
                      ros::init_options::AnonymousName);
    };

    /**
     * ROS 1 has one process-global context and does not support reliable
     * shutdown/reinitialization cycles.  GzDRL therefore never shuts down a
     * context that may also be used by the host application.
     */
    static void ReleaseCtxt() {};
#endif
    /**
     * @brief Setup the DRL server
     */
    void SetupDRLServer();
    /**
     *  @brief Create the ROS node
     *
     * @param name The name of the node
     * @param ns The namespace of the node
     */
    void MakeNode(std::string name, std::string ns);

    /**
     *  @brief Reset the ROS node
     */
    void ResetNode();
    /** @brief Helper function to create a shared pointer
     *
     * @tparam T The type of the object to create
     * @tparam Args The types of the arguments to pass to the constructor
     * @param args_ The arguments to pass to the constructor
     * @return std::shared_ptr<T> The shared pointer to the created object
     */
    template <typename T, typename... Args>
    inline std::shared_ptr<T> make_shared_(Args &&...args_)
    {
        return std::make_shared<T>(std::forward<Args>(args_)...);
    }

    std::shared_ptr<DRLServer> server_;                                  ///< DRLServer
    std::string sdf_file_;                                               ///< sdf file for initializing Gazebo
    std::vector<std::string> model_names_;                               ///< Model names
    std::string partition_;                                              ///< partition
    std::shared_ptr<std::thread> spin_t_;                                ///< Spinning thread
    std::atomic<bool> internal_spin_flag_{false};                        ///< Internal spinning flag
    double phys_dt_{0.0};                                                ///< Physics dt
    double rtf_{1.0};                                                    ///< rtf, when rtf = 1.0, the server is stepped 1/(physics_dt) times per second (subject to hardware)
    std::shared_ptr<std::thread> async_spin_t_;                          ///< Async spin thread
    bool enable_sensors_{false};                                         ///< If sensors is enabled
    std::mutex server_mutex_;                                            ///< Mutex to access drlserver, shared across publishers and subscribers
    std::unordered_map<std::string, std::vector<std::string>> link_map_; ///< Map of links, indexed as model_name -> link_name

#if ROS_VER == 1
    std::shared_ptr<ros::NodeHandle> node_;                         ///< Node handle shared ptr
    std::map<std::string, std::shared_ptr<ros::WallTimer>> timers_; ///< wall timers
    std::shared_ptr<ros::WallTimer> run_timer_;                     ///< Run timer for running server at rtf
    std::unique_ptr<ros::AsyncSpinner> async_spinner_;               ///< Non-blocking ROS 1 spinner
#elif ROS_VER == 2
    std::shared_ptr<rclcpp::Node> node_;                             ///< Node handle
    std::shared_ptr<rclcpp::executors::MultiThreadedExecutor> exec_; ///< ROS2 executor
    std::vector<rclcpp::TimerBase::SharedPtr> timers_;               ///< timers
    rclcpp::TimerBase::SharedPtr run_timer_;                         ///< Run timer for running server at rtf
#endif
    // tf2 broadcaster
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf2_broadcaster_; ///< TF2 broadcaster

    std::unordered_map<std::string, std::shared_ptr<Subscriber<FloatArrayMsg, FloatArrayMsgPtr>>> float_array_subs_; ///< subscribers for float array msgs
    std::unordered_map<std::string, std::shared_ptr<Subscriber<PoseMsg, PoseMsgPtr>>> pose_subs_;                    ///< subscribers for pose msg
    std::unordered_map<std::string, std::shared_ptr<Subscriber<TwistMsg, TwistMsgPtr>>> twist_subs_;                 ///< subscribers for twist msg
    std::unordered_map<std::string, std::shared_ptr<Subscriber<Vector3Msg, Vector3MsgPtr>>> vector3_subs_;           ///< subscribers for Vector3 msg
    std::unordered_map<std::string, std::shared_ptr<Subscriber<WrenchMsg, WrenchMsgPtr>>> wrench_subs_;              ///< subscribers for wrench msg
    std::unordered_map<std::string, std::shared_ptr<Subscriber<InertiaMsg, InertiaMsgPtr>>> inertia_subs_;           ///< subscribers for inertia msg
    std::unordered_map<std::string, std::shared_ptr<Publisher<ImageMsg>>> img_pubs_;                                 ///< publishers for image msg
    std::unordered_map<std::string, std::shared_ptr<Publisher<CameraInfoMsg>>> cam_info_pubs_;                       ///< publishers for camera info
    std::unordered_map<std::string, std::shared_ptr<Publisher<PointCloudMsg>>> pc_pubs_;                             ///< publishers for pointcloud2 msg
    std::unordered_map<std::string, std::shared_ptr<Publisher<OdomMsg>>> odom_pubs_;                                 ///< publishers for odometry msg

    // sensor msg maps, prevents reallocation on each publish
    std::unordered_map<std::string, ImageMsg> internal_img_msgs_;            ///< image msg map, for efficiency
    std::unordered_map<std::string, CameraInfoMsg> internal_cam_info_msgs_;  ///< camera info msg map, for efficiency
    std::unordered_map<std::string, gz::msgs::Image> internal_gz_img_msgs_;  ///< gz image msg map, for efficiency
    std::unordered_map<std::string, PointCloudMsg> internal_pc_msgs_;        ///< pointcloud2 msg map, for efficiency
    std::unordered_map<std::string, LidarFrameView> internal_gz_pc_msgs_;    ///< gz pointcloud2 msg map, for efficiency
    std::unordered_map<std::string, OdomMsg> internal_odom_msgs_;            ///< odometry msg map, for efficiency
    std::unordered_map<std::string, Stated *> internal_state_refs_;          ///< POinters to states
    std::unordered_map<std::string, TransformStampedMsg> internal_tf2_msgs_; ///< tf2 msg map, for efficiency

    // A topic map
    std::unordered_map<std::string, std::vector<std::string>> published_topic_map_;  ///< topics that are published by this server
    std::unordered_map<std::string, std::vector<std::string>> subscribed_topic_map_; ///< topics that are subscribed by this server
    // camera pose map (conjunction with camera_info)
    std::unordered_map<std::string, Eigen::Vector3d> camera_rel_pos_;     ///< Relative pos of camera w.r.t models base link, indexed by camera name
    std::unordered_map<std::string, Eigen::Quaterniond> camera_rel_quat_; ///< Relative orientation of camera w.r.t models base link, indexed by camera name
    std::unordered_map<std::string, Eigen::Vector3d> lidar_rel_pos_;     ///< Relative pos of lidar w.r.t models base link, indexed by lidar name
    std::unordered_map<std::string, Eigen::Quaterniond> lidar_rel_quat_; ///< Relative orientation of lidar w.r.t models base link, indexed by camera name

    // atomic double for elapsed sim time, updated in the timer callback
    std::atomic<double> elapsed_sim_time_{0.0};   ///< elapsed sim time
    std::chrono::nanoseconds calibrated_sleep_t_; ///< sleep time for the server running thread
    /**
     * @brief updates the elapsed simulation time */
    inline void UpdateElapsedSimTime()
    {
        elapsed_sim_time_.store(server_->sim_iterations() * phys_dt_);
    }
    /**
     * @brief Returns the elapsed sim time
     *
     * @return const double elapsed sim time
     */
    inline const double ElapsedSimTime() const
    {
        return elapsed_sim_time_.load();
    }
    /**
     * @brief Replaces all instances of a substring in a string with another string
     *
     * @param s_ base string
     * @param from string to be removed
     * @param to string to be replaced
     * @return std::string replaced string
     */
    std::string ReplaceAll(std::string &s_, std::string from, std::string to)
    {
        std::string s = s_;
        if (from.empty())
            return s;
        std::size_t pos = 0;
        while ((pos = s.find(from, pos)) != std::string::npos)
        {
            s.replace(pos, from.size(), to);
            pos += to.size();
        }
        return s;
    }
};

#endif
