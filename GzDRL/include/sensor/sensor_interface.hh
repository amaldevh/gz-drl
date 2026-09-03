/*
 * Copyright (C) 2018 Open Source Robotics Foundation
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
#ifndef GZ_SIM_SYSTEMS_SENSORS_HH_
#define GZ_SIM_SYSTEMS_SENSORS_HH_

#include <memory>
#include <string>

#include <gz/sim/config.hh>
#include <gz/sim/System.hh>
#include <sdf/Sensor.hh>
#include <gz/msgs/image.pb.h>
#include <gz/msgs/camera_info.pb.h>
#include <gz/sensors/CameraSensor.hh>
#include <gz/sensors/GpuLidarSensor.hh>

namespace gz
{
    namespace sim
    {
        // Inline bracket to help doxygen filtering.
        inline namespace GZ_SIM_VERSION_NAMESPACE
        {
            namespace systems
            {
                namespace custom_plugins
                {
                    // Forward declarations.
                    class SensorsPrivate;

                    /// \class Sensors Sensors.hh gz/sim/systems/Sensors.hh
                    /// \brief A system that manages sensors.
                    ///
                    /// ## System Parameters
                    ///
                    /// - `<render_engine>`: Name of the render engine, such as "ogre" or "ogre2".
                    /// - `<background_color>`: Color used for the scene's background. This
                    /// will override the background color specified in a world's SDF <scene>
                    /// element. This background color is used by sensors, not the GUI.
                    /// - `<ambient_light>`: Color used for the scene's ambient light. This
                    /// will override the ambient value specified in a world's SDF <scene>
                    /// element. This ambient light is used by sensors, not the GUI.
                    /// - `<disable_on_drained_battery>`: Disable sensors if the model's
                    /// battery plugin charge reaches zero. Sensors that are in nested
                    /// models are also affected.
                    ///
                    /// \TODO(louise) Have one system for all sensors, or one per
                    /// sensor / sensor type?
                    class Sensors : public System,
                                    public ISystemConfigure,
                                    public ISystemReset,
                                    public ISystemUpdate,
                                    public ISystemPostUpdate
                    {
                        /// \brief Constructor
                    public:
                        explicit Sensors();

                        /// \brief Destructor
                    public:
                        ~Sensors() override;

                        // Documentation inherited
                    public:
                        void Configure(const Entity &_id,
                                       const std::shared_ptr<const sdf::Element> &_sdf,
                                       EntityComponentManager &_ecm,
                                       EventManager &_eventMgr) final;

                        /// Documentation inherited
                    public:
                        void Reset(const UpdateInfo &_info,
                                   EntityComponentManager &_ecm) final;

                        // Documentation inherited
                    public:
                        void Update(const UpdateInfo &_info,
                                    EntityComponentManager &_ecm) final;

                        // Documentation inherited
                    public:
                        void PostUpdate(const UpdateInfo &_info,
                                        const EntityComponentManager &_ecm) final;

                        /// \brief Create a rendering sensor from sdf
                        /// \param[in] _entity Entity of the sensor
                        /// \param[in] _sdf SDF description of the sensor
                        /// \param[in] _parentName Name of parent that the sensor is attached to
                        /// \return Sensor name
                    private:
                        std::string CreateSensor(const Entity &_entity,
                                                 const sdf::Sensor &_sdf,
                                                 const std::string &_parentName);

                        /// \brief Removes a rendering sensor
                        /// \param[in] _entity Entity of the sensor
                    private:
                        void RemoveSensor(const Entity &_entity);

                        /// \brief Private data pointer.
                    private:
                        std::unique_ptr<SensorsPrivate> dataPtr;

                        /// Customized API for DRL applications
                        /// \brief Camera/depth sensor callback
                        /// \param[in] img Image message (RGB or depth data)
                        /// \param[in] name Sensor name
                        /// \details Called from rendering thread when sensor updates.
                        ///          Stores image in image_data_map for retrieval.
                    private:
                        void ImageSensorCB(const gz::msgs::Image &img, std::string name);

                        /// \brief GPU lidar sensor callback
                        /// \param[in] scan Pointer to scan data (owned by sensor)
                        /// \param[in] width Scan width (points per scan line)
                        /// \param[in] height Scan height (number of scan lines)
                        /// \param[in] channels Data channels per point (typically 3: x, y, z)
                        /// \param[in] subscriber Subscriber name (unused)
                        /// \param[in] name Sensor name
                        /// \details Called from rendering thread. Stores non-owning pointer in
                        ///          gpu_lidar_data_map. Pointer valid only until next sensor update.
                    private:
                        void LidarSensorCB(const float *scan, unsigned int width, unsigned int height,
                                           unsigned int channels, const std::string &subscriber,
                                           std::string name);

                    private:
                        void DepthImageSensorCB(const float *_data, unsigned int _width, unsigned int _height,
                                                unsigned int /*_channels*/, const std::string & /*_format*/, std::string name);
                        // =====================================================================
                        // DATA STORAGE
                        // =====================================================================

                        /// Connection pointers for all bound sensors (kept alive for callbacks)
                    private:
                        std::unordered_map<std::string, gz::common::ConnectionPtr> conn_ptrs;

                        /// Camera sensor pointers (for direct access if needed)
                    private:
                        std::unordered_map<std::string, sensors::CameraSensor *> cam_ptrs;

                        /// GPU lidar sensor pointers (for direct access if needed)
                    private:
                        std::unordered_map<std::string, sensors::GpuLidarSensor *> gpu_lidar_ptrs;

                        /// Map of sensor names to latest image data
                    private:
                        std::unordered_map<std::string, gz::msgs::Image> image_data_map;

                        /// Map of CameraInfos
                    private:
                        std::unordered_map<std::string, gz::msgs::CameraInfo> camera_info_data_map;

                        /// Map of Camera Poses
                    private:
                        std::unordered_map<std::string, gz::math::Pose3d> camera_poses;
                        /// Map of Lidar poses
                    private:
                        std::unordered_map<std::string, gz::math::Pose3d> lidar_poses;

                        /// Map of depth cam index
                    private:
                        std::unordered_map<std::string, int> depth_cam_idx;

                        /// \brief Lightweight view into GPU lidar scan data
                        /// \details Non-owning pointer to scan data with dimensions
                    public:
                        struct LidarFrameView
                        {
                            const float *data;    ///< Scan data pointer (owned by sensor)
                            unsigned int w, h, c; ///< Width, height, channels
                            /// \brief Beam angle bounds (radians), needed to convert the polar
                            /// range image into Cartesian points. Horizontal = azimuth (per column),
                            /// vertical = elevation (per row).
                            float angle_min, angle_max;           ///< Horizontal scan bounds
                            float vert_angle_min, vert_angle_max; ///< Vertical scan bounds
                        };

                        /// Map of sensor names to latest lidar data views
                    private:
                        std::unordered_map<std::string, LidarFrameView> gpu_lidar_data_map;

                        /// Map of sensor names to enable/disable flags
                        // private: std::unordered_map<std::string, bool> requested_sensor_map;

                        // =====================================================================
                        // SENSOR QUERY METHODS
                        // =====================================================================

                        /// \brief Print all sensor names to console
                        /// \details Useful for discovering available sensors in the world
                    public:
                        void PrintSensorNames()
                        {
                            for (auto *s : this->GetSensors())
                                std::cout << s->Name() << "\n";
                        };
                        /// \brief Get all sensors in the world
                        /// \return Vector of sensor pointers
                    private:
                        std::vector<sensors::Sensor *> GetSensors();
                        /// \brief Get list of all camera sensor names
                        /// \return Vector of camera sensor names (Camera, DepthCamera, RGBD, etc.)
                    public:
                        std::vector<std::string> CameraSensorNames();
                        /// \brief Get list of all lidar sensor names
                        /// \return Vector of lidar sensor names (GpuLidar, Lidar)
                    public:
                        std::vector<std::string> LidarSensorNames();
                        /// \brief Bind image callback for a specific sensor
                    private:
                        std::unordered_map<std::string, std::vector<std::function<void(const gz::msgs::Image &)>>> img_cbs;

                        /// \brief Bind image callback for a specific sensor
                        /// \param[in] cb Callback function to bind
                        /// \param[in] name Sensor name
                        /// \details Registers a user-defined callback to be invoked when new
                        ///          image data is available from the specified sensor.
                    public:
                        void BindImgCB(std::function<void(const gz::msgs::Image &)> cb, std::string_view name)
                        {
                            std::lock_guard<std::mutex> l(img_mutex);
                            auto name_str = std::string(name);
                            if (img_cbs.find(name_str) == img_cbs.end())
                                img_cbs[name_str] = std::vector<std::function<void(const gz::msgs::Image &)>>();
                            img_cbs[name_str].push_back(cb);
                        };

                        /// \brief Map of sensor names to lidar callbacks
                        /// \details Each sensor name maps to a vector of callback functions
                    private:
                        std::unordered_map<std::string, std::vector<std::function<void(const LidarFrameView &)>>> lidar_cbs;

                        /// \brief Bind lidar callback for a specific sensor
                        /// \param[in] cb Callback function to bind
                        /// \param[in] name Sensor name
                        /// \details Registers a user-defined callback to be invoked when new
                        ///          lidar data is available from the specified sensor.
                    public:
                        void BindLidarCB(std::function<void(const LidarFrameView &)> cb, std::string_view name)
                        {
                            std::lock_guard<std::mutex> l(lidar_mutex);
                            auto name_str = std::string(name);
                            if (lidar_cbs.find(name_str) == lidar_cbs.end())
                                lidar_cbs[name_str] = std::vector<std::function<void(const LidarFrameView &)>>();
                            lidar_cbs[name_str].push_back(cb);
                        };

                        /// \brief Mutexes for thread-safe data access
                    private:
                        std::mutex img_mutex;

                    private:
                        std::mutex lidar_mutex;

                        /// \brief Get latest image data for a sensor
                        /// \param[in] name Sensor name
                        /// \return Image message (empty if not found)
                    public:
                        gz::msgs::Image GetSensorImg(std::string_view name)
                        {
                            auto name_str = std::string(name);
                            std::lock_guard<std::mutex> l(img_mutex);
                            if (image_data_map.find(name_str) != image_data_map.end())
                                return (image_data_map[name_str]);
                            return gz::msgs::Image();
                        };

                        /// \brief Get latest GPU lidar data for a sensor
                        /// \param[in] name Sensor name
                        /// \return LidarFrameView struct with scan data (nullptr if not found)
                    public:
                        LidarFrameView GetSensorGpuLidar(std::string_view name)
                        {
                            auto name_str = std::string(name);
                            std::lock_guard<std::mutex> l(lidar_mutex);
                            if (gpu_lidar_data_map.find(name_str) != gpu_lidar_data_map.end())
                                return gpu_lidar_data_map[name_str];
                            return {nullptr, 0, 0, 0};
                        }

                        /// \brief Get the CameraInfo for a camera
                        /// \param[in] name Sensor name
                        /// \return gz::msgs::CameraInfo
                    public:
                        gz::msgs::CameraInfo GetCameraInfo(std::string_view name)
                        {
                            auto name_str = std::string(name);
                            if (camera_info_data_map.find(name_str) != camera_info_data_map.end())
                                return camera_info_data_map[name_str];
                            return gz::msgs::CameraInfo();
                        }

                        /// \brief Get the Pose for a camera (relative to body of the model)
                        /// \param[in] name Sensor name
                        /// \return gz::math::Pose3d
                    public:
                        gz::math::Pose3d GetCameraPose(std::string_view name)
                        {
                            auto name_str = std::string(name);
                            if (camera_poses.find(name_str) != camera_poses.end())
                                return camera_poses[name_str];
                            return gz::math::Pose3d();
                        }
                        
                        /// \brief Get the Pose for a lidar (relative to body of the model)
                        /// \param[in] name Sensor name
                        /// \return gz::math::Pose3d
                    public:
                        gz::math::Pose3d GetLidarPose(std::string_view name)
                        {
                            auto name_str = std::string(name);
                            if (lidar_poses.find(name_str) != lidar_poses.end())
                                return lidar_poses[name_str];
                            return gz::math::Pose3d();
                        }
                        /// \brief stops the sensor system
                    public:
                        void Stop();
                    /// \brief adds video recording functionality
                    public: 
                    void StartCameraRecording(std::string cam_name, int height, int width, int fps,
                        const math::Pose3d& initial_pose, 
                        std::string output_file);
                    /// \brief Updates the camera pose
                    public:
                    void UpdateCameraRecordingPose(std::string cam_name, const math::Pose3d& pose);
                    /// \brief stops camera recording and write the output
                    public:
                    void StopCameraRecording(std::string cam_name);
                    /// \brief whether a recording start is still pending
                    /// (camera not created on the render thread yet)
                    public:
                    bool CameraRecordingQueued(const std::string& cam_name);
                    /// \brief whether a recording is queued or actively
                    /// encoding (false once the video file is finalized)
                    public:
                    bool CameraRecordingBusy(const std::string& cam_name);
                        
                    };
                }
            }

        }
    }
}
#endif