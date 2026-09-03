# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

# This code provides an interfacing for DRL envs run via ROS2
# Note that these environments violate common assumptions taken in
# RL such as synchronous agent-env interaction
# This is also used for comparison of our raw DRL server
# NOTE: this is for showcasing the limitations of default Gazebo software for RL
# proper way to use DRLSErver for RL is through our DRLServer or AsyncDRLServer, that
# directly interfaces Gazebo without sockets
# A use case for this type of ROs2 envs is when you want to test SITL or HITL stuff
import rclpy 
import gzdrl
from gzdrl.sitl import RosDRLServer
from multiprocessing import Process
import multiprocessing
import os 
from multiprocessing import Queue, Process
import signal
import time
from threading import (Lock, Thread)
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Twist, Vector3, Wrench, Inertia
from std_msgs.msg import Float32MultiArray
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from scipy.spatial.transform import Rotation

def _update_env(env_var, val):
    os.environ.update({env_var: os.environ.get(env_var, "")+":"+str(val)})


_plugin_path = gzdrl.get_plugin_path()
_update_env("GZ_SIM_RESOURCE_PATH", gzdrl.get_sdf_path())
_update_env("LD_LIBRARY_PATH", _plugin_path)
_update_env("GZ_SIM_SYSTEM_PLUGIN_PATH", _plugin_path)

def _server_proc(
    ros2_server_partition: str,
    ros2_env_sdf: str,
    model_names,
    use_sensors: bool,
    rtf: float,
    link_map: dict
) -> None:
    try:
        rclpy.init()
        serv = RosDRLServer(
            ros2_server_partition, ros2_env_sdf, model_names, use_sensors, rtf, link_map )
        # If run() blocks and spins internally, this is sufficient; otherwise spin explicitly.
        serv.run()
        # If your server requires spinning, prefer a blocking spin to avoid busy-loops.
        serv.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass



class BaseRos2DRLEnv:
    
    def __init__(self, ros2_server_partition,  model_names, canonical_link_names, link_map={}, 
                 construct_ros2_env=False, ros2_env_sdf='', rtf=1.0, use_sensors=False, sep_proc=False):
        # NOTE Server partition should be same as the RosDRLServer partition
    
        if construct_ros2_env and not sep_proc:
            self.serv = RosDRLServer(ros2_server_partition, ros2_env_sdf, model_names, use_sensors, 
                                     rtf, link_map)
        self._base_reset_msg = dict()
        self.model_names = model_names
        self.canonical_link_names = canonical_link_names
        self.link_map = link_map.copy()

        if not rclpy.ok():
            rclpy.init()
        self._node =    Node("drl_server", namespace=ros2_server_partition)#Node(partition, namespace="")
        self.server_partition = ros2_server_partition
        self._publishers = dict()
        self._reset_publishers = dict()
        self._make_publishers()
        self._subscribers = dict()
        self._state_locks = dict()
        self._model_states = dict()
        self._make_subscribers()
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spint = Thread(target= lambda: self._executor.spin() )
        self._spint.start()
        if construct_ros2_env and not sep_proc:
            self.serv.run()
        if construct_ros2_env and sep_proc:
            ctx = multiprocessing.get_context("spawn")
            self.serv = ctx.Process(
                target=_server_proc,
                args=(ros2_server_partition, ros2_env_sdf, model_names, 
                      use_sensors, rtf, link_map),
                daemon=True,
            )
            self.serv.start()
        self._wait_flag = True
        while self._wait_flag:
            # print("waiting")
            time.sleep(1e-2)
            
    def control_states(self, model_name):
        """Returns the control state (pos, vel, quat, and omega)
        NOTE: quat is (w, x, y, z) format (or scalar first format)"""
        # no runtime checks to make this very fast
        # if model_name is not in the default names, then this will error
        with self._state_locks[model_name]:
            return self._model_states[model_name].copy()
    
    def set_wrench(self, model_name, link_name, cmd):
        """Sets wrench on link on a model"""
        if len(cmd) != 6:
            raise RuntimeError("wrench command should have length 6 [Force, Moments]")
        msg = Wrench()
        msg.force.x = cmd[0]
        msg.force.y = cmd[1]
        msg.force.z = cmd[2]
        msg.torque.x = cmd[3]
        msg.torque.y = cmd[4]
        msg.torque.z = cmd[5]
        self._publishers[model_name][link_name]["wrench"].publish(msg)
    
    def set_set_ackermann_cmd(self, model_name, link_name, cmd):
        """Sets ackermann comamnds to be tracked by a controller to a model  or model/link,
        set link_name to '' to send to model"""
        if len(cmd) != 2:
            raise RuntimeError("ackermann command should have length 2 [velocity, omega]")
        msg = Twist()
        msg.linear.x = cmd[0]
        msg.angular.z = cmd[1]
        self._publishers[model_name][link_name]["ackermann_cmd"].publish(msg)
    
    def set_joint_position_cmd(self, model_name, link_name, cmd):
        """Sets joint position commands to be tracked by a joint controller for  model  or model/link
        ,
        set link_name to '' to send to model"""
        if len(cmd) != 3:
            raise RuntimeError("joint position command should have length 3 [ax0, ax1, ax3]")
        msg = Vector3()
        msg.x = cmd[0]
        msg.y = cmd[1]
        msg.z = cmd[2]
        self._publishers[model_name][link_name]["joint_position_cmd"].publish(msg)
    
    def set_rotor_velocity_cmd(self, model_name, link_name, cmd):
        """Sets rotor velocity commands to be tracked by rotorplugin on a model or model/link,
        set link_name to '' to send to model"""
        msg = Float32MultiArray()
        msg.data = cmd
        self._publishers[model_name][link_name]["rotor_velocity_cmd"].publish(msg)
    
    def set_velocity_cmd(self, model_name, link_name, cmd):
        """Sets velocity commands directly model or model/link,
        set link_name to '' to send to model"""
        if len(cmd) != 3:
            raise RuntimeError("verlocity command should have length 3 [vx, vy, vz]")
        msg = Vector3()
        msg.x = cmd[0]
        msg.y = cmd[1]
        msg.z = cmd[2]
        self._publishers[model_name][link_name]["velocity_cmd"].publish(msg)
    
    def set_inertia(self, model_name, link_name, mass, inertia):
        """Sets inertia and mass of a model/link,
        set link_name to '' to send to model"""
        if len(inertia)!=9 and not isinstance(inertia, (float)):
            raise RuntimeError("inertia should have 9 elements (ixx, ixy, ixz, ixy, iyy, iyz, ixy, iyz, izz) and mass should be scalar")
        msg = Inertia()
        msg.m = mass
        msg.ixx = inertia[0]
        msg.ixy = inertia[1]
        msg.ixz = inertia[2]    
        msg.iyy = inertia[3]
        msg.iyz = inertia[4]
        msg.izz = inertia[5]

        self._publishers[model_name][link_name]["set_inertia"].publish(msg)
    
    def reset_pose(self, model_name, position, orientation):
        if len(position) != 3 or len(orientation) != 3:
            raise RuntimeError("position and orientation should be 3d (x,y,z) and (r, p, y)")
        msg = Pose()
        msg.position.x = float(position[0])
        msg.position.y = float(position[1])
        msg.position.z = float(position[2])
        quat = Rotation.from_euler("ZYX", (orientation)).as_quat(scalar_first=True)
        msg.orientation.x = float(quat[1])
        msg.orientation.y = float(quat[2])
        msg.orientation.z = float(quat[3])
        msg.orientation.w = float(quat[0])
        self._reset_publishers[model_name].publish(msg)
        
    def _make_subscribers(self):
        if len(self._subscribers.keys()) :
            raise RuntimeError("subscribers are already made")
        for (model, canon_link) in zip(self.model_names, self.canonical_link_names):
            self._model_states[model] = dict()
            self._model_states[model][canon_link] = np.zeros((13), dtype=np.float64)
            self._subscribers[model] = dict()
            msg = Odometry()
            state_lock = Lock()
            self._state_locks[model] = state_lock
            # create cbs 
            def cb(odom_msg):
                self._wait_flag = False
                # print("got cb")
                position = [odom_msg.pose.pose.position.x,odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z] 
                quat = [odom_msg.pose.pose.orientation.w, odom_msg.pose.pose.orientation.x,odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z] 
                linear_vel = [odom_msg.twist.twist.linear.x,odom_msg.twist.twist.linear.y, odom_msg.twist.twist.linear.z]
                angular_vel = [odom_msg.twist.twist.angular.x,odom_msg.twist.twist.angular.y, odom_msg.twist.twist.angular.z] 
                state = position+linear_vel+quat+angular_vel
                with state_lock:
                    self._model_states[model][canon_link][:] = state
            base_topic = f"/server_partition_{self.server_partition}"
            odom_topic = os.path.join(base_topic, model, canon_link, "odom")
            self._subscribers[model][canon_link]=self._node.create_subscription(Odometry, odom_topic, cb, 10)
        # create for link map as well
        for model in self.link_map:
            links = self.link_map[model]
            state_lock = self._state_locks.get(model, Lock())
            self._state_locks[model] = state_lock
            for link in links:
                if link in self.canonical_link_names:
                    continue
                self._model_states[model] = dict()
                self._model_states[model][link] = np.zeros((13), dtype=np.float64)
                self._subscribers[model] = dict()
                msg = Odometry()
                state_lock = Lock()
                self._state_locks[model] = state_lock
                # create cbs 
                def cb(odom_msg, model=model, link=link):
                    self._wait_flag = False
                    position = [odom_msg.pose.pose.position.x,odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z] 
                    quat = [odom_msg.pose.pose.orientation.w, odom_msg.pose.pose.orientation.x,odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z] 
                    linear_vel = [odom_msg.twist.twist.linear.x,odom_msg.twist.twist.linear.y, odom_msg.twist.twist.linear.z]
                    angular_vel = [odom_msg.twist.twist.angular.x,odom_msg.twist.twist.angular.y, odom_msg.twist.twist.angular.z] 
                    state = position+linear_vel+quat+angular_vel
                    with state_lock:
                        self._model_states[model][link][:] = state
                base_topic = f"/server_partition_{self.server_partition}"
                odom_topic = os.path.join(base_topic, model, link, "odom")
                self._subscribers[model][link]=self._node.create_subscription(Odometry, odom_topic, cb, 10)

    def _make_publishers(self):
        # Create publishers, if pubs are already there
        # this will throw error
        if len(self._publishers.keys()) :
            raise RuntimeError("publishers are already made")
        base_topic = f"/server_partition_{self.server_partition}"
        ackermann_cmd_topic_template= os.path.join( base_topic,"{}","{}",
                                                    "ackermann_velocity_cmd")
        angular_velocity_cmd_topic_template = os.path.join( base_topic,"{}","{}",
                                                    "angular_velocity_cmd")
        joint_position_cmd_topic_template = os.path.join( base_topic,"{}","{}",
                                                    "joint_position_cmd")
        odom_topic_template = os.path.join( base_topic,"{}","{}",
                                                    "odom")
        rotor_velocity_cmd_topic_template = os.path.join( base_topic,"{}","{}",
                                                    "rotor_velocity_cmd")
        set_inertia_topic_template = os.path.join( base_topic,"{}","{}",
                                                    "set_inertia")
        wrench_topic_template = os.path.join( base_topic,"{}","{}",
                                                    "set_wrench")
        velocity_cmd_topic_template = os.path.join( base_topic,"{}","{}",
                                                    "velocity_cmd")
        
        reset_topic_template =  os.path.join( base_topic,"{}",
                                                    "reset_pose")
        for model_name, link_name in zip(self.model_names, self.canonical_link_names):
            ackermann_cmd_topic = ackermann_cmd_topic_template.format(model_name, link_name)
            angular_velocity_cmd_topic = angular_velocity_cmd_topic_template.format(model_name, link_name)
            joint_position_cmd_topic = joint_position_cmd_topic_template.format(model_name, link_name)
            odom_topic = odom_topic_template.format(model_name, link_name)
            rotor_velocity_cmd_topic = rotor_velocity_cmd_topic_template.format(model_name, link_name)
            set_inertia_topic = set_inertia_topic_template.format(model_name, link_name)
            wrench_topic = wrench_topic_template.format(model_name, link_name)
            velocity_cmd_topic = velocity_cmd_topic_template.format(model_name, link_name)
            reset_topic = reset_topic_template.format(model_name)
 
            self._publishers[model_name] = dict()
            self._publishers[model_name][link_name] = dict()
            self._publishers[model_name][link_name]["ackermann_cmd"] = self._node.create_publisher(Twist, ackermann_cmd_topic, 10)
            self._publishers[model_name][link_name]["angular_velocity_cmd"] = self._node.create_publisher(Vector3, angular_velocity_cmd_topic, 10)
            self._publishers[model_name][link_name]["joint_position_cmd"] = self._node.create_publisher(Vector3, joint_position_cmd_topic, 10)
            self._publishers[model_name][link_name]["rotor_velocity_cmd"] = self._node.create_publisher(Float32MultiArray, rotor_velocity_cmd_topic, 10)
            self._publishers[model_name][link_name]["set_inertia"] = self._node.create_publisher(Inertia, set_inertia_topic, 10)
            self._publishers[model_name][link_name]["wrench"] = self._node.create_publisher(Wrench, wrench_topic, 10)
            self._publishers[model_name][link_name]["velocity_cmd"] = self._node.create_publisher(Vector3, velocity_cmd_topic, 10)
            self._reset_publishers[model_name] = self._node.create_publisher(Pose, reset_topic, 10)

        # for link map we need to also create publishers 
        for model in self.link_map:
            links = self.link_map[model]
            for link in links:
                if link in self.canonical_link_names:
                    continue
                ackermann_cmd_topic = ackermann_cmd_topic_template.format(model, link)
                angular_velocity_cmd_topic = angular_velocity_cmd_topic_template.format(model, link)
                joint_position_cmd_topic = joint_position_cmd_topic_template.format(model, link)
                odom_topic = odom_topic_template.format(model, link)
                rotor_velocity_cmd_topic = rotor_velocity_cmd_topic_template.format(model, link)
                set_inertia_topic = set_inertia_topic_template.format(model, link)
                wrench_topic = wrench_topic_template.format(model, link)
                velocity_cmd_topic = velocity_cmd_topic_template.format(model, link)
                self._publishers[model][link] = dict()
                self._publishers[model][link]["ackermann_cmd"] = self._node.create_publisher(Twist, ackermann_cmd_topic, 10)
                self._publishers[model][link]["angular_velocity_cmd"] = self._node.create_publisher(Vector3, angular_velocity_cmd_topic, 10)
                self._publishers[model][link]["joint_position_cmd"] = self._node.create_publisher(Vector3, joint_position_cmd_topic, 10)
                self._publishers[model][link]["rotor_velocity_cmd"] = self._node.create_publisher(Float32MultiArray, rotor_velocity_cmd_topic, 10)
                self._publishers[model][link]["set_inertia"] = self._node.create_publisher(Inertia, set_inertia_topic, 10)
                self._publishers[model][link]["wrench"] = self._node.create_publisher(Wrench, wrench_topic, 10)
                self._publishers[model][link]["velocity_cmd"] = self._node.create_publisher(Vector3, velocity_cmd_topic, 10)
                
    def __del__(self):
        if hasattr(self, "serv") and self.serv.is_alive():
            self.serv.terminate()
        rclpy.shutdown()
