# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

ROS_VERSION = None
try: 
    import rospy
    ROS_VERSION = 1
except ImportError as e:
    try:
        import rclpy 
        ROS_VERSION = 2
    except ImportError as e2:
        print("Error importing ROS modules:", e, e2)

if ROS_VERSION is None:
    raise ImportError("Could not import ROS1 or ROS2 modules. Please ensure that ROS is installed and sourced correctly.")

class RosStream:
    """A class that will manage ROS communication for streaming data.
    This is agnostic to ROS version (1 or 2).
    """
    def __init__(self, node_name: str):
        self.node_name = node_name
        if ROS_VERSION == 1:
            rospy.init_node(self.node_name)
        elif ROS_VERSION == 2:
            if (not rclpy.ok()):
                rclpy.init()
            self.node = rclpy.create_node(self.node_name)
        else:
            raise RuntimeError("ROS version not supported.")

        self.async_spin_thread = None
        self.publishers = []
        self.subscribers = []
        self.timers = []
    
    def __del__(self):
        
        if ROS_VERSION == 2:
            for timer in self.timers:
                timer.cancel()
            if self.node is not None:
                self.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            
        elif ROS_VERSION == 1:
            rospy.signal_shutdown("Node deleted.")

        if (self.async_spin_thread is not None and self.async_spin_thread.is_alive()):
            self.async_spin_thread.join()

    def subscribe(self, topic_name: str,
                  msg_type, callback, queue_size: int):
        """Subscribe to a ROS topic."""
        if ROS_VERSION == 1:
            self.subscribers.append(rospy.Subscriber(topic_name, msg_type, callback,
                                               queue_size=queue_size))
        elif ROS_VERSION == 2:
            self.subscribers.append(self.node.create_subscription(msg_type, topic_name, 
                                                            callback, queue_size))
        else:
            raise RuntimeError("ROS version not supported.")
        return self.subscribers[-1]
    
    def publish(self, topic_name: str,
                  msg_type, queue_size: int):
        """Create a ROS publisher."""
        if ROS_VERSION == 1:
            self.publishers.append(rospy.Publisher(topic_name, msg_type,
                                              queue_size=queue_size))
        elif ROS_VERSION == 2:
            self.publishers.append(self.node.create_publisher(msg_type, topic_name,
                                                        queue_size))
        else:
            raise RuntimeError("ROS version not supported.")
        return self.publishers[-1]
    
    def create_timer(self, period_sec: float, callback):
        """Create a ROS timer."""
        if ROS_VERSION == 1:
            self.timers.append(rospy.Timer(rospy.Duration(period_sec), callback))
        elif ROS_VERSION == 2:
            self.timers.append(self.node.create_timer(period_sec, callback))
        else:
            raise RuntimeError("ROS version not supported.")
        return self.timers[-1]
    
    def spin_async(self):
        """Spin the ROS node asynchronously."""
        if ROS_VERSION == 1:
            import threading
            self.async_spin_thread = threading.Thread(target=rospy.spin)
            self.async_spin_thread.start()
        elif ROS_VERSION == 2:
            from rclpy.executors import MultiThreadedExecutor
            self.executor = MultiThreadedExecutor()
            self.executor.add_node(self.node)
            import threading
            self.async_spin_thread = threading.Thread(target=self.executor.spin)
            self.async_spin_thread.start()
        else:
            raise RuntimeError("ROS version not supported.")
        
    def spin(self):
        """Spin the ROS node to process callbacks."""
        if ROS_VERSION == 1:
            rospy.spin()
        elif ROS_VERSION == 2:
            rclpy.spin(self.node)
        else:
            raise RuntimeError("ROS version not supported.")