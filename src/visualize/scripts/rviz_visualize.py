#!/usr/bin/env python3
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from visualization_msgs.msg import Marker, MarkerArray


class RvizVisualizer(Node):
    def __init__(self):
        super().__init__('rviz_visualizer')

        # define a publisher to publish JointState message
        # define a TF Buffer and a TransfomListener
        # define a publisher to publish Marker message

        self.timer = self.create_timer(0.1, self.on_timer)
        self.step = 0

    def publish_robot_joint_states(self, joints_pos, joints_name):
        # publish a JointState type message
        pass

    def publish_marker(self):
        # use lookup_transform to get transform from visualize/wrist_3_link to visualize/base_link
        # publish a Marker type message

        pass

    def on_timer(self):
        if self.step < 100:
            joint_pos = np.zeros(6)
            # change joint positions

            self.publish_robot_joint_states(joint_pos, self.joint_name)
            self.publish_marker()
        self.step += 1


if __name__ == "__main__":

    rclpy.init()
    node = RvizVisualizer()
    time.sleep(1)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
