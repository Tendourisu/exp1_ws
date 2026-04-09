#!/usr/bin/env python3
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from visualization_msgs.msg import Marker


class RvizVisualizer(Node):
    def __init__(self):
        super().__init__('rviz_visualizer')

        # define a publisher to publish JointState message
        # define a TF Buffer and a TransfomListener
        # define a publisher to publish Marker message
        self.joint_state_pub = self.create_publisher(JointState, '/visualize/joint_states', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.marker_pub = self.create_publisher(Marker, '/visualize/marker', 10)
        self.joint_name = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint',
        ]

        self.timer = self.create_timer(0.1, self.on_timer)
        self.step = 0

    def publish_robot_joint_states(self, joints_pos, joints_name):
        # publish a JointState type message
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(joints_name)
        msg.position = [float(p) for p in joints_pos]
        print(f"Publishing joint states: {msg.name} with positions {msg.position}")
        self.joint_state_pub.publish(msg)

    def publish_marker(self):
        # use lookup_transform to get transform from visualize/wrist_3_link to visualize/base_link
        # publish a Marker type message
        try:
            transform = self.tf_buffer.lookup_transform(
                'visualize/base_link',
                'visualize/wrist_3_link',
                rclpy.time.Time(),
            )
        except TransformException as ex:
            self.get_logger().warn(f'Could not transform wrist_3_link to base_link: {ex}')
            return

        marker = Marker()
        marker.header.frame_id = 'visualize/base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'wrist_marker'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = transform.transform.translation.x
        marker.pose.position.y = transform.transform.translation.y
        marker.pose.position.z = transform.transform.translation.z
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.marker_pub.publish(marker)

    def on_timer(self):
        
        if self.step < 100000:
            joint_pos = np.zeros(6)
            # change joint positions
            angle = self.step * 0.01
            joint_pos[0] = angle

            self.publish_robot_joint_states(joint_pos, self.joint_name)
            self.publish_marker()
        else:
            return
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
