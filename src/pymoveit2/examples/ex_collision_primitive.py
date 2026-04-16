#!/usr/bin/env python3
"""
Example of adding and removing a collision object with a primitive geometry.
- ros2 run pymoveit2 ex_collision_primitive.py --ros-args -p shape:="sphere" -p position:="[0.5, 0.0, 0.5]" -p dimensions:="[0.04]"
- ros2 run pymoveit2 ex_collision_primitive.py --ros-args -p shape:="cylinder" -p position:="[0.2, 0.0, -0.045]" -p quat_xyzw:="[0.0, 0.0, 0.0, 1.0]" -p dimensions:="[0.04, 0.02]"
- ros2 run pymoveit2 ex_collision_primitive.py --ros-args -p action:="remove" -p shape:="sphere"
- ros2 run pymoveit2 ex_collision_primitive.py --ros-args -p action:="move" -p shape:="sphere" -p position:="[0.2, 0.0, 0.2]"
"""

from threading import Thread

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from visualization_msgs.msg import Marker

from pymoveit2 import MoveIt2
from pymoveit2.robots import ur as robot


def main():
    rclpy.init()

    # Create node for this example
    node = Node("ex_collision_primitive")

    # Declare parameter for joint positions
    node.declare_parameter(
        "shape",
        "box",
    )
    node.declare_parameter(
        "action",
        "add",
    )
    node.declare_parameter("position", [0.3, 0.2, 0.22])
    node.declare_parameter("quat_xyzw", [0.0, 0.0, 0.0, 1.0])
    node.declare_parameter("dimensions", [0.12, 0.08, 0.06])

    # Create callback group that allows execution of callbacks in parallel without restrictions
    callback_group = ReentrantCallbackGroup()

    # Create MoveIt 2 interface
    moveit2 = MoveIt2(
        node=node,
        joint_names=robot.joint_names(),
        base_link_name=robot.base_link_name(),
        end_effector_name=robot.end_effector_name(),
        group_name=robot.MOVE_GROUP_ARM,
        callback_group=callback_group,
    )

    marker_pub = node.create_publisher(Marker, "collision_object_marker", 10)

    # Spin the node in background thread(s) and wait a bit for initialization
    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True, args=())
    executor_thread.start()
    node.create_rate(1.0).sleep()

    # Get parameters
    shape = node.get_parameter("shape").get_parameter_value().string_value
    action = node.get_parameter("action").get_parameter_value().string_value
    position = node.get_parameter("position").get_parameter_value().double_array_value
    quat_xyzw = node.get_parameter("quat_xyzw").get_parameter_value().double_array_value
    dimensions = (
        node.get_parameter("dimensions").get_parameter_value().double_array_value
    )

    # Use the name of the primitive shape as the ID
    object_id = shape

    marker = Marker()
    marker.header.frame_id = robot.base_link_name()
    marker.header.stamp = node.get_clock().now().to_msg()
    marker.ns = "collision_primitives"
    marker.id = 0
    marker.pose.position.x = float(position[0])
    marker.pose.position.y = float(position[1])
    marker.pose.position.z = float(position[2])
    marker.pose.orientation.x = float(quat_xyzw[0])
    marker.pose.orientation.y = float(quat_xyzw[1])
    marker.pose.orientation.z = float(quat_xyzw[2])
    marker.pose.orientation.w = float(quat_xyzw[3])
    marker.color.r = 0.15
    marker.color.g = 0.70
    marker.color.b = 0.25
    marker.color.a = 0.85

    if shape == "box":
        marker.type = Marker.CUBE
        marker.scale.x = float(dimensions[0])
        marker.scale.y = float(dimensions[1])
        marker.scale.z = float(dimensions[2])
    elif shape == "sphere":
        marker.type = Marker.SPHERE
        marker.scale.x = float(dimensions[0] * 2.0)
        marker.scale.y = float(dimensions[0] * 2.0)
        marker.scale.z = float(dimensions[0] * 2.0)
    elif shape == "cylinder":
        marker.type = Marker.CYLINDER
        marker.scale.x = float(dimensions[1] * 2.0)
        marker.scale.y = float(dimensions[1] * 2.0)
        marker.scale.z = float(dimensions[0])
    elif shape == "cone":
        # RViz Marker has no cone primitive; use cylinder as visual approximation.
        marker.type = Marker.CYLINDER
        marker.scale.x = float(dimensions[1] * 2.0)
        marker.scale.y = float(dimensions[1] * 2.0)
        marker.scale.z = float(dimensions[0])
    else:
        raise ValueError(f"Unknown shape '{shape}'")

    if action == "add":
        # Add collision primitive
        node.get_logger().info(
            f"Adding collision primitive of type '{shape}' "
            f"{{position: {list(position)}, quat_xyzw: {list(quat_xyzw)}, dimensions: {list(dimensions)}}}"
        )
        if shape == "box":
            moveit2.add_collision_box(
                id=object_id, position=position, quat_xyzw=quat_xyzw, size=dimensions
            )
        elif shape == "sphere":
            moveit2.add_collision_sphere(
                id=object_id, position=position, radius=dimensions[0]
            )
        elif shape == "cylinder":
            moveit2.add_collision_cylinder(
                id=object_id,
                position=position,
                quat_xyzw=quat_xyzw,
                height=dimensions[0],
                radius=dimensions[1],
            )
        elif shape == "cone":
            moveit2.add_collision_cone(
                id=object_id,
                position=position,
                quat_xyzw=quat_xyzw,
                height=dimensions[0],
                radius=dimensions[1],
            )
        else:
            raise ValueError(f"Unknown shape '{shape}'")
        marker.action = Marker.ADD
        marker_pub.publish(marker)
    elif action == "remove":
        # Remove collision primitive
        node.get_logger().info(f"Removing collision primitive with ID '{object_id}'")
        moveit2.remove_collision_object(id=object_id)
        marker.action = Marker.DELETE
        marker_pub.publish(marker)
    elif action == "move":
        # Move collision primitive
        node.get_logger().info(
            f"Moving collision primitive with ID '{object_id}' to "
            f"{{position: {list(position)}, quat_xyzw: {list(quat_xyzw)}}}"
        )
        moveit2.move_collision(id=object_id, position=position, quat_xyzw=quat_xyzw)
        marker.action = Marker.ADD
        marker_pub.publish(marker)
    else:
        raise ValueError(
            f"Unknown action '{action}'. Valid values are 'add', 'remove', 'move'"
        )

    rclpy.shutdown()
    executor_thread.join()
    exit(0)


if __name__ == "__main__":
    main()
