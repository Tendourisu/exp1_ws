#!/usr/bin/env python3
"""
Example of moving to a pose goal.
- ros2 run pymoveit2 ex_pose_goal.py --ros-args -p position:="[0.3, 0.35, 0.25]" -p quat_xyzw:="[1.0, 0.0, 0.0, 0.0]" -p use_demo_obstacle:=True
- ros2 run pymoveit2 ex_pose_goal.py --ros-args -p position:="[0.25, 0.0, 1.0]" -p quat_xyzw:="[0.0, 0.0, 0.0, 1.0]" -p cartesian:=False -p synchronous:=False -p cancel_after_secs:=1.0
- ros2 run pymoveit2 ex_pose_goal.py --ros-args -p position:="[0.25, 0.0, 1.0]" -p quat_xyzw:="[0.0, 0.0, 0.0, 1.0]" -p cartesian:=False -p synchronous:=False -p cancel_after_secs:=0.0
"""

from threading import Thread

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from visualization_msgs.msg import Marker

from pymoveit2 import MoveIt2, MoveIt2State
from pymoveit2.robots import ur as robot


def main():
    rclpy.init()

    # Create node for this example
    node = Node("ex_pose_goal")

    # Declare parameters for start and goal poses
    node.declare_parameter("start_position", [0.3, 0.0, 0.22])
    node.declare_parameter("start_quat_xyzw", [1.0, 0.0, 0.0, 0.0])
    node.declare_parameter("position", [0.3, 0.35, 0.22])
    node.declare_parameter("quat_xyzw", [1.0, 0.0, 0.0, 0.0])
    node.declare_parameter("synchronous", True)
    # If non-positive, don't cancel. Only used if synchronous is False
    node.declare_parameter("cancel_after_secs", 0.0)
    # Planner ID
    node.declare_parameter("planner_id", "RRTConnectkConfigDefault")
    # Declare parameters for cartesian planning
    node.declare_parameter("cartesian", True)
    node.declare_parameter("cartesian_max_step", 0.0025)
    node.declare_parameter("cartesian_fraction_threshold", 0.2)
    node.declare_parameter("cartesian_jump_threshold", 0.0)
    node.declare_parameter("cartesian_avoid_collisions", True)
    node.declare_parameter("use_demo_obstacle", True)
    node.declare_parameter("obstacle_shape", "box")
    node.declare_parameter("obstacle_quat_xyzw", [0.0, 0.0, 0.0, 1.0])
    node.declare_parameter("obstacle_dimensions", [0.03, 0.03, 0.03])
    node.declare_parameter("planning_scene_sync_timeout", 2.0)

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
    moveit2.planner_id = (
        node.get_parameter("planner_id").get_parameter_value().string_value
    )

    marker_pub = node.create_publisher(Marker, "collision_object_marker", 10)

    # Spin the node in background thread(s) and wait a bit for initialization
    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True, args=())
    executor_thread.start()
    node.create_rate(1.0).sleep()

    # Scale down velocity and acceleration of joints (percentage of maximum)
    moveit2.max_velocity = 0.5
    moveit2.max_acceleration = 0.5

    # Get parameters
    start_position = (
        node.get_parameter("start_position").get_parameter_value().double_array_value
    )
    start_quat_xyzw = (
        node.get_parameter("start_quat_xyzw")
        .get_parameter_value()
        .double_array_value
    )
    position = node.get_parameter("position").get_parameter_value().double_array_value
    quat_xyzw = node.get_parameter("quat_xyzw").get_parameter_value().double_array_value
    synchronous = node.get_parameter("synchronous").get_parameter_value().bool_value
    cancel_after_secs = (
        node.get_parameter("cancel_after_secs").get_parameter_value().double_value
    )
    cartesian = node.get_parameter("cartesian").get_parameter_value().bool_value
    cartesian_max_step = (
        node.get_parameter("cartesian_max_step").get_parameter_value().double_value
    )
    cartesian_fraction_threshold = (
        node.get_parameter("cartesian_fraction_threshold")
        .get_parameter_value()
        .double_value
    )
    cartesian_jump_threshold = (
        node.get_parameter("cartesian_jump_threshold")
        .get_parameter_value()
        .double_value
    )
    cartesian_avoid_collisions = (
        node.get_parameter("cartesian_avoid_collisions")
        .get_parameter_value()
        .bool_value
    )
    use_demo_obstacle = (
        node.get_parameter("use_demo_obstacle").get_parameter_value().bool_value
    )
    obstacle_shape = (
        node.get_parameter("obstacle_shape").get_parameter_value().string_value
    )
    obstacle_quat_xyzw = (
        node.get_parameter("obstacle_quat_xyzw")
        .get_parameter_value()
        .double_array_value
    )
    obstacle_dimensions = (
        node.get_parameter("obstacle_dimensions")
        .get_parameter_value()
        .double_array_value
    )
    planning_scene_sync_timeout = (
        node.get_parameter("planning_scene_sync_timeout")
        .get_parameter_value()
        .double_value
    )

    # Set parameters for cartesian planning
    # Keep obstacle avoidance on in demo mode to clearly show rerouting behavior.
    moveit2.cartesian_avoid_collisions = cartesian_avoid_collisions or use_demo_obstacle
    moveit2.cartesian_jump_threshold = cartesian_jump_threshold

    obstacle_id = "task_obstacle"

    # Stage 1: move the arm to the start pose first.
    node.get_logger().info(
        f"Moving to start pose {{position: {list(start_position)}, quat_xyzw: {list(start_quat_xyzw)}}}"
    )
    moveit2.move_to_pose(
        position=start_position,
        quat_xyzw=start_quat_xyzw,
        cartesian=cartesian,
        cartesian_max_step=cartesian_max_step,
        cartesian_fraction_threshold=cartesian_fraction_threshold,
    )
    if synchronous:
        moveit2.wait_until_executed()
    else:
        print("Current State: " + str(moveit2.query_state()))
        rate = node.create_rate(10)
        while moveit2.query_state() != MoveIt2State.EXECUTING:
            rate.sleep()

        print("Current State: " + str(moveit2.query_state()))
        future = moveit2.get_execution_future()

        if cancel_after_secs > 0.0:
            sleep_time = node.create_rate(cancel_after_secs)
            sleep_time.sleep()
            print("Cancelling goal")
            moveit2.cancel_execution()

        while not future.done():
            rate.sleep()

        print("Result status: " + str(future.result().status))
        print("Result error code: " + str(future.result().result.error_code))

    if use_demo_obstacle:
        obstacle_position = [
            float((start_position[0] + position[0]) / 2.0),
            float((start_position[1] + position[1]) / 2.0),
            float((start_position[2] + position[2]) / 2.0),
        ]

        marker = Marker()
        marker.header.frame_id = robot.base_link_name()
        marker.header.stamp = node.get_clock().now().to_msg()
        marker.ns = "pose_goal_obstacle"
        marker.id = 0
        marker.action = Marker.ADD
        marker.pose.position.x = float(obstacle_position[0])
        marker.pose.position.y = float(obstacle_position[1])
        marker.pose.position.z = float(obstacle_position[2])
        marker.pose.orientation.x = float(obstacle_quat_xyzw[0])
        marker.pose.orientation.y = float(obstacle_quat_xyzw[1])
        marker.pose.orientation.z = float(obstacle_quat_xyzw[2])
        marker.pose.orientation.w = float(obstacle_quat_xyzw[3])
        marker.color.r = 0.90
        marker.color.g = 0.25
        marker.color.b = 0.25
        marker.color.a = 0.85

        node.get_logger().info(
            f"Adding demo obstacle '{{shape: {obstacle_shape}, position: {list(obstacle_position)}, quat_xyzw: {list(obstacle_quat_xyzw)}, dimensions: {list(obstacle_dimensions)}}}'"
        )
        if obstacle_shape == "box":
            # moveit2.add_collision_box(
            #     id=obstacle_id,
            #     position=obstacle_position,
            #     quat_xyzw=obstacle_quat_xyzw,
            #     size=obstacle_dimensions,
            # )
            marker.type = Marker.CUBE
            marker.scale.x = float(obstacle_dimensions[0])
            marker.scale.y = float(obstacle_dimensions[1])
            marker.scale.z = float(obstacle_dimensions[2])
        elif obstacle_shape == "sphere":
            moveit2.add_collision_sphere(
                id=obstacle_id,
                position=obstacle_position,
                radius=obstacle_dimensions[0],
            )
            marker.type = Marker.SPHERE
            marker.scale.x = float(obstacle_dimensions[0] * 2.0)
            marker.scale.y = float(obstacle_dimensions[0] * 2.0)
            marker.scale.z = float(obstacle_dimensions[0] * 2.0)
        elif obstacle_shape == "cylinder":
            moveit2.add_collision_cylinder(
                id=obstacle_id,
                position=obstacle_position,
                quat_xyzw=obstacle_quat_xyzw,
                height=obstacle_dimensions[0],
                radius=obstacle_dimensions[1],
            )
            marker.type = Marker.CYLINDER
            marker.scale.x = float(obstacle_dimensions[1] * 2.0)
            marker.scale.y = float(obstacle_dimensions[1] * 2.0)
            marker.scale.z = float(obstacle_dimensions[0])
        elif obstacle_shape == "cone":
            moveit2.add_collision_cone(
                id=obstacle_id,
                position=obstacle_position,
                quat_xyzw=obstacle_quat_xyzw,
                height=obstacle_dimensions[0],
                radius=obstacle_dimensions[1],
            )
            # RViz Marker has no cone primitive; use cylinder as visual approximation.
            marker.type = Marker.CYLINDER
            marker.scale.x = float(obstacle_dimensions[1] * 2.0)
            marker.scale.y = float(obstacle_dimensions[1] * 2.0)
            marker.scale.z = float(obstacle_dimensions[0])
        else:
            raise ValueError(f"Unknown obstacle_shape '{obstacle_shape}'")

        print("Publishing marker for demo obstacle")
        marker_pub.publish(marker)
        # Confirm obstacle is visible in planning scene before planning the next motion.
        node.get_logger().info("Waiting for obstacle to appear in planning scene...")
        wait_rate = node.create_rate(20.0)
        deadline_ns = (
            node.get_clock().now().nanoseconds
            + int(planning_scene_sync_timeout * 1_000_000_000.0)
        )
        obstacle_in_scene = False
        while node.get_clock().now().nanoseconds < deadline_ns:
            if moveit2.update_planning_scene() and moveit2.planning_scene is not None:
                scene_object_ids = {
                    collision_object.id
                    for collision_object in moveit2.planning_scene.world.collision_objects
                }
                if obstacle_id in scene_object_ids:
                    obstacle_in_scene = True
                    break
            wait_rate.sleep()

        if obstacle_in_scene:
            node.get_logger().info(
                f"Obstacle '{obstacle_id}' is in planning scene. Continuing to goal planning."
            )
        else:
            node.get_logger().warning(
                f"Obstacle '{obstacle_id}' was not observed in planning scene within "
                f"{planning_scene_sync_timeout:.2f}s. Planning may ignore obstacle."
            )

    # Stage 2: move to the final pose with avoidance enabled.
    node.get_logger().info(
        f"Moving to {{position: {list(position)}, quat_xyzw: {list(quat_xyzw)}}}"
    )
    moveit2.move_to_pose(
        position=position,
        quat_xyzw=quat_xyzw,
        cartesian=cartesian,
        cartesian_max_step=cartesian_max_step,
        cartesian_fraction_threshold=cartesian_fraction_threshold,
    )
    if synchronous:
        # Note: the same functionality can be achieved by setting
        # `synchronous:=false` and `cancel_after_secs` to a negative value.
        moveit2.wait_until_executed()
    else:
        # Wait for the request to get accepted (i.e., for execution to start)
        print("Current State: " + str(moveit2.query_state()))
        rate = node.create_rate(10)
        while moveit2.query_state() != MoveIt2State.EXECUTING:
            rate.sleep()

        # Get the future
        print("Current State: " + str(moveit2.query_state()))
        future = moveit2.get_execution_future()

        # Cancel the goal
        if cancel_after_secs > 0.0:
            # Sleep for the specified time
            sleep_time = node.create_rate(cancel_after_secs)
            sleep_time.sleep()
            # Cancel the goal
            print("Cancelling goal")
            moveit2.cancel_execution()

        # Wait until the future is done
        while not future.done():
            rate.sleep()

        # Print the result
        print("Result status: " + str(future.result().status))
        print("Result error code: " + str(future.result().result.error_code))

    if use_demo_obstacle:
        moveit2.remove_collision_object(id=obstacle_id)
        marker = Marker()
        marker.header.frame_id = robot.base_link_name()
        marker.header.stamp = node.get_clock().now().to_msg()
        marker.ns = "pose_goal_obstacle"
        marker.id = 0
        marker.action = Marker.DELETE
        marker_pub.publish(marker)

    rclpy.shutdown()
    executor_thread.join()
    exit(0)


if __name__ == "__main__":
    main()
