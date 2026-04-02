#!/usr/bin/python
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue


def generate_launch_description():
    urdf_file = os.path.join(
        get_package_share_directory("visualize"),
        "robot_description/ur3",
        "ur3.xacro",
    )
    # panda_mesh_dir = "package://my_robot_description/urdf/panda/meshes/"
    # rviz_config_file = os.path.join("src/space_imitation", "rviz", "vis.rviz")

    safe_rviz_ld_path = ":".join(
        [
            "/opt/ros/humble/opt/rviz_ogre_vendor/lib",
            "/opt/ros/humble/lib/x86_64-linux-gnu",
            "/opt/ros/humble/lib",
            "/lib/x86_64-linux-gnu",
            "/usr/lib/x86_64-linux-gnu",
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
                description="Whether to start RViz2",
            ),
            DeclareLaunchArgument(
                "robot_description",
                default_value=urdf_file,
                description="Path to URDF file",
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                namespace="visualize",
                name="robot_state_publisher",
                output="screen",
                parameters=[
                    {
                        "robot_description": ParameterValue(
                            Command(
                                [
                                    "xacro ",
                                    str(urdf_file),
                                ]
                            ),
                            value_type=str,
                        )
                    },
                    {"frame_prefix": "/visualize/"},
                ],
                remappings=[("/joint_states", "joint_states")],
            ),
            Node(
                package="joint_state_publisher",
                executable="joint_state_publisher",
                namespace="visualize",
                name="joint_state_publisher",
                output="screen",
                parameters=[{"source_list": ["robot_joint_states"]}],
                remappings=[("/joint_states", "joint_states")],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                condition=IfCondition(LaunchConfiguration("use_rviz")),
                additional_env={
                    # Avoid loading incompatible snap runtime libraries.
                    "LD_LIBRARY_PATH": safe_rviz_ld_path,
                    "LD_PRELOAD": "",
                    "GTK_PATH": "",
                },
                # arguments=["-d", rviz_config_file],
            ),
        ]
    )
