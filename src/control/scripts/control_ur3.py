#!/usr/bin/python3

import numpy as np
import rtde_control
import rtde_receive
import time
import serial
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

from urdf_parser_py.urdf import URDF
import pinocchio
import math
from pyquaternion import Quaternion

np.set_printoptions(precision=4, suppress=True)


def interpolate_trajectory(start, end, num_points, speed=0.25, acceleration=0.5, blend=0.01):

    # Hint: interpolate a trajectory from start poase to end pose
    # Hint: interpolate translation and orientation seperately
    # Hint: use scipy.spatial.transform.Slerp() to interpolate orientation
    # Hint: the shape of returned path should be [num_points, 9] ([x, y, z, rx, ry, rz, speed, acceleration, blend])
    # Hint: set the blend as 0.0 for the last point
    # Hint: moveL for executing a trajectory: https://sdurobotics.gitlab.io/ur_rtde/api/api.html#_CPPv4N7ur_rtde20RTDEControlInterface5moveLERKNSt6vectorINSt6vectorIdEEEEb

    return path


def get_jacobian(kinematics_model, kinematics_data, actual_joint, frame_id):
    J = pinocchio.computeFrameJacobian(kinematics_model, kinematics_data, q=np.array(
        actual_joint), frame_id=frame_id, reference_frame=pinocchio.LOCAL_WORLD_ALIGNED)
    J[0, :], J[1, :], J[3, :], J[4, :] = -J[0, :], -J[1, :], -J[3, :], -J[4, :]

    return J


def joint_space_vel_control(path, acceleration=0.5, freq=10):

    dt = 1/freq
    joint_speed = np.array([0, 0, 0, 0, 0, 0])
    Kp = 5*np.eye(6)
    error = np.zeros([6, 1])
    index = 0

    kinematics_model = pinocchio.buildModelFromUrdf(
        "/home/user/exp1_ws/src/visualize/robot_description/ur3/ur3.urdf")
    kinematics_data = kinematics_model.createData()
    frame_id = kinematics_model.getFrameId("wrist_3_link")

    while True:
        t = time.time()
        actual_pose = rtde_r.getActualTCPPose()
        actual_joint = rtde_r.getActualQ()
        index += 1
        if index >= len(path)-1:
            index = len(path)-1
            if np.linalg.norm(error) < 0.001:
                rtde_c.speedStop()
                break

        target_pose = path[index]

        # get Jacobian matrix
        J = get_jacobian(kinematics_model, kinematics_data,
                         actual_joint, frame_id)
        J_inv = np.linalg.pinv(J)

        # Hint: caculate the error between actual pose and target pose
        # Hint: caculate error for  translation and orientation seperately
        # Hint: 1. use Quaternion to convert axis angle to quaternion
        # Hint: 2. caculate the error between two quaternions and convert to axis angle
        # Hint: 3. convert the axis angle error to base_link
        # Hint: 4. use Jacobian matrix to get controller input

        # velocity control in joint space
        rtde_c.speedJ(joint_speed, acceleration, dt)

        loop_time = time.time() - t
        if loop_time < 0.1:
            time.sleep(0.1 - loop_time)


if __name__ == "__main__":

    robot_ip = "192.168.56.3"

    global rtde_c, rtde_r
    rtde_c = rtde_control.RTDEControlInterface(robot_ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)

    actual_pose = rtde_r.getActualTCPPose()
    print("curr tcp pose: ", actual_pose)

    # # you need to define
    # start_pose = xxx
    # end_pose = xxx

    # path = interpolate_trajectory(np.array(start_pose), np.array(end_pose), 50)
    # for Task3
    # rtde_c.moveL(path)

    # for Task4
    # joint_space_vel_control(path=path)
