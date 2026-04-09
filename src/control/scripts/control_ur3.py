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


def moveL(point, speed=0.25, acceleration=0.5):
    """Move UR3 TCP linearly to a single target point/pose.

    Args:
        point: [x, y, z] or [x, y, z, rx, ry, rz].
        speed: Linear speed used by RTDE moveL.
        acceleration: Linear acceleration used by RTDE moveL.
    """
    point = np.asarray(point, dtype=float).reshape(-1)
    if point.size not in (3, 6):
        raise ValueError("point must be [x,y,z] or [x,y,z,rx,ry,rz]")

    actual_pose = np.asarray(rtde_r.getActualTCPPose(), dtype=float)
    target_pose = actual_pose.copy()
    if point.size == 3:
        target_pose[:3] = point
    else:
        target_pose = point

    rtde_c.moveL(target_pose.tolist(), speed, acceleration)
    return target_pose.tolist()


def interpolate_trajectory(start, end, num_points, speed=0.25, acceleration=0.5, blend=0.01):

    # Hint: interpolate a trajectory from start poase to end pose
    # Hint: interpolate translation and orientation seperately
    # Hint: use scipy.spatial.transform.Slerp() to interpolate orientation
    # Hint: the shape of returned path should be [num_points, 9] ([x, y, z, rx, ry, rz, speed, acceleration, blend])
    # Hint: set the blend as 0.0 for the last point
    # Hint: moveL for executing a trajectory: https://sdurobotics.gitlab.io/ur_rtde/api/api.html#_CPPv4N7ur_rtde20RTDEControlInterface5moveLERKNSt6vectorINSt6vectorIdEEEEb

    start = np.asarray(start, dtype=float).reshape(-1)
    end = np.asarray(end, dtype=float).reshape(-1)
    if start.size != 6 or end.size != 6:
        raise ValueError("start and end must be [x,y,z,rx,ry,rz]")
    if num_points < 1:
        raise ValueError("num_points must be >= 1")

    if num_points == 1:
        single = np.hstack([end, [speed, acceleration, 0.0]])
        return [single.tolist()]

    # Interpolate translation linearly.
    trans = np.linspace(start[:3], end[:3], num_points)

    # Interpolate orientation on SO(3) via Slerp, using rotvec form from UR.
    key_rots = Rotation.from_rotvec(np.vstack([start[3:6], end[3:6]]))
    slerp = Slerp([0.0, 1.0], key_rots)
    interp_rots = slerp(np.linspace(0.0, 1.0, num_points)).as_rotvec()

    path = []
    for i in range(num_points):
        blend_i = 0.0 if i == num_points - 1 else blend
        waypoint = np.hstack([trans[i], interp_rots[i], [speed, acceleration, blend_i]])
        path.append(waypoint.tolist())

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

    # # Task2: single-point linear motion by moveL(point)
    # target_point = [actual_pose[0] + 0.05, actual_pose[1], actual_pose[2]]
    # reached_pose = moveL(target_point)
    # print("reached tcp pose: ", reached_pose)

    # Task3: interpolate a Cartesian trajectory and execute with moveL(path)
    print("Task3: interpolate a Cartesian trajectory and execute with moveL(path)")
    start_pose = actual_pose
    end_pose = [actual_pose[0] + 0.05, actual_pose[1], actual_pose[2],
                actual_pose[3], actual_pose[4], actual_pose[5]]
    path = interpolate_trajectory(np.array(start_pose), np.array(end_pose), 50)
    rtde_c.moveL(path)

    # for Task4
    # joint_space_vel_control(path=path)
