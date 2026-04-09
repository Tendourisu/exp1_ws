#!/usr/bin/python3

import numpy as np
import rtde_control
import rtde_receive
import time
import serial
import shutil
import subprocess
import tempfile
from pathlib import Path
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

from urdf_parser_py.urdf import URDF
import pinocchio
import math
from pyquaternion import Quaternion

np.set_printoptions(precision=4, suppress=True)


def get_ur3_urdf_path():
    """Resolve UR3 URDF path, generating from xacro if needed."""
    script_path = Path(__file__).resolve()
    search_roots = [Path.cwd(), script_path.parent]
    search_roots.extend(script_path.parents)

    ws_root = None
    for root in search_roots:
        if (root / "src/visualize/robot_description/ur3").exists():
            ws_root = root
            break
    if ws_root is None:
        raise FileNotFoundError(
            "Cannot find workspace root containing src/visualize/robot_description/ur3"
        )

    urdf_path = ws_root / "src/visualize/robot_description/ur3/ur3.urdf"
    if urdf_path.exists():
        return str(urdf_path)

    xacro_path = ws_root / "src/visualize/robot_description/ur3/ur3.xacro"
    if not xacro_path.exists():
        raise FileNotFoundError(
            f"Neither URDF nor xacro found. Checked: {urdf_path} and {xacro_path}"
        )

    xacro_exe = shutil.which("xacro")
    if xacro_exe is None:
        raise FileNotFoundError(
            "xacro command not found. Please install ros xacro package first."
        )

    tmp_urdf = Path(tempfile.gettempdir()) / "ur3_generated_from_xacro.urdf"
    result = subprocess.run(
        [xacro_exe, str(xacro_path), "-o", str(tmp_urdf)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"xacro conversion failed: {result.stderr.strip()}")

    return str(tmp_urdf)


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

    if len(path) == 0:
        raise ValueError("path must contain at least one waypoint")

    def rotvec_to_quat(rotvec):
        rotvec = np.asarray(rotvec, dtype=float)
        angle = np.linalg.norm(rotvec)
        if angle < 1e-12:
            return Quaternion()
        return Quaternion(axis=rotvec / angle, angle=angle)

    dt = 1 / freq
    joint_speed = np.zeros(6)
    Kp = 5 * np.eye(6)
    error = np.zeros((6, 1))
    index = 0

    urdf_path = get_ur3_urdf_path()
    kinematics_model = pinocchio.buildModelFromUrdf(urdf_path)
    kinematics_data = kinematics_model.createData()
    frame_id = kinematics_model.getFrameId("wrist_3_link")

    while True:
        t = time.time()
        actual_pose = np.asarray(rtde_r.getActualTCPPose(), dtype=float)
        actual_joint = np.asarray(rtde_r.getActualQ(), dtype=float)
        target_pose = np.asarray(path[index][:6], dtype=float)

        # get Jacobian matrix
        J = get_jacobian(kinematics_model, kinematics_data,
                         actual_joint, frame_id)
        J_inv = np.linalg.pinv(J)

        # Translation error in base frame.
        pos_error = target_pose[:3] - actual_pose[:3]

        # Convert rotvec to quaternion and enforce shortest-path representation.
        q_current = rotvec_to_quat(actual_pose[3:6])
        q_target = rotvec_to_quat(target_pose[3:6])
        if np.dot(q_current.elements, q_target.elements) < 0.0:
            q_target = -q_target

        # Orientation error in target frame: q_err = q_target^{-1} * q_current.
        q_err_target = q_target.inverse * q_current
        if q_err_target.angle < 1e-12:
            rot_error_target = np.zeros(3)
        else:
            rot_error_target = np.asarray(q_err_target.axis) * q_err_target.angle

        # Convert orientation error to base frame, then negate for feedback control.
        rot_target_to_base = Rotation.from_rotvec(target_pose[3:6]).as_matrix()
        rot_error = -rot_target_to_base @ rot_error_target

        error = np.hstack([pos_error, rot_error]).reshape(6, 1)

        # P control in Cartesian space and Jacobian mapping to joint speed.
        cartesian_cmd = Kp @ error
        joint_speed = (J_inv @ cartesian_cmd).reshape(-1)
        joint_speed = np.clip(joint_speed, -1.0, 1.0)

        # velocity control in joint space
        rtde_c.speedJ(joint_speed.tolist(), acceleration, dt)

        # Move to next waypoint when current one is reached.
        if np.linalg.norm(error) < 0.005 and index < len(path) - 1:
            index += 1

        # Stop at final waypoint.
        if index == len(path) - 1 and np.linalg.norm(error) < 0.001:
            rtde_c.speedStop()
            break

        loop_time = time.time() - t
        if loop_time < dt:
            time.sleep(dt - loop_time)


if __name__ == "__main__":

    robot_ip = "192.168.56.3"

    global rtde_c, rtde_r
    rtde_c = rtde_control.RTDEControlInterface(robot_ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)

    actual_pose = rtde_r.getActualTCPPose()
    print("curr tcp pose: ", actual_pose)

    # Move to a fixed initial pose before each Task4 run.
    initial_pose = [-0.32320880930417173, -0.15693014664092225, 0.2957849476113575,
                    -3.067469847862608, -0.10388452518924547, 0.11325065141792895]
    reached_pose = moveL(initial_pose)
    print("moved to initial pose: ", reached_pose)

    # # Task2: single-point linear motion by moveL(point)
    # target_point = [actual_pose[0] + 0.05, actual_pose[1], actual_pose[2]]
    # reached_pose = moveL(target_point)
    # print("reached tcp pose: ", reached_pose)

    # Task4: interpolate a Cartesian trajectory and execute with joint-space velocity control
    print("Task4: execute interpolated trajectory with joint_space_vel_control(path)")
    start_pose = np.array(rtde_r.getActualTCPPose(), dtype=float)
    end_pose = [start_pose[0] - 0.10, start_pose[1], start_pose[2],
                start_pose[3] + 0.4, start_pose[4], start_pose[5]]
    path = interpolate_trajectory(start_pose, np.array(end_pose), 50)
    joint_space_vel_control(path=path)
