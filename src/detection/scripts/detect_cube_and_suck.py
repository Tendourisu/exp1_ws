#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
import sys
from threading import Event, Lock, Thread
import time

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation
import serial
from tf2_ros import Buffer, TransformException, TransformListener

from detect_cube import CubeDetector


def normalize(vector: np.ndarray) -> np.ndarray:
	vector = np.asarray(vector, dtype=np.float64)
	norm = np.linalg.norm(vector)
	if norm < 1e-9:
		raise ValueError(f"Cannot normalize near-zero vector: {vector}")
	return vector / norm


def matrix_from_transform(transform) -> np.ndarray:
	translation = transform.transform.translation
	rotation = transform.transform.rotation

	matrix = np.eye(4, dtype=np.float64)
	matrix[:3, :3] = Rotation.from_quat(
		[rotation.x, rotation.y, rotation.z, rotation.w]
	).as_matrix()
	matrix[:3, 3] = [translation.x, translation.y, translation.z]
	return matrix


def matrix_from_pose(position: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
	matrix = np.eye(4, dtype=np.float64)
	matrix[:3, :3] = np.asarray(rotation_matrix, dtype=np.float64)
	matrix[:3, 3] = np.asarray(position, dtype=np.float64)
	return matrix


BASE_TO_RTDE_ROTATION = np.diag([-1.0, -1.0, 1.0])


def base_vector_to_rtde(vector: np.ndarray) -> np.ndarray:
	return BASE_TO_RTDE_ROTATION @ np.asarray(vector, dtype=np.float64)


def base_rotation_to_rtde(rotation_matrix: np.ndarray) -> np.ndarray:
	return BASE_TO_RTDE_ROTATION @ np.asarray(rotation_matrix, dtype=np.float64)


def rtde_rotation_to_base(rotation_matrix: np.ndarray) -> np.ndarray:
	return BASE_TO_RTDE_ROTATION.T @ np.asarray(rotation_matrix, dtype=np.float64)


def tool_rotation_from_z_axis(
	tool_z_axis_in_base: np.ndarray,
	reference_x_axis_in_base: np.ndarray,
) -> np.ndarray:
	z_axis = normalize(tool_z_axis_in_base)
	x_axis = np.asarray(reference_x_axis_in_base, dtype=np.float64)
	x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis

	if np.linalg.norm(x_axis) < 1e-6:
		fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
		if abs(np.dot(fallback, z_axis)) > 0.9:
			fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
		x_axis = fallback - np.dot(fallback, z_axis) * z_axis

	x_axis = normalize(x_axis)
	y_axis = normalize(np.cross(z_axis, x_axis))
	x_axis = normalize(np.cross(y_axis, z_axis))
	return np.column_stack((x_axis, y_axis, z_axis))


@dataclass
class CubePoseInBase:
	position: np.ndarray
	rotation_matrix: np.ndarray
	normal_towards_camera: np.ndarray
	camera_position: np.ndarray


class SuctionCup:
	SUCK_CMD = bytes.fromhex("01 06 00 02 00 01 E9 CA")
	RELEASE_CMD = bytes.fromhex("01 06 00 02 00 02 A9 CB")

	def __init__(self, port="/dev/ttyUSB0", baudrate=115200, timeout=1.0, enabled=True):
		self.enabled = enabled
		self.serial = None
		if not self.enabled:
			return
		if not hasattr(serial, "Serial"):
			raise RuntimeError(
				"Imported package 'serial' has no Serial class. "
				"Install pyserial and remove the unrelated 'serial' package."
			)
		self.serial = serial.Serial(port, baudrate, timeout=timeout, bytesize=8)

	def suck(self):
		if self.serial is not None:
			self.serial.write(self.SUCK_CMD)

	def release(self):
		if self.serial is not None:
			self.serial.write(self.RELEASE_CMD)

	def close(self):
		if self.serial is not None and self.serial.is_open:
			self.serial.close()


class CubePickAndSuck(CubeDetector):
	def __init__(self):
		super().__init__(visualization_mode="deferred")

		self.declare_parameter("base_frame", "base_link")
		self.declare_parameter("camera_frame", "camera_color_optical_frame")
		self.declare_parameter("cube_frame", "cube")

		self.declare_parameter("wait_cube_timeout", 20.0)
		self.declare_parameter("max_detection_age", 1.0)
		self.declare_parameter("approach_distance", 0.10)
		self.declare_parameter("grasp_distance", 0.08)
		self.declare_parameter("tool_z_points_to_cube", True)

		self.declare_parameter("robot_ip", "192.168.56.3")
		self.declare_parameter("move_speed", 0.03)
		self.declare_parameter("move_acceleration", 0.10)
		self.declare_parameter("lift_distance", 0.10)
		self.declare_parameter("hold_before_release", 1.0)
		self.declare_parameter("suck_settle_time", 0.2)

		self.declare_parameter("enable_suction", True)
		self.declare_parameter("suction_port", "/dev/ttyUSB0")
		self.declare_parameter("suction_baudrate", 115200)
		self.declare_parameter("suction_timeout", 1.0)

		self.base_frame = self.get_parameter("base_frame").value
		self.camera_frame = self.get_parameter("camera_frame").value
		self.cube_frame = self.get_parameter("cube_frame").value

		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

		self.rtde_c = None
		self.rtde_r = None

		self._latest_cube_lock = Lock()
		self._latest_cube_camera_pose = None
		self.get_logger().info("CubePickAndSuck node started.")

	def _broadcast_cube_tf(self, position: np.ndarray, rot: np.ndarray, stamp) -> None:
		super()._broadcast_cube_tf(position, rot, stamp)
		with self._latest_cube_lock:
			self._latest_cube_camera_pose = (
				np.array(position, dtype=np.float64),
				np.array(rot, dtype=np.float64),
				stamp,
			)

	def latest_cube_camera_pose(self):
		with self._latest_cube_lock:
			if self._latest_cube_camera_pose is None:
				return None
			position, rotation_matrix, stamp = self._latest_cube_camera_pose
			return position.copy(), rotation_matrix.copy(), stamp

	def lookup_transform_matrix(
		self,
		target_frame: str,
		source_frame: str,
		timeout_sec: float = 1.0,
	) -> np.ndarray:
		transform = self.tf_buffer.lookup_transform(
			target_frame,
			source_frame,
			Time(),
			timeout=Duration(seconds=timeout_sec),
		)
		return matrix_from_transform(transform)

	def wait_for_transform_matrix(
		self,
		target_frame: str,
		source_frame: str,
		timeout_sec: float,
	) -> np.ndarray:
		deadline = time.monotonic() + timeout_sec
		last_error = None
		while rclpy.ok() and time.monotonic() < deadline:
			try:
				return self.lookup_transform_matrix(target_frame, source_frame, 0.2)
			except TransformException as error:
				last_error = error
				time.sleep(0.05)
		raise TimeoutError(
			f"Timed out waiting for TF {target_frame} <- {source_frame}: {last_error}"
		)

	def wait_for_cube_pose_in_base(self, timeout_sec: float) -> CubePoseInBase:
		deadline = time.monotonic() + timeout_sec
		last_error = None
		last_log_time = 0.0
		max_detection_age = float(self.get_parameter("max_detection_age").value)

		while rclpy.ok() and time.monotonic() < deadline:
			latest = self.latest_cube_camera_pose()
			if latest is None:
				if time.monotonic() - last_log_time > 2.0:
					last_log_time = time.monotonic()
					self.get_logger().info(
						"Waiting for CubeDetector to publish the cube TF..."
					)
				time.sleep(0.05)
				continue

			position_camera, rotation_camera, stamp = latest
			if max_detection_age > 0.0:
				age = (self.get_clock().now() - Time.from_msg(stamp)).nanoseconds * 1e-9
				if age > max_detection_age:
					if time.monotonic() - last_log_time > 2.0:
						last_log_time = time.monotonic()
						self.get_logger().info(
							f"Latest cube detection is stale ({age:.2f}s old)."
						)
					time.sleep(0.05)
					continue

			try:
				base_to_cube = self.lookup_transform_matrix(
					self.base_frame,
					self.cube_frame,
					0.2,
				)
			except TransformException as error:
				last_error = error
				if time.monotonic() - last_log_time > 2.0:
					last_log_time = time.monotonic()
					self.get_logger().info(
						f"Waiting for TF {self.base_frame} <- {self.cube_frame}: {error}"
					)
				time.sleep(0.05)
				continue

			camera_to_cube = matrix_from_pose(position_camera, rotation_camera)
			cube_position = base_to_cube[:3, 3]
			cube_rotation = base_to_cube[:3, :3]

			normal = normalize(cube_rotation[:, 2])
			base_to_camera = base_to_cube @ np.linalg.inv(camera_to_cube)
			camera_position = base_to_camera[:3, 3]
			cube_to_camera = camera_position - cube_position
			if np.dot(normal, cube_to_camera) < 0.0:
				normal = -normal

			return CubePoseInBase(
				position=cube_position,
				rotation_matrix=cube_rotation,
				normal_towards_camera=normal,
				camera_position=camera_position,
			)

		raise TimeoutError(
			f"Timed out waiting for TF {self.base_frame} <- {self.cube_frame}. "
			"Check that /tf_static contains the hand-eye transform and that the "
			f"camera TF chain reaches {self.camera_frame}. Last TF error: {last_error}"
		)

	def connect_robot(self) -> None:
		if self.rtde_c is not None and self.rtde_r is not None:
			return
		robot_ip = self.get_parameter("robot_ip").value
		self.get_logger().info(f"Connecting RTDE to robot at {robot_ip}...")
		self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
		self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)

	def current_tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
		self.connect_robot()
		pose = np.asarray(self.rtde_r.getActualTCPPose(), dtype=np.float64)
		if pose.size != 6:
			raise RuntimeError(f"Unexpected TCP pose from RTDE: {pose}")
		rotation_matrix = Rotation.from_rotvec(pose[3:6]).as_matrix()
		return pose, rotation_matrix

	def moveL_pose(self, label: str, position: np.ndarray, rotation_matrix: np.ndarray) -> None:
		self.connect_robot()
		rtde_position = base_vector_to_rtde(position)
		rtde_rotation = base_rotation_to_rtde(rotation_matrix)
		rotvec = Rotation.from_matrix(rtde_rotation).as_rotvec()
		target_pose = np.hstack((rtde_position, rotvec))
		speed = float(self.get_parameter("move_speed").value)
		acceleration = float(self.get_parameter("move_acceleration").value)

		self.get_logger().info(
			f"{label}: moveL pose={np.round(target_pose, 4).tolist()}, "
			f"speed={speed:.4f}, acceleration={acceleration:.4f}"
		)
		self.rtde_c.moveL(target_pose.tolist(), speed, acceleration)

	def create_suction(self) -> SuctionCup:
		return SuctionCup(
			port=self.get_parameter("suction_port").value,
			baudrate=int(self.get_parameter("suction_baudrate").value),
			timeout=float(self.get_parameter("suction_timeout").value),
			enabled=bool(self.get_parameter("enable_suction").value),
		)

	def run_pick_sequence(self) -> None:
		self.get_logger().info("Waiting for current TCP pose...")
		home_pose, home_rotation = self.current_tcp_pose()
		home_position = home_pose[:3]
		home_rotation_base = rtde_rotation_to_base(home_rotation)
		self.get_logger().info(
			f"Recorded home pose: pose={np.round(home_pose, 4).tolist()}"
		)

		self.get_logger().info("Waiting for cube detection and hand-eye transform...")
		cube_pose = self.wait_for_cube_pose_in_base(
			float(self.get_parameter("wait_cube_timeout").value)
		)
		self.get_logger().info(
			f"Cube in {self.base_frame}: position={np.round(cube_pose.position, 4).tolist()}, "
			f"normal={np.round(cube_pose.normal_towards_camera, 4).tolist()}"
		)

		approach_distance = float(self.get_parameter("approach_distance").value)
		grasp_distance = float(self.get_parameter("grasp_distance").value)
		normal = cube_pose.normal_towards_camera
		pre_grasp_position = cube_pose.position + normal * approach_distance
		grasp_position = cube_pose.position + normal * grasp_distance

		if bool(self.get_parameter("tool_z_points_to_cube").value):
			tool_z_axis = -normal
		else:
			tool_z_axis = normal
		target_rotation = tool_rotation_from_z_axis(tool_z_axis, home_rotation_base[:, 0])

		suction = self.create_suction()
		suction_on = False
		try:
			self.moveL_pose(
				"Stage 1 pre-grasp",
				pre_grasp_position,
				target_rotation,
			)
			self.moveL_pose(
				"Stage 2 grasp",
				grasp_position,
				target_rotation,
			)

			self.get_logger().info("Suction on.")
			suction.suck()
			suction_on = True
			time.sleep(float(self.get_parameter("suck_settle_time").value))

			lift_position = grasp_position.copy()
			lift_position[2] += float(self.get_parameter("lift_distance").value)
			self.moveL_pose(
				"Stage 3 lift",
				lift_position,
				target_rotation,
			)

			time.sleep(float(self.get_parameter("hold_before_release").value))
			self.get_logger().info("Suction release.")
			suction.release()
			suction_on = False
		finally:
			if suction_on:
				self.get_logger().warn("Releasing suction during cleanup.")
				suction.release()
			suction.close()


def main(args=None):
	rclpy.init(args=args)
	node = CubePickAndSuck()
	executor = MultiThreadedExecutor(4)
	executor.add_node(node)
	executor_thread = Thread(target=executor.spin, daemon=True)
	executor_thread.start()
	gui_stop = Event()
	pick_done = Event()
	pick_error = []

	def pick_worker():
		try:
			time.sleep(1.0)
			node.run_pick_sequence()
		except Exception as error:
			pick_error.append(error)
			node.get_logger().error(str(error))
		finally:
			pick_done.set()

	pick_thread = Thread(target=pick_worker, daemon=True)
	pick_thread.start()

	try:
		while rclpy.ok() and not pick_done.is_set():
			node.render_gui()
			time.sleep(0.01)
	except KeyboardInterrupt:
		pass
	finally:
		gui_stop.set()
		node.close_gui()
		pick_thread.join(timeout=2.0)
		executor.shutdown()
		node.destroy_node()
		rclpy.shutdown()
		executor_thread.join(timeout=1.0)

	if pick_error:
		sys.exit(1)


if __name__ == "__main__":
	main()
