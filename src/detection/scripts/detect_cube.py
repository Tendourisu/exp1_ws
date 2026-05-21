#!/usr/bin/env python3

import cv2
from cv_bridge import CvBridge, CvBridgeError
from dataclasses import dataclass
import message_filters
import numpy as np
import open3d as o3d
import rclpy
import threading
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


@dataclass
class DetectedCube:
	color: str
	position: np.ndarray
	rotation_matrix: np.ndarray
	normal_towards_camera: np.ndarray
	center_px: tuple[int, int]
	contour_area: float
	icp_fitness: float
	frame_id: str
	stamp: object


class CubeDetector(Node):
	def __init__(self, visualization_mode: str = "inline"):
		super().__init__("cube_detector")

		self.bridge = CvBridge()
		self.camera_info = None

		self.rgb_topic = "/camera/camera/color/image_raw"
		self.depth_topic = "/camera/camera/aligned_depth_to_color/image_raw"
		self.camera_info_topic = "/camera/camera/color/camera_info"

		self.camera_info_sub = self.create_subscription(
			CameraInfo,
			self.camera_info_topic,
			self.camera_info_callback,
			10,
		)

		self.rgb_sub = message_filters.Subscriber(self, Image, self.rgb_topic)
		self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic)

		self.sync = message_filters.ApproximateTimeSynchronizer(
			[self.rgb_sub, self.depth_sub],
			queue_size=10,
			slop=0.1,
			allow_headerless=False,
		)
		self.sync.registerCallback(self.image_callback)

		self.color_names = ("pink", "yellow", "purple")
		self.color_hsv_ranges = {
			"pink": [
				(np.array([0, 75, 110], dtype=np.uint8), np.array([3, 255, 255], dtype=np.uint8)),
				(np.array([168, 75, 110], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8)),
			],
			"yellow": [
				(np.array([20, 85, 120], dtype=np.uint8), np.array([36, 255, 255], dtype=np.uint8)),
			],
			"purple": [
				(np.array([105, 45, 50], dtype=np.uint8), np.array([155, 255, 255], dtype=np.uint8)),
			],
		}
		self.color_draw_bgr = {
			"pink": (203, 192, 255),
			"yellow": (0, 255, 255),
			"purple": (255, 0, 255),
		}
		self.pink_hsv_ranges = self.color_hsv_ranges["pink"]
		self._cube_list_lock = threading.Lock()
		self.pink_cubes = []
		self.yellow_cubes = []
		self.purple_cubes = []
		self.cubes_by_color = {
			"pink": self.pink_cubes,
			"yellow": self.yellow_cubes,
			"purple": self.purple_cubes,
		}
		self.min_contour_area = 300.0
		self.depth_min_m = 0.05
		self.depth_max_m = 5.0
		self.point_sample_step = 4
		self.icp_min_points = 80
		self.normal_line_length = 0.08
		self.last_debug_time = self.get_clock().now()
		self.visualization_mode = visualization_mode
		self._viz_lock = threading.Lock()
		self._viz_display = None
		self._viz_mask = None
		self._viz_real_points = None
		self._viz_virtual_before = None
		self._viz_virtual_after = None
		self._viz_plane_center = None
		self._viz_plane_normal = None
		self._viz_ready = False

		self.o3d_vis = None
		self.o3d_window_ready = False
		self.o3d_pcd = o3d.geometry.PointCloud()
		self.o3d_virtual_pcd = o3d.geometry.PointCloud()
		self.o3d_virtual_aligned_pcd = o3d.geometry.PointCloud()
		self.o3d_normal_line = o3d.geometry.LineSet()
		self.view_initialized = False
		if self.visualization_mode == "inline":
			self._ensure_open3d_window()

		self.tf_broadcaster = TransformBroadcaster(self)

		self.get_logger().info("CubeDetector node started.")

	def _ensure_open3d_window(self) -> None:
		if self.o3d_window_ready:
			return
		self.o3d_vis = o3d.visualization.Visualizer()
		self.o3d_window_ready = self.o3d_vis.create_window(
			window_name="Cube Point Cloud",
			width=960,
			height=540,
		)
		if self.o3d_window_ready:
			self.o3d_vis.add_geometry(self.o3d_pcd)
			self.o3d_vis.add_geometry(self.o3d_virtual_pcd)
			self.o3d_vis.add_geometry(self.o3d_virtual_aligned_pcd)
			self.o3d_vis.add_geometry(self.o3d_normal_line)
			render_option = self.o3d_vis.get_render_option()
			render_option.point_size = 6.0
			render_option.background_color = np.array([0.08, 0.08, 0.08])
		else:
			self.get_logger().warn("Open3D window creation failed, skip 3D visualization.")

	@staticmethod
	def depth_to_meters(depth_values: np.ndarray) -> np.ndarray:
		if depth_values.dtype == np.uint16:
			return depth_values.astype(np.float32) * 0.001

		depth_m = depth_values.astype(np.float32)
		finite_positive = depth_m[np.isfinite(depth_m) & (depth_m > 0.0)]
		if finite_positive.size > 0:
			# Some drivers publish float depth in millimeters; auto-scale to meters.
			if float(np.median(finite_positive)) > 10.0:
				depth_m *= 0.001
		return depth_m

	def contour_pixels_to_point_cloud(self, contour: np.ndarray, depth_image: np.ndarray) -> np.ndarray:
		if self.camera_info is None:
			return np.empty((0, 3), dtype=np.float32)

		fx = self.camera_info.k[0]
		fy = self.camera_info.k[4]
		cx = self.camera_info.k[2]
		cy = self.camera_info.k[5]
		if fx == 0.0 or fy == 0.0:
			return np.empty((0, 3), dtype=np.float32)

		roi_mask = np.zeros(depth_image.shape[:2], dtype=np.uint8)
		cv2.drawContours(roi_mask, [contour], -1, 255, thickness=-1)

		v_coords, u_coords = np.where(roi_mask > 0)
		if v_coords.size == 0:
			return np.empty((0, 3), dtype=np.float32)

		if self.point_sample_step > 1:
			u_coords = u_coords[:: self.point_sample_step]
			v_coords = v_coords[:: self.point_sample_step]

		z = self.depth_to_meters(depth_image[v_coords, u_coords])
		valid = np.isfinite(z)
		valid &= z > self.depth_min_m
		valid &= z < self.depth_max_m
		if not np.any(valid):
			return np.empty((0, 3), dtype=np.float32)

		u = u_coords[valid].astype(np.float32)
		v = v_coords[valid].astype(np.float32)
		z = z[valid]

		x = (u - cx) * z / fx
		y = (v - cy) * z / fy

		return np.stack((x, y, z), axis=1)

	@staticmethod
	def build_virtual_plane_points(target_points: np.ndarray) -> np.ndarray:
		target_centered = target_points - np.mean(target_points, axis=0, keepdims=True)
		cov = np.cov(target_centered, rowvar=False)
		eig_vals, eig_vecs = np.linalg.eigh(cov)
		order = np.argsort(eig_vals)[::-1]
		axis_u = eig_vecs[:, order[0]]
		axis_v = eig_vecs[:, order[1]]

		u = target_centered @ axis_u
		v = target_centered @ axis_v
		u_min, u_max = float(np.min(u)), float(np.max(u))
		v_min, v_max = float(np.min(v)), float(np.max(v))

		width = max(u_max - u_min, 1e-3)
		height = max(v_max - v_min, 1e-3)
		aspect = width / height

		target_count = target_points.shape[0]
		n_u = max(10, int(np.sqrt(target_count * aspect)))
		n_v = max(10, int(np.sqrt(target_count / max(aspect, 1e-6))))

		u_lin = np.linspace(u_min, u_max, n_u, dtype=np.float32)
		v_lin = np.linspace(v_min, v_max, n_v, dtype=np.float32)
		u_grid, v_grid = np.meshgrid(u_lin, v_lin)
		plane_points = np.stack(
			(
				u_grid.reshape(-1),
				v_grid.reshape(-1),
				np.zeros(u_grid.size, dtype=np.float32),
			),
			axis=1,
		)

		if plane_points.shape[0] > target_count:
			indices = np.linspace(0, plane_points.shape[0] - 1, target_count, dtype=np.int32)
			plane_points = plane_points[indices]

		return plane_points

	def run_icp_and_get_normal(
		self, target_points: np.ndarray
	) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
		virtual_points = self.build_virtual_plane_points(target_points)

		target_center = np.mean(target_points, axis=0)
		virtual_center = np.mean(virtual_points, axis=0)

		source_pcd = o3d.geometry.PointCloud()
		source_pcd.points = o3d.utility.Vector3dVector(virtual_points.astype(np.float64))
		target_pcd = o3d.geometry.PointCloud()
		target_pcd.points = o3d.utility.Vector3dVector(target_points.astype(np.float64))

		bbox = np.ptp(target_points, axis=0)
		diag = float(np.linalg.norm(bbox))
		max_corr_dist = max(0.01, 0.25 * diag)

		init_transform = np.eye(4, dtype=np.float64)
		init_transform[:3, 3] = (target_center - virtual_center).astype(np.float64)
		virtual_points_init = virtual_points + init_transform[:3, 3].astype(np.float32)

		reg_result = o3d.pipelines.registration.registration_icp(
			source_pcd,
			target_pcd,
			max_corr_dist,
			init_transform,
			o3d.pipelines.registration.TransformationEstimationPointToPoint(),
		)

		source_pcd.transform(reg_result.transformation)
		aligned_virtual_points = np.asarray(source_pcd.points, dtype=np.float32)

		normal = reg_result.transformation[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
		norm = np.linalg.norm(normal)
		if norm > 1e-9:
			normal = normal / norm
		normal = normal.astype(np.float32)

		# Keep normal pointing from plane center towards camera origin for consistency.
		if np.dot(normal, target_center.astype(np.float32)) > 0.0:
			normal = -normal

		rot = reg_result.transformation[:3, :3].astype(np.float32)

		return (
			virtual_points_init,
			aligned_virtual_points,
			target_center.astype(np.float32),
			normal,
			rot,
			float(reg_result.fitness),
		)

	def update_open3d(
		self,
		real_points: np.ndarray,
		virtual_points_before: np.ndarray,
		virtual_points_after: np.ndarray,
		plane_center: np.ndarray,
		plane_normal: np.ndarray,
	) -> None:
		with self._viz_lock:
			self._viz_real_points = real_points.astype(np.float64)
			self._viz_virtual_before = virtual_points_before.astype(np.float64)
			self._viz_virtual_after = virtual_points_after.astype(np.float64)
			self._viz_plane_center = plane_center.astype(np.float64)
			self._viz_plane_normal = plane_normal.astype(np.float64)
			self._viz_ready = True

	def _store_2d_visualization(self, display: np.ndarray, mask: np.ndarray) -> None:
		with self._viz_lock:
			self._viz_display = display.copy()
			self._viz_mask = mask.copy()

	def render_gui(self) -> None:
		if self.visualization_mode == "disabled":
			return
		self._ensure_open3d_window()
		with self._viz_lock:
			display = None if self._viz_display is None else self._viz_display.copy()
			mask = None if self._viz_mask is None else self._viz_mask.copy()
			real_points = self._viz_real_points
			virtual_before = self._viz_virtual_before
			virtual_after = self._viz_virtual_after
			plane_center = self._viz_plane_center
			plane_normal = self._viz_plane_normal
			viz_ready = self._viz_ready

		if display is not None:
			cv2.imshow("CubeDetector RGB", display)
		if mask is not None:
			cv2.imshow("CubeDetector Mask", mask)
		cv2.waitKey(1)

		if not self.o3d_window_ready or not viz_ready:
			return
		if real_points is None or real_points.size == 0:
			self.o3d_vis.poll_events()
			self.o3d_vis.update_renderer()
			return

		self.o3d_pcd.points = o3d.utility.Vector3dVector(real_points)
		real_colors = np.tile(np.array([[1.0, 0.2, 0.2]], dtype=np.float64), (real_points.shape[0], 1))
		self.o3d_pcd.colors = o3d.utility.Vector3dVector(real_colors)

		if virtual_before is not None:
			self.o3d_virtual_pcd.points = o3d.utility.Vector3dVector(virtual_before)
			virtual_colors = np.tile(np.array([[0.2, 1.0, 0.2]], dtype=np.float64), (virtual_before.shape[0], 1))
			self.o3d_virtual_pcd.colors = o3d.utility.Vector3dVector(virtual_colors)

		if virtual_after is not None:
			self.o3d_virtual_aligned_pcd.points = o3d.utility.Vector3dVector(virtual_after)
			aligned_colors = np.tile(np.array([[0.2, 0.6, 1.0]], dtype=np.float64), (virtual_after.shape[0], 1))
			self.o3d_virtual_aligned_pcd.colors = o3d.utility.Vector3dVector(aligned_colors)

		if plane_center is not None and plane_normal is not None:
			line_start = plane_center.astype(np.float64)
			line_end = (plane_center + self.normal_line_length * plane_normal).astype(np.float64)
			line_points = np.vstack((line_start, line_end))
			self.o3d_normal_line.points = o3d.utility.Vector3dVector(line_points)
			self.o3d_normal_line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
			self.o3d_normal_line.colors = o3d.utility.Vector3dVector(np.array([[0.2, 0.7, 1.0]], dtype=np.float64))

		self.o3d_vis.update_geometry(self.o3d_pcd)
		self.o3d_vis.update_geometry(self.o3d_virtual_pcd)
		self.o3d_vis.update_geometry(self.o3d_virtual_aligned_pcd)
		self.o3d_vis.update_geometry(self.o3d_normal_line)

		if not self.view_initialized:
			self.o3d_vis.reset_view_point(True)
			self.view_initialized = True

		self.o3d_vis.poll_events()
		self.o3d_vis.update_renderer()

	def camera_info_callback(self, msg: CameraInfo) -> None:
		self.camera_info = msg

	def _cube_frame_id(self, color: str, index: int | None = None) -> str:
		if index is None:
			return color
		return f"{color}_cube_{index}"

	def _build_color_mask(self, hsv_image: np.ndarray, hsv_ranges: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
		mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
		for lower, upper in hsv_ranges:
			mask = cv2.bitwise_or(mask, cv2.inRange(hsv_image, lower, upper))

		kernel = np.ones((5, 5), dtype=np.uint8)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
		return mask

	@staticmethod
	def _contour_center(contour: np.ndarray) -> tuple[int, int] | None:
		moments = cv2.moments(contour)
		if moments["m00"] == 0:
			return None
		return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])

	def _find_cube_candidates(self, mask: np.ndarray) -> list[tuple[float, np.ndarray, np.ndarray, tuple[int, int]]]:
		contours_result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours = contours_result[0] if len(contours_result) == 2 else contours_result[1]

		candidates = []
		for contour in contours:
			area = cv2.contourArea(contour)
			if area < self.min_contour_area:
				continue

			perimeter = cv2.arcLength(contour, True)
			if perimeter <= 0.0:
				continue
			approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

			if len(approx) == 4:
				quad = approx
			else:
				rect = cv2.minAreaRect(contour)
				width, height = rect[1]
				rect_area = width * height
				if width < 1.0 or height < 1.0 or rect_area <= 0.0:
					continue
				if area / rect_area < 0.45:
					continue
				quad = cv2.boxPoints(rect).astype(np.int32).reshape((-1, 1, 2))

			center_px = self._contour_center(quad)
			if center_px is None:
				center_px = self._contour_center(contour)
			if center_px is None:
				continue

			candidates.append((float(area), quad, contour, center_px))

		candidates.sort(key=lambda item: item[0], reverse=True)
		return candidates

	def _replace_detected_cube_lists(self, cubes_by_color: dict[str, list[DetectedCube]]) -> None:
		with self._cube_list_lock:
			self.pink_cubes = list(cubes_by_color["pink"])
			self.yellow_cubes = list(cubes_by_color["yellow"])
			self.purple_cubes = list(cubes_by_color["purple"])
			self.cubes_by_color = {
				"pink": self.pink_cubes,
				"yellow": self.yellow_cubes,
				"purple": self.purple_cubes,
			}

	def get_detected_cubes(self, color: str) -> list[DetectedCube]:
		with self._cube_list_lock:
			return list(self.cubes_by_color.get(color, []))

	def _on_cube_lists_updated(self) -> None:
		pass

	def image_callback(self, rgb_msg: Image, depth_msg: Image) -> None:
		try:
			rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
			depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
		except CvBridgeError as error:
			self.get_logger().error(f"CvBridge conversion failed: {error}")
			return

		# Keep depth as numpy for downstream detection logic.
		if not isinstance(depth_image, np.ndarray):
			depth_image = np.array(depth_image)

		hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
		display = rgb_image.copy()
		combined_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
		detected_cubes = {color: [] for color in self.color_names}
		first_viz_payload = None
		debug_cube = None

		for color in self.color_names:
			color_mask = self._build_color_mask(hsv_image, self.color_hsv_ranges[color])
			combined_mask = cv2.bitwise_or(combined_mask, color_mask)
			draw_color = self.color_draw_bgr[color]

			for area, quad, contour, center_px in self._find_cube_candidates(color_mask):
				cv2.drawContours(display, [quad], -1, draw_color, 2)
				points_3d = self.contour_pixels_to_point_cloud(contour, depth_image)
				if points_3d.shape[0] < self.icp_min_points:
					cv2.putText(
						display,
						f"{color}: no depth",
						(center_px[0] + 10, center_px[1] - 10),
						cv2.FONT_HERSHEY_SIMPLEX,
						0.5,
						draw_color,
						1,
						cv2.LINE_AA,
					)
					continue

				virtual_before, virtual_aligned, plane_center, plane_normal, icp_rot, icp_fitness = (
					self.run_icp_and_get_normal(points_3d)
				)
				cube_index = len(detected_cubes[color])
				frame_id = self._cube_frame_id(color, cube_index)
				cube = DetectedCube(
					color=color,
					position=plane_center,
					rotation_matrix=icp_rot,
					normal_towards_camera=plane_normal,
					center_px=center_px,
					contour_area=area,
					icp_fitness=icp_fitness,
					frame_id=frame_id,
					stamp=rgb_msg.header.stamp,
				)
				detected_cubes[color].append(cube)
				self._broadcast_cube_tf(plane_center, icp_rot, rgb_msg.header.stamp, color, cube_index)

				if first_viz_payload is None:
					first_viz_payload = (
						points_3d,
						virtual_before,
						virtual_aligned,
						plane_center,
						plane_normal,
					)
					debug_cube = cube

				cv2.circle(display, center_px, 5, (0, 0, 255), -1)
				cv2.putText(
					display,
					f"{color}[{cube_index}] center=({center_px[0]}, {center_px[1]})",
					(center_px[0] + 10, center_px[1] - 10),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.5,
					draw_color,
					1,
					cv2.LINE_AA,
				)

		self._replace_detected_cube_lists(detected_cubes)
		self._on_cube_lists_updated()

		if first_viz_payload is not None:
			self.update_open3d(*first_viz_payload)

		now = self.get_clock().now()
		if (now - self.last_debug_time).nanoseconds > 2_000_000_000:
			self.last_debug_time = now
			counts = ", ".join(
				f"{color}: {len(detected_cubes[color])}" for color in self.color_names
			)
			if debug_cube is None:
				self.get_logger().info(
					f"Detected cubes: {counts}, depth dtype: {depth_image.dtype}"
				)
			else:
				normal = debug_cube.normal_towards_camera
				self.get_logger().info(
					f"Detected cubes: {counts}, depth dtype: {depth_image.dtype}, "
					f"first {debug_cube.color} normal: "
					f"[{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}], "
					f"icp_fitness: {debug_cube.icp_fitness:.3f}"
				)

		self._store_2d_visualization(display, combined_mask)
		if self.visualization_mode == "inline":
			self.render_gui()

	def _broadcast_cube_tf(
		self,
		position: np.ndarray,
		rot: np.ndarray,
		stamp,
		color: str = "cube",
		index: int | None = None,
	) -> None:
		from scipy.spatial.transform import Rotation
		qx, qy, qz, qw = Rotation.from_matrix(rot.astype(np.float64)).as_quat()
		t = TransformStamped()
		t.header.stamp = stamp
		t.header.frame_id = "camera_color_optical_frame"
		t.child_frame_id = self._cube_frame_id(color, index)
		t.transform.translation.x = float(position[0])
		t.transform.translation.y = float(position[1])
		t.transform.translation.z = float(position[2])
		t.transform.rotation.x = qx
		t.transform.rotation.y = qy
		t.transform.rotation.z = qz
		t.transform.rotation.w = qw
		self.tf_broadcaster.sendTransform(t)

	def destroy_node(self):
		if self.visualization_mode == "inline":
			self.close_gui()
		super().destroy_node()

	def close_gui(self):
		cv2.destroyAllWindows()
		if self.o3d_window_ready and self.o3d_vis is not None:
			self.o3d_vis.destroy_window()
			self.o3d_window_ready = False
			self.o3d_vis = None


def main(args=None):
	rclpy.init(args=args)
	node = CubeDetector()

	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()


if __name__ == "__main__":
	main()
