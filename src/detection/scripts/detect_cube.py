#!/usr/bin/env python3

import cv2
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import numpy as np
import open3d as o3d
import rclpy
import threading
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


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

		# Target the pink cube with dual HSV hue ranges.
		self.pink_hsv_ranges = [
			(np.array([0, 70, 80], dtype=np.uint8), np.array([12, 255, 255], dtype=np.uint8)),
			(np.array([145, 70, 80], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8)),
		]
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
		mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
		for lower, upper in self.pink_hsv_ranges:
			mask = cv2.bitwise_or(mask, cv2.inRange(hsv_image, lower, upper))

		# Light cleanup to reduce isolated noise in the binary mask.
		kernel = np.ones((5, 5), dtype=np.uint8)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

		contours_result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours = contours_result[0] if len(contours_result) == 2 else contours_result[1]

		best_quad = None
		best_contour = None
		best_area = 0.0

		for contour in contours:
			area = cv2.contourArea(contour)
			if area < self.min_contour_area:
				continue

			perimeter = cv2.arcLength(contour, True)
			approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

			if len(approx) == 4 and area > best_area:
				best_quad = approx
				best_contour = contour
				best_area = area

		display = rgb_image.copy()
		if best_quad is not None and best_contour is not None:
			cv2.drawContours(display, [best_quad], -1, (0, 255, 0), 2)

			points_3d = self.contour_pixels_to_point_cloud(best_contour, depth_image)
			if points_3d.shape[0] >= self.icp_min_points:
				virtual_before, virtual_aligned, plane_center, plane_normal, icp_rot, icp_fitness = self.run_icp_and_get_normal(points_3d)
				self.update_open3d(points_3d, virtual_before, virtual_aligned, plane_center, plane_normal)
				self._broadcast_cube_tf(plane_center, icp_rot, rgb_msg.header.stamp)
			else:
				plane_normal = np.array([0.0, 0.0, 0.0], dtype=np.float32)
				icp_fitness = 0.0

			now = self.get_clock().now()
			if (now - self.last_debug_time).nanoseconds > 2_000_000_000:
				self.last_debug_time = now
				self.get_logger().info(
					"3D points in contour: "
					f"{points_3d.shape[0]}, depth dtype: {depth_image.dtype}, "
					f"normal: [{plane_normal[0]:.3f}, {plane_normal[1]:.3f}, {plane_normal[2]:.3f}], "
					f"icp_fitness: {icp_fitness:.3f}"
				)

			moments = cv2.moments(best_quad)
			if moments["m00"] != 0:
				cx = int(moments["m10"] / moments["m00"])
				cy = int(moments["m01"] / moments["m00"])
				cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
				cv2.putText(
					display,
					f"center=({cx}, {cy})",
					(cx + 10, cy - 10),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.5,
					(0, 0, 255),
					1,
					cv2.LINE_AA,
				)

		self._store_2d_visualization(display, mask)
		if self.visualization_mode == "inline":
			self.render_gui()

	def _broadcast_cube_tf(self, position: np.ndarray, rot: np.ndarray, stamp) -> None:
		from scipy.spatial.transform import Rotation
		qx, qy, qz, qw = Rotation.from_matrix(rot.astype(np.float64)).as_quat()
		t = TransformStamped()
		t.header.stamp = stamp
		t.header.frame_id = "camera_color_optical_frame"
		t.child_frame_id = "cube"
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
