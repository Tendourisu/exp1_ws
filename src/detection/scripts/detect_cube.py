#!/usr/bin/env python3

import cv2
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class CubeDetector(Node):
	def __init__(self):
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
		self.last_debug_time = self.get_clock().now()

		self.o3d_vis = o3d.visualization.Visualizer()
		self.o3d_window_ready = self.o3d_vis.create_window(
			window_name="Cube Point Cloud",
			width=960,
			height=540,
		)
		self.o3d_pcd = o3d.geometry.PointCloud()
		self.view_initialized = False
		if self.o3d_window_ready:
			self.o3d_vis.add_geometry(self.o3d_pcd)
			render_option = self.o3d_vis.get_render_option()
			render_option.point_size = 6.0
			render_option.background_color = np.array([0.08, 0.08, 0.08])
		else:
			self.get_logger().warn("Open3D window creation failed, skip 3D visualization.")

		self.get_logger().info("CubeDetector node started.")

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

	def update_open3d(self, points_3d: np.ndarray) -> None:
		if not self.o3d_window_ready:
			return
		if points_3d.size == 0:
			return

		self.o3d_pcd.points = o3d.utility.Vector3dVector(points_3d.astype(np.float64))
		point_colors = np.tile(np.array([[1.0, 0.2, 0.2]], dtype=np.float64), (points_3d.shape[0], 1))
		self.o3d_pcd.colors = o3d.utility.Vector3dVector(point_colors)
		self.o3d_vis.update_geometry(self.o3d_pcd)

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
			if points_3d.size > 0:
				self.update_open3d(points_3d)

			now = self.get_clock().now()
			if (now - self.last_debug_time).nanoseconds > 2_000_000_000:
				self.last_debug_time = now
				self.get_logger().info(
					f"3D points in contour: {points_3d.shape[0]}, depth dtype: {depth_image.dtype}"
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

		cv2.imshow("CubeDetector RGB", display)
		cv2.imshow("CubeDetector Mask", mask)
		cv2.waitKey(1)

	def destroy_node(self):
		cv2.destroyAllWindows()
		if self.o3d_window_ready:
			self.o3d_vis.destroy_window()
		super().destroy_node()


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
