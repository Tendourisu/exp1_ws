#!/usr/bin/env python3

import cv2
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import numpy as np
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

		self.get_logger().info("CubeDetector node started.")

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
		best_area = 0.0

		for contour in contours:
			area = cv2.contourArea(contour)
			if area < self.min_contour_area:
				continue

			perimeter = cv2.arcLength(contour, True)
			approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

			if len(approx) == 4 and area > best_area:
				best_quad = approx
				best_area = area

		display = rgb_image.copy()
		if best_quad is not None:
			cv2.drawContours(display, [best_quad], -1, (0, 255, 0), 2)

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
