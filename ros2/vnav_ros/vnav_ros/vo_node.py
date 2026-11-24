"""ROS2 node: monocular visual odometry.

Two input modes:
  * playback of a recorded sequence folder (rgb.txt + images, optional groundtruth.txt for step scale):
        ros2 run vnav_ros vo_node --ros-args -p sequence_dir:=/data/synthetic
  * live images on camera/image_raw (sensor_msgs/Image), scaled by the step_scale parameter.

Publishes nav_msgs/Odometry on vnav/odom, nav_msgs/Path on vnav/path, and the odom -> base_link transform.
Poses are converted from the camera's optical convention to ROS body convention (REP-103) with vnav.frames.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as PathMsg
from rclpy.node import Node
from sensor_msgs.msg import Image
from tf2_ros import TransformBroadcaster

from vnav import frames, io, se3
from vnav.camera import PinholeCamera
from vnav.odometry import VisualOdometry, step_lengths


class VoNode(Node):
    def __init__(self, **kwargs):
        super().__init__("vnav_vo", **kwargs)
        self.declare_parameter("sequence_dir", "")
        self.declare_parameter("camera_file", "")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("step_scale", 1.0)       # metres per frame when ground truth is not used
        self.declare_parameter("use_gt_scale", True)    # take step lengths from groundtruth.txt if present
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.odom_frame, self.base_frame = p("odom_frame"), p("base_frame")
        self.step_scale = float(p("step_scale"))
        seq_dir = Path(p("sequence_dir")) if p("sequence_dir") else None
        cam_file = p("camera_file") or (str(seq_dir / "camera.json") if seq_dir else "")
        if not cam_file:
            raise ValueError("set camera_file (or sequence_dir containing camera.json)")
        self.vo = VisualOdometry(PinholeCamera.load(cam_file))

        self.odom_pub = self.create_publisher(Odometry, "vnav/odom", 10)
        self.path_pub = self.create_publisher(PathMsg, "vnav/path", 10)
        self.tf = TransformBroadcaster(self)
        self.path = PathMsg()
        self.path.header.frame_id = self.odom_frame
        self.frames_done = 0

        if seq_dir:
            _, files = io.read_image_list(seq_dir / "rgb.txt")
            self.files = [seq_dir / f for f in files]
            gt = seq_dir / "groundtruth.txt"
            if p("use_gt_scale") and gt.exists():
                self.scales = step_lengths(io.read_tum(gt)[1])
            else:
                self.scales = np.full(len(self.files), self.step_scale)
            self.idx = 0
            self.timer = self.create_timer(1.0 / float(p("rate_hz")), self._on_timer)
            self.get_logger().info(f"playing back {len(self.files)} frames from {seq_dir}")
        else:
            from cv_bridge import CvBridge

            self.bridge = CvBridge()
            self.create_subscription(Image, "camera/image_raw", self._on_image, 10)
            self.get_logger().info("waiting for images on camera/image_raw")

    # ---- inputs ----
    def _on_timer(self) -> None:
        if self.idx >= len(self.files):
            self.timer.cancel()
            self.get_logger().info(f"playback finished: {self.frames_done} frames")
            return
        img = cv2.imread(str(self.files[self.idx]), cv2.IMREAD_GRAYSCALE)
        scale = float(self.scales[self.idx]) if self.idx < len(self.scales) else self.step_scale
        self.idx += 1
        if img is not None:
            self._process(img, scale)

    def _on_image(self, msg: Image) -> None:
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        self._process(img, self.step_scale, msg.header.stamp)

    # ---- odometry and outputs ----
    def _process(self, gray: np.ndarray, scale: float, stamp=None) -> None:
        T_opt, info = self.vo.step(gray, scale)
        T = frames.optical_to_body(T_opt)
        stamp = stamp or self.get_clock().now().to_msg()
        q = se3.rot_to_quat(T[:3, :3])
        x, y, z = (float(v) for v in T[:3, 3])

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = x, y, z
        o = odom.pose.pose.orientation
        o.x, o.y, o.z, o.w = (float(v) for v in q)
        self.odom_pub.publish(odom)

        ps = PoseStamped()
        ps.header = odom.header
        ps.pose = odom.pose.pose
        self.path.header.stamp = stamp
        self.path.poses.append(ps)
        self.path_pub.publish(self.path)

        tf = TransformStamped()
        tf.header = odom.header
        tf.child_frame_id = self.base_frame
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = x, y, z
        tf.transform.rotation = odom.pose.pose.orientation
        self.tf.sendTransform(tf)

        self.frames_done += 1
        if info.used_fallback:
            self.get_logger().warn(f"frame {self.frames_done}: weak geometry, assumed constant velocity")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
