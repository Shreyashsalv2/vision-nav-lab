"""In-process test: play back a short synthetic sequence and check the node publishes sensible odometry."""
import time

import pytest
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter

from vnav import synthetic


@pytest.fixture
def ros():
    rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def test_node_publishes_forward_odometry(ros, tmp_path):
    from vnav_ros.vo_node import VoNode

    synthetic.generate(tmp_path, synthetic.SyntheticConfig(frames=15, seed=3))
    node = VoNode(parameter_overrides=[Parameter("sequence_dir", value=str(tmp_path)), Parameter("rate_hz", value=50.0)])
    listener = rclpy.create_node("odom_listener")
    got = []
    listener.create_subscription(Odometry, "vnav/odom", got.append, 20)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    ex.add_node(listener)
    deadline = time.time() + 30
    while len(got) < 15 and time.time() < deadline:
        ex.spin_once(timeout_sec=0.1)
    node.destroy_node()
    listener.destroy_node()

    assert len(got) >= 12
    last = got[-1]
    assert last.header.frame_id == "odom" and last.child_frame_id == "base_link"
    # 14 steps of 0.3 m, mostly straight ahead: body +x is forward
    assert 3.0 < last.pose.pose.position.x < 5.0
    assert abs(last.pose.pose.position.y) < 1.5
