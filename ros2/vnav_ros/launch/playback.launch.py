"""Play back a recorded sequence through the VO node:  ros2 launch vnav_ros playback.launch.py sequence_dir:=/data/synthetic"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("sequence_dir", description="folder with rgb.txt, images/, camera.json"),
        DeclareLaunchArgument("rate_hz", default_value="10.0"),
        Node(
            package="vnav_ros",
            executable="vo_node",
            name="vnav_vo",
            output="screen",
            parameters=[{"sequence_dir": LaunchConfiguration("sequence_dir"), "rate_hz": LaunchConfiguration("rate_hz")}],
        ),
    ])
