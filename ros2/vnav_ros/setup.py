from glob import glob

from setuptools import setup

package_name = "vnav_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Shreyash Singh Salv",
    maintainer_email="shreyashsalv@gmail.com",
    description="ROS2 wrapper for vision-nav-lab monocular visual odometry",
    license="MIT",
    tests_require=["pytest"],
    entry_points={"console_scripts": ["vo_node = vnav_ros.vo_node:main"]},
)
