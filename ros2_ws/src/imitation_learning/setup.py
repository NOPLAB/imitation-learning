import os
from glob import glob

from setuptools import find_packages, setup

package_name = "imitation_learning"

setup(
    name=package_name,
    version="2.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
        (os.path.join("share", package_name, "maps"), glob("maps/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="noplab",
    maintainer_email="noplab90@gmail.com",
    description="Imitation learning (DAgger / supervised) for Turtlebot3 on ROS 2 Humble.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "learner = imitation_learning.learner:main",
            "daglearner = imitation_learning.daglearner:main",
            "policyexec = imitation_learning.policyexec:main",
            "reset_robot_pos = imitation_learning.reset_robot_pos:main",
            "teleop_mode_key = imitation_learning.teleop_mode_key:main",
        ],
    },
)
