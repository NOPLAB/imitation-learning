"""Bring up the Turtlebot3 Gazebo world and the Nav2 stack (the expert).

The key trick for imitation learning is that Nav2's controller output is
remapped from ``/cmd_vel`` to ``/cmd_vel_temp`` so the learner node can
intercept the expert commands and decide what is actually sent to the robot on
``/cmd_vel``.

Arguments:
    world      : turtlebot3 gazebo launch file to use (default house)
    map        : map yaml for Nav2 (default: this package's maps/house_map.yaml)
    use_sim_time, headless, params_file
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction,
                            IncludeLaunchDescription)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import SetRemap


def generate_launch_description():
    pkg_il = get_package_share_directory("imitation_learning")
    pkg_tb3_gazebo = get_package_share_directory("turtlebot3_gazebo")
    pkg_nav2 = get_package_share_directory("nav2_bringup")

    use_sim_time = LaunchConfiguration("use_sim_time")
    map_yaml = LaunchConfiguration("map")
    params_file = LaunchConfiguration("params_file")
    world_launch = LaunchConfiguration("world_launch")

    default_map = os.path.join(pkg_il, "maps", "house_map.yaml")

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [pkg_tb3_gazebo, "/launch/", world_launch]),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    # Remap the expert (Nav2 controller) output so the learner can intercept it.
    nav2 = GroupAction([
        SetRemap(src="/cmd_vel", dst="/cmd_vel_temp"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_nav2, "launch", "bringup_launch.py")),
            launch_arguments={
                "use_sim_time": use_sim_time,
                "map": map_yaml,
                "params_file": params_file,
            }.items(),
        ),
    ])

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("world_launch", default_value="turtlebot3_house.launch.py",
                              description="launch file inside turtlebot3_gazebo/launch"),
        DeclareLaunchArgument("map", default_value=default_map),
        DeclareLaunchArgument(
            "params_file",
            default_value=os.path.join(pkg_il, "config", "nav2_params.yaml")),
        gazebo,
        nav2,
    ])
