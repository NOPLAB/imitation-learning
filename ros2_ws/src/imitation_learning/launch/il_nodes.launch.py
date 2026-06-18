"""Launch the imitation-learning nodes (reset + learner/dagger + teleop helper).

This launch assumes the simulation and Nav2 are already running (see
``simulation.launch.py`` or the README).  Select the algorithm with
``algo:=dagger`` (default) or ``algo:=supervised``.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    algo = LaunchConfiguration("algo")
    use_sim_time = LaunchConfiguration("use_sim_time")
    data_dir = LaunchConfiguration("data_dir")
    model_dir = LaunchConfiguration("model_dir")
    entity_name = LaunchConfiguration("entity_name")
    start_teleop = LaunchConfiguration("start_teleop")

    default_root = os.path.expanduser("~/il_data")

    is_dagger = PythonExpression(["'", algo, "' == 'dagger'"])
    is_supervised = PythonExpression(["'", algo, "' == 'supervised'"])

    return LaunchDescription([
        DeclareLaunchArgument("algo", default_value="dagger",
                              description="dagger or supervised"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("data_dir", default_value=os.path.join(default_root, "data")),
        DeclareLaunchArgument("model_dir", default_value=os.path.join(default_root, "models")),
        DeclareLaunchArgument("entity_name", default_value="turtlebot3_waffle"),
        DeclareLaunchArgument("start_teleop", default_value="false",
                              description="also start the keyboard mode helper"),

        Node(
            package="imitation_learning",
            executable="reset_robot_pos",
            name="reset_robot_pos",
            output="screen",
            parameters=[{"use_sim_time": use_sim_time, "entity_name": entity_name}],
        ),
        Node(
            package="imitation_learning",
            executable="daglearner",
            name="daglearner",
            output="screen",
            condition=IfCondition(is_dagger),
            parameters=[{"use_sim_time": use_sim_time,
                         "data_dir": data_dir, "model_dir": model_dir}],
        ),
        Node(
            package="imitation_learning",
            executable="learner",
            name="learner",
            output="screen",
            condition=IfCondition(is_supervised),
            parameters=[{"use_sim_time": use_sim_time,
                         "data_dir": data_dir, "model_dir": model_dir}],
        ),
        Node(
            package="imitation_learning",
            executable="teleop_mode_key",
            name="teleop_mode_key",
            output="screen",
            prefix="xterm -e",
            condition=IfCondition(start_teleop),
        ),
    ])
