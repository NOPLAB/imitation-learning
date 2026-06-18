#!/usr/bin/env bash
set -e

# Source ROS 2 and the built workspace, then hand off to the command.
source /opt/ros/humble/setup.bash
if [ -f /ros2_ws/install/setup.bash ]; then
  source /ros2_ws/install/setup.bash
fi

# Turtlebot3 / Gazebo environment.
export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-waffle}"
export GAZEBO_MODEL_PATH="${GAZEBO_MODEL_PATH}:/opt/ros/humble/share/turtlebot3_gazebo/models"

exec "$@"
