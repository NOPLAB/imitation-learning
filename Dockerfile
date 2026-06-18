# Imitation Learning for Turtlebot3 -- ROS 2 Humble
#
# Provides ROS 2 Humble (desktop), the Turtlebot3 + Gazebo simulation, the Nav2
# navigation stack (used as the expert), and a CPU build of PyTorch, with the
# colcon workspace pre-built.
FROM osrf/ros:humble-desktop-full

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# System / ROS dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-colcon-common-extensions \
        xterm \
        ros-humble-navigation2 \
        ros-humble-nav2-bringup \
        ros-humble-turtlebot3 \
        ros-humble-turtlebot3-gazebo \
        ros-humble-turtlebot3-msgs \
        ros-humble-gazebo-ros-pkgs \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python / ML dependencies (CPU PyTorch by default)
# ---------------------------------------------------------------------------
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        -r /tmp/requirements.txt

# ---------------------------------------------------------------------------
# Build the workspace
# ---------------------------------------------------------------------------
WORKDIR /ros2_ws
COPY ros2_ws/src ./src
RUN source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV TURTLEBOT3_MODEL=waffle
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> /root/.bashrc && \
    echo "export TURTLEBOT3_MODEL=waffle" >> /root/.bashrc

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
