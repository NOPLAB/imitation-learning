"""Standalone policy execution node (ROS 2 Humble).

Loads a trained policy (``best_model.pt`` by default) and drives the robot from
the start pose to the goal once, then resets the robot and stops.  This is the
ROS 2 port of the original ``policyexec`` script and does not need the expert /
Nav2 planner running.
"""

import os

import numpy as np
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

import rclpy
from rclpy.node import Node

from imitation_learning import model_il
from imitation_learning.model_il import ImitationNet


class PolicyExec(Node):
    def __init__(self):
        super().__init__("policyexec")
        self.declare_parameter("model_dir", os.path.expanduser("~/il_data/models"))
        self.declare_parameter("model_file", "best_model.pt")
        self.declare_parameter("device", "auto")
        self.declare_parameter("control_rate", 20.0)
        self.declare_parameter("goal_x", 6.0)
        self.declare_parameter("goal_y", -1.0)
        self.declare_parameter("goal_tol_x", 1.0)
        self.declare_parameter("goal_tol_y", 0.3)

        import torch
        dev = self.get_parameter("device").value
        self.device = "cuda" if (dev != "cpu" and torch.cuda.is_available()) else "cpu"

        self.laser = None
        self.pos = None
        self.done = False

        self.model = ImitationNet(control_dim=2, device=self.device).to(self.device)
        model_path = os.path.join(self.get_parameter("model_dir").value,
                                  self.get_parameter("model_file").value)
        if os.path.exists(model_path):
            self.model.load(model_path)
            self.get_logger().info(f"Loaded policy from {model_path}")
        else:
            self.get_logger().warn(f"Model not found at {model_path}; running an untrained policy.")
        self.model.eval()

        self.create_subscription(LaserScan, "scan", self._laser_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, "amcl_pose", self._amcl_cb, 10)
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_reset = self.create_publisher(Bool, "reset_robot_pos", 10)

        rate = float(self.get_parameter("control_rate").value)
        self.create_timer(1.0 / rate, self._on_timer)
        self.get_logger().info("Starting policy execution.")

    def _laser_cb(self, msg):
        self.laser = msg

    def _amcl_cb(self, msg):
        self.pos = msg

    def _on_timer(self):
        if self.done or self.laser is None or self.pos is None:
            return

        p = self.pos.pose.pose
        odom_input = np.array([[p.position.x, p.position.y, p.position.z,
                                p.orientation.x, p.orientation.y,
                                p.orientation.z, p.orientation.w]])
        laser_scan = np.array([list(self.laser.ranges)], dtype=np.float32)
        vel = model_il.predict(self.model, odom_input, laser_scan, self.device)

        cmd = Twist()
        cmd.linear.x = float(vel[0, 0])
        cmd.angular.z = float(vel[0, 1])
        self.pub_cmd.publish(cmd)

        gx = self.get_parameter("goal_x").value
        gy = self.get_parameter("goal_y").value
        tx = self.get_parameter("goal_tol_x").value
        ty = self.get_parameter("goal_tol_y").value
        if abs(p.position.x - gx) < tx and abs(p.position.y - gy) < ty:
            self.pub_cmd.publish(Twist())
            self.get_logger().info("Goal reached. Resetting robot. Done.")
            self.pub_reset.publish(Bool(data=True))
            self.pub_reset.publish(Bool(data=True))
            self.done = True


def main(args=None):
    rclpy.init(args=args)
    node = PolicyExec()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
