"""DAgger imitation-learning node (ROS 2 Humble).

DAgger (Dataset Aggregation) mixes the expert and the current policy while
collecting data: with probability ``beta`` the expert drives, otherwise the
learned policy drives (but only if its command is close enough to the expert's,
otherwise we fall back to the expert).  ``beta`` decays after every completed
episode so the robot increasingly visits states induced by its own policy --
the key idea that lets DAgger fix the distribution-shift problem of plain
supervised imitation learning.

This is the ROS 2 port of the original ``daglearner`` script.
"""

import os
import random

from geometry_msgs.msg import Twist

import rclpy

from imitation_learning import model_il
from imitation_learning.base_node import ImitationLearnerNode

# How far the policy command may differ from the expert before we discard it.
LIN_THRESH = 0.2
ANG_THRESH = 0.3


class DaggerLearner(ImitationLearnerNode):
    mode_name = "dagger"

    def __init__(self):
        super().__init__("daglearner")
        self.declare_parameter("beta", 0.9)
        self.declare_parameter("beta_decay", 0.95)
        self.beta = float(self.get_parameter("beta").value)
        self.beta_decay = float(self.get_parameter("beta_decay").value)
        self._fallback_count = 0
        self._policy_loaded = False

    def _ensure_policy_loaded(self):
        if self._policy_loaded:
            return
        model_path = os.path.join(self.model_dir, "model.pt")
        if os.path.exists(model_path):
            self.model.load(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        self._policy_loaded = True

    def select_command(self):
        self._ensure_policy_loaded()
        cmd = Twist()

        if random.random() <= self.beta:
            # Expert drives.
            cmd.linear.x = self.expert_cmd.linear.x
            cmd.angular.z = self.expert_cmd.angular.z
            self.robot_cmd = cmd
            return cmd

        # Policy drives -- but sanity-check against the expert.
        vel = model_il.predict(self.model, self._odom_vector(), self._laser_matrix(), self.device)
        lin, ang = float(vel[0, 0]), float(vel[0, 1])
        if (abs(self.expert_cmd.linear.x - lin) < LIN_THRESH and
                abs(self.expert_cmd.angular.z - ang) < ANG_THRESH):
            cmd.linear.x = lin
            cmd.angular.z = ang
            self.robot_cmd = cmd
            self._fallback_count = 0
            return cmd

        # Policy deviated too much: hold previous command briefly, then fall
        # back to the expert so the robot does not get stuck.
        self._fallback_count += 1
        if self._fallback_count < 50:
            return self.robot_cmd
        cmd.linear.x = self.expert_cmd.linear.x
        cmd.angular.z = self.expert_cmd.angular.z
        self.robot_cmd = cmd
        return cmd

    def on_episode_complete(self):
        self.beta *= self.beta_decay
        self._policy_loaded = False  # reload the freshly trained weights next time
        self.get_logger().info(f"Episode complete. beta -> {self.beta:.4f}")


def main(args=None):
    rclpy.init(args=args)
    node = DaggerLearner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
