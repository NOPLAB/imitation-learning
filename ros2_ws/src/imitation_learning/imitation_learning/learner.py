"""Supervised imitation-learning node (ROS 2 Humble).

Repeatedly collects expert demonstrations -- the Nav2 plan with a little
additive steering noise -- and, when the user switches to TRAIN, fits the
policy on everything collected so far.  This is the ROS 2 port of the original
``learner`` script.
"""

import numpy as np
from geometry_msgs.msg import Twist

import rclpy

from imitation_learning.base_node import ImitationLearnerNode


class SupervisedLearner(ImitationLearnerNode):
    mode_name = "supervised"

    def __init__(self):
        super().__init__("learner")
        self._noise_count = 0
        self._noise = 0.0

    def select_command(self):
        # Always follow the expert; inject angular noise that is refreshed
        # every 100 steps so demonstrations cover slightly off-policy states.
        cmd = Twist()
        cmd.linear.x = self.expert_cmd.linear.x
        if self._noise_count % 100 == 0:
            if np.random.randint(2):
                self._noise = np.random.uniform(low=0.1, high=0.2)
            else:
                self._noise = np.random.uniform(low=-0.2, high=-0.1)
        cmd.angular.z = self.expert_cmd.angular.z + self._noise
        self._noise_count += 1
        self.robot_cmd = cmd
        return cmd


def main(args=None):
    rclpy.init(args=args)
    node = SupervisedLearner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
