"""Neural-network architecture for the imitation-learning policy.

The policy is a small fully-connected network that maps a concatenation of the
robot pose (7 values: position xyz + orientation quaternion xyzw) and a
sub-sampled laser scan (12 values) to a 2-D control command
(linear.x, angular.z).

This module is intentionally free of any ROS dependency so it can be unit
tested and reused outside of a running ROS 2 graph.
"""

import torch
import torch.nn as nn

# Every Nth laser range is kept (a 360-beam scan becomes 12 values).
LASER_SUBSAMPLE = 30
# 7 pose values + 12 sub-sampled laser ranges = 19 inputs
DEFAULT_INPUT_DIM = 19


class PoseToControl(nn.Module):
    """Feed-forward network: (pose, laser) -> control command."""

    def __init__(self, control_dim: int = 2, input_dim: int = DEFAULT_INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, control_dim),
        )

    def forward(self, odom_input: torch.Tensor, laser_scan: torch.Tensor) -> torch.Tensor:
        total_input = torch.cat((odom_input, laser_scan), dim=1)
        return self.net(total_input)
