"""Imitation-learning model wrapper plus train / predict helpers.

Ported from the original ROS 1 ``model_il.py`` to ROS 2 Humble.  All of the
``rospkg`` / ``argparse`` global state of the original has been removed: the
functions here take explicit directories and hyper-parameters so they can be
driven from the rclpy nodes (or a plain Python script).
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from imitation_learning import neural_network as neural_net
from imitation_learning.dataset import ImitationDataset
from imitation_learning.neural_network import LASER_SUBSAMPLE  # re-exported for callers


class ImitationNet(nn.Module):
    """Policy network that maps robot pose + laser scan to a velocity command."""

    def __init__(self, control_dim: int = 2, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.pose_to_control = neural_net.PoseToControl(control_dim=control_dim)

    def forward(self, odom_input, laser_scan):
        return self.pose_to_control(odom_input, laser_scan)

    def step(self, odom_input, laser_scan, target_control):
        """Sum-of-squared-errors loss over linear and angular velocity."""
        control_predict = self.forward(odom_input, laser_scan)
        return ((control_predict - target_control).pow(2)).sum()

    def save(self, path):
        torch.save({"model": self.state_dict()}, path)

    def load(self, path):
        path = os.path.abspath(path)
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["model"])


def train(
    model: ImitationNet,
    data_dir: str,
    model_dir: str,
    mode: str = "dagger",
    device: str = "cpu",
    epochs: int = None,
    lr: float = 0.002,
    lr_decay: float = 0.1,
    batch_size: int = 50,
    logger=None,
) -> ImitationNet:
    """Train ``model`` on every trajectory collected so far and save it.

    ``data_dir`` / ``model_dir`` should already include the mode sub-folder
    (e.g. ``.../data/dagger/``).  The trained weights are written to
    ``<model_dir>/model.pt``.
    """
    if epochs is None:
        epochs = 50 if mode == "dagger" else 150

    os.makedirs(model_dir, exist_ok=True)

    train_data = ImitationDataset(data_dir=data_dir, device=device)
    if len(train_data) == 0:
        _log(logger, "No training data found - skipping training round.")
        return model

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=lr_decay)
    model.train()

    for epoch in range(epochs):
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
        losses = []
        for pose, scan, vel in train_loader:
            opt.zero_grad()
            loss = model.step(odom_input=pose, laser_scan=scan, target_control=vel)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        _log(logger, "Epoch {:>4}  train loss: {:10.6f}".format(epoch, float(np.mean(losses))))

    model.save(os.path.join(model_dir, "model.pt"))
    return model


def predict(model: ImitationNet, odom_input, laser_scan, device: str = "cpu"):
    """Run the policy for a single observation and return an (1, 2) numpy array."""
    odom_input = torch.as_tensor(odom_input, dtype=torch.float32, device=device)

    laser_scan = np.asarray(laser_scan, dtype=np.float32)
    laser_scan[~np.isfinite(laser_scan)] = 0.0
    laser_scan = laser_scan[:, ::LASER_SUBSAMPLE]
    laser_scan = torch.as_tensor(laser_scan, dtype=torch.float32, device=device)

    with torch.no_grad():
        vel = model(odom_input, laser_scan)
    return vel.cpu().numpy()


def _log(logger, msg):
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)
