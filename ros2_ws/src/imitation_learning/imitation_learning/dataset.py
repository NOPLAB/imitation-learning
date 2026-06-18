"""PyTorch dataset that loads recorded expert trajectories.

Each trajectory is a JSON file (``trajectory_<i>.json``) holding three parallel
lists: ``robot_pos`` (7 values), ``laser_scan`` (full scan) and ``robot_vel``
(2 values).  The laser scans are sub-sampled to match the network input.
"""

import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from imitation_learning.neural_network import LASER_SUBSAMPLE


class ImitationDataset(Dataset):
    def __init__(self, data_dir: str, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.poses = np.empty((0, 7), dtype=np.float32)
        self.scans = None
        self.velocities = np.empty((0, 2), dtype=np.float32)

        traj_files = sorted(glob.glob(os.path.join(data_dir, "trajectory_*.json")))
        poses, scans, vels = [], [], []
        for path in traj_files:
            with open(path, "r") as f:
                traj = json.load(f)
            poses.append(np.asarray(traj["robot_pos"], dtype=np.float32))
            scans.append(np.asarray(traj["laser_scan"], dtype=np.float32))
            vels.append(np.asarray(traj["robot_vel"], dtype=np.float32))

        if poses:
            self.poses = np.concatenate(poses, axis=0)
            self.scans = np.concatenate(scans, axis=0)
            self.velocities = np.concatenate(vels, axis=0)

            # Replace inf/nan with 0 and sub-sample the laser exactly like predict().
            self.scans[~np.isfinite(self.scans)] = 0.0
            self.scans = self.scans[:, ::LASER_SUBSAMPLE]

            self.poses = torch.as_tensor(self.poses, dtype=torch.float32, device=device)
            self.scans = torch.as_tensor(self.scans, dtype=torch.float32, device=device)
            self.velocities = torch.as_tensor(self.velocities, dtype=torch.float32, device=device)

    def __len__(self):
        return self.poses.shape[0]

    def __getitem__(self, i):
        return self.poses[i, :], self.scans[i, :], self.velocities[i, :]
