"""Observation terms: what the robot's IMU and encoders can measure."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def imu_pitch_angle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Tilt about the wheel axis (rad), from gravity seen in the body frame. 0 = upright, tilting
    toward +X (forward) is negative. Shape (N, 1)."""
    pg = env.scene["robot"].data.projected_gravity_b  # [0, 0, -1] when upright
    return torch.atan2(-pg[:, 0], -pg[:, 2]).unsqueeze(1)


def imu_pitch_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Angular velocity about the body Y axis (rad/s), what the gyro measures. Note: this is the
    NEGATIVE of d(pitch angle)/dt. Shape (N, 1)."""
    return env.scene["robot"].data.root_ang_vel_b[:, 1].unsqueeze(1)


def imu_yaw_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Angular velocity about the body Z axis (rad/s), + = turning left. Shape (N, 1)."""
    return env.scene["robot"].data.root_ang_vel_b[:, 2].unsqueeze(1)


def relative_yaw(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Yaw (rad, [-pi, pi]) relative to the heading the robot spawned with, + = rotated left. The
    spawn heading is stored by the UniformYawCommand term ``command_name``. Shape (N, 1)."""
    return env.command_manager.get_term(command_name).relative_yaw.unsqueeze(1)


def last_action_index(env: ManagerBasedRLEnv, index: int) -> torch.Tensor:
    """Previous action of one wheel (0 = left, 1 = right), in [-1, 1]. Shape (N, 1)."""
    return env.action_manager.action[:, index].unsqueeze(1)
