"""Reward terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_upright_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Squared tilt: x/y components of gravity in the body frame (0 when perfectly upright)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def base_upright_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float = 0.1) -> torch.Tensor:
    """Bonus in (0, 1], equal to 1 when perfectly upright."""
    return torch.exp(-base_upright_penalty(env, asset_cfg) / std**2)


def ang_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Squared yaw rate (spinning)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])


def velocity_command_error_l2(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, norm_scale: float = 0.2
) -> torch.Tensor:
    """Squared error between the target and the true forward velocity, in units of ``norm_scale`` m/s."""
    command = env.command_manager.get_command(command_name)[:, 0]
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square((command - asset.data.root_lin_vel_b[:, 0]) / norm_scale)


def velocity_command_tracking_bonus(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, std: float = 0.05
) -> torch.Tensor:
    """Bonus in (0, 1], equal to 1 when the forward velocity matches the target."""
    command = env.command_manager.get_command(command_name)[:, 0]
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.exp(-torch.square(command - asset.data.root_lin_vel_b[:, 0]) / std**2)


def yaw_command_error_l2(env: ManagerBasedRLEnv, command_name: str, norm_scale: float = 1.0) -> torch.Tensor:
    """Squared heading error (shortest angle, so 179 deg vs -179 deg = 2 deg), in units of ``norm_scale`` rad."""
    return torch.square(env.command_manager.get_term(command_name).yaw_error / norm_scale)


def yaw_command_tracking_bonus(env: ManagerBasedRLEnv, command_name: str, std: float = 0.2) -> torch.Tensor:
    """Bonus in (0, 1], equal to 1 when the robot faces the target heading."""
    return torch.exp(-torch.square(env.command_manager.get_term(command_name).yaw_error) / std**2)
