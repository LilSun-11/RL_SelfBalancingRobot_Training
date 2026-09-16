# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

from .observations import wheel_distance

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_upright_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tilt, based on projected_gravity_b (world Z projected into the body frame — [0,0,1]
    when perfectly upright). Penalizes both X (roll) and Y (pitch)."""
    asset: Articulation = env.scene[asset_cfg.name]
    proj_grav = asset.data.projected_gravity_b
    return torch.sum(torch.square(proj_grav[:, :2]), dim=1)


def base_upright_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float = 0.2) -> torch.Tensor:
    """Exponential bonus for standing upright: 1.0 when perfectly upright, decaying toward 0 with
    tilt. Complements base_upright_penalty (an unbounded penalty)."""
    asset: Articulation = env.scene[asset_cfg.name]
    proj_grav = asset.data.projected_gravity_b
    tilt_sq = torch.sum(torch.square(proj_grav[:, :2]), dim=1)
    return torch.exp(-tilt_sq / std**2)


def ang_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize yaw (rotation about Z), squared angular velocity.

    With independent per-wheel torque (see ActionsCfg), the robot has an actual mechanism to spin in
    place, and nothing else stops it from doing so -- this term penalizes that directly."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])


def yaw_angle_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize squared current yaw angle — catches accumulated drift that ang_vel_z_l2 (rate only)
    would miss. Yaw always starts at 0 (reset_base only randomizes "pitch"), so no need to track a
    spawn baseline."""
    asset: Articulation = env.scene[asset_cfg.name]
    _, _, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.square(yaw)


def lin_vel_x_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize motion along X (body frame), squared linear velocity."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 0])


def lin_vel_x_normalized_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, norm_scale: float = 0.05
) -> torch.Tensor:
    """Like lin_vel_x_l2 but divides velocity by ``norm_scale`` (m/s) before squaring, bringing it to
    the same O(1) scale as rewards like upright so weights stay comparable."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 0] / norm_scale)


def distance_command_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    wheel_radius: float = 0.034,
    norm_scale: float = 0.05,
) -> torch.Tensor:
    """Penalize squared error (normalized by ``norm_scale``, meters) between target distance and
    actual distance traveled (wheel encoder). norm_scale defaults to the out_of_range threshold
    (0.05 m), so a normalized value of 1.0 = right at that threshold."""
    command = env.command_manager.get_command(command_name)[:, 0]
    traveled = wheel_distance(env, asset_cfg, wheel_radius)[:, 0]
    return torch.square((command - traveled) / norm_scale)


def distance_command_tracking_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float = 0.05,
    wheel_radius: float = 0.034,
) -> torch.Tensor:
    """Exponential bonus (complementing distance_command_error_l2) for being close to the target
    distance (wheel position)."""
    command = env.command_manager.get_command(command_name)[:, 0]
    traveled = wheel_distance(env, asset_cfg, wheel_radius)[:, 0]
    return torch.exp(-torch.square(command - traveled) / std**2)


def true_position_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    norm_scale: float = 0.05,
) -> torch.Tensor:
    """Like distance_command_error_l2 but uses the TRUE position (root_pos_w, ground-truth) instead
    of the wheel encoder — immune to wheel slip fooling the encoder into reporting near-zero error.
    Checks both X (error against the target) and Y (drift off the X axis): with independent
    per-wheel torque the robot can yaw and drift sideways while X alone still looks fine. root_pos_w
    is privileged info, so only valid for reward at train time, never as an observation."""
    command = env.command_manager.get_command(command_name)[:, 0]
    asset: Articulation = env.scene[asset_cfg.name]
    true_pos_x = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    true_pos_y = asset.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    error_x = (command - true_pos_x) / norm_scale
    error_y = true_pos_y / norm_scale  # target Y is always 0
    return torch.square(error_x) + torch.square(error_y)


def true_position_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float = 0.1,
) -> torch.Tensor:
    """Exponential bonus (bounded at 1.0) for the TRUE position (root_pos_w, ground-truth, both X
    and Y — see true_position_error_l2) being close to the target. Complements
    true_position_error_l2 (penalty only, unbounded), the same way base_upright_reward complements
    base_upright_penalty."""
    command = env.command_manager.get_command(command_name)[:, 0]
    asset: Articulation = env.scene[asset_cfg.name]
    true_pos_x = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    true_pos_y = asset.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    error_sq = torch.square(command - true_pos_x) + torch.square(true_pos_y)
    return torch.exp(-error_sq / std**2)


def velocity_command_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    norm_scale: float = 0.2,
) -> torch.Tensor:
    """Penalize squared error (normalized by ``norm_scale``, m/s) between the target linear velocity
    (see UniformVelocityCommand) and the body's TRUE linear velocity (root_lin_vel_b, ground-truth,
    body X). No per-wheel split needed here (unlike distance_command_error_l2's per-wheel variants):
    root_lin_vel_b is already the body's true velocity, not derived from individual wheel encoders,
    so there's no "wheels spin opposite, average looks fine" exploit -- two wheels spinning against
    each other shows up as ang_vel_z (yaw), which ang_vel_z_l2 already penalizes separately."""
    command = env.command_manager.get_command(command_name)[:, 0]
    asset: Articulation = env.scene[asset_cfg.name]
    actual_vel = asset.data.root_lin_vel_b[:, 0]
    return torch.square((command - actual_vel) / norm_scale)


def velocity_command_tracking_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float = 0.1,
) -> torch.Tensor:
    """Exponential bonus (complementing velocity_command_error_l2) for the body's true linear
    velocity being close to the target."""
    command = env.command_manager.get_command(command_name)[:, 0]
    asset: Articulation = env.scene[asset_cfg.name]
    actual_vel = asset.data.root_lin_vel_b[:, 0]
    return torch.exp(-torch.square(command - actual_vel) / std**2)


def wheel_vel_diff_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the two wheels spinning at different speeds/directions (squared velocity
    difference). asset_cfg.joint_ids must point at exactly the 2 wheel joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.square(wheel_vel[:, 0] - wheel_vel[:, 1])


def wheel_pos_diff_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, max_diff: float = 3.14) -> torch.Tensor:
    """Penalize the two wheels' accumulated rotation drifting apart (squared difference, clamped to
    [-max_diff, max_diff] first) — catches real veering that wheel_vel_diff_l2 (instantaneous only)
    would miss. joint_pos is unbounded, so the clamp is required: without it, even a tiny persistent
    velocity mismatch (e.g. from randomize_com) makes the penalty grow quadratically with episode
    length, rewarding early falls (observed: fell_over jumped to 87% before this clamp was added)."""
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    diff = torch.clamp(wheel_pos[:, 0] - wheel_pos[:, 1], -max_diff, max_diff)
    return torch.square(diff)
