# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .observations import wheel_distance

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def distance_command_error_exceeded(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    threshold: float,
    wheel_radius: float = 0.034,
) -> torch.Tensor:
    """End the episode if the encoder-measured distance strays more than ``threshold`` (m) from the
    target. Encoder-based (assumes no-slip), so pair with :func:`base_position_error_exceeded` as a
    ground-truth safety net."""
    command = env.command_manager.get_command(command_name)[:, 0]
    traveled = wheel_distance(env, asset_cfg, wheel_radius)[:, 0]
    return torch.abs(command - traveled) > threshold


def base_position_error_exceeded(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """End the episode if the body's TRUE position (root_pos_w, ground-truth) strays more than
    ``threshold`` (m) from the reset point, on EITHER the X or Y axis.

    Checking Y matters here because this robot's two wheels get independent torque (see ActionsCfg),
    so it can actively yaw/turn and drift sideways while X alone stays within range — unlike a
    symmetric-action TWIP, which physically can't leave the X axis. Privileged info — valid for
    termination, never as an observation.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pos_x = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    pos_y = asset.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    return (torch.abs(pos_x) > threshold) | (torch.abs(pos_y) > threshold)
