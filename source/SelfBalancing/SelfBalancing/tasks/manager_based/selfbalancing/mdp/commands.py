# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# UniformDistanceCommand (target a fixed X position, hold still) has been replaced by
# UniformVelocityCommand below (target a constant forward/backward speed) -- the two goals are
# mutually exclusive, so they can't coexist. See git history for the old position-command
# implementation.


class UniformVelocityCommand(CommandTerm):
    """Target linear velocity (m/s) along the robot's body X axis.

    Sampled uniformly from ``cfg.ranges``, held fixed for ``cfg.resampling_time_range`` (= usually
    episode_length_s, i.e. once per episode). Positive = forward, negative = backward. Single axis
    only -- unlike IsaacLab's built-in ``mdp.UniformVelocityCommand`` (lin_vel_x/y + ang_vel_z, for
    robots that can strafe/turn in place), this TWIP robot can't move laterally, and yaw is always
    treated as unwanted behavior (actively penalized via ``yaw_rate`` in RewardsCfg) rather than a
    controllable axis.

    Tracked against the body's true linear velocity (``root_lin_vel_b``, ground-truth), not a wheel
    encoder -- velocity is instantaneous, so there's no accumulated-slip concern like there was for
    the old distance command.
    """

    cfg: UniformVelocityCommandCfg

    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.vel_command_b = torch.zeros(self.num_envs, 1, device=self.device)
        self.metrics["error_vel"] = torch.zeros(self.num_envs, device=self.device)
        # Body id for base_link's true CoM (debug marker) -- found by name rather than hardcoded
        # index 0 so it doesn't depend on the URDF importer's link ordering.
        self._base_link_body_id = self.robot.find_bodies("base_link")[0][0]

    def __str__(self) -> str:
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tVelocity range: {self.cfg.ranges}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Target linear velocity along body X (m/s). Shape (num_envs, 1)."""
        return self.vel_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        actual_vel = self.robot.data.root_lin_vel_b[:, 0]
        self.metrics["error_vel"] += torch.abs(self.vel_command_b[:, 0] - actual_vel) / max_command_step

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges)

    def _update_command(self):
        pass

    """
    Debug visualization.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers (goal/actual velocity arrows + CoM frame) on first use
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
            if not hasattr(self, "current_vel_visualizer"):
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            if not hasattr(self, "com_visualizer"):
                self.com_visualizer = VisualizationMarkers(self.cfg.com_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
            self.com_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)
            if hasattr(self, "com_visualizer"):
                self.com_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.3
        # Green arrow = target velocity, blue = actual -- length scales with magnitude, flipped 180°
        # when the velocity is negative (reverse).
        goal_scale, goal_quat = self._resolve_x_velocity_to_arrow(self.vel_command_b[:, 0])
        actual_scale, actual_quat = self._resolve_x_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, 0])
        self.goal_vel_visualizer.visualize(base_pos_w, goal_quat, goal_scale)
        self.current_vel_visualizer.visualize(base_pos_w, actual_quat, actual_scale)

        # True CoM frame (body_com_pos_w/quat_w already includes the URDF <inertial> offset plus any
        # runtime randomize_com shift), to see how far/which way the CoM lands per env.
        com_pos_w = self.robot.data.body_com_pos_w[:, self._base_link_body_id, :]
        com_quat_w = self.robot.data.body_com_quat_w[:, self._base_link_body_id, :]
        self.com_visualizer.visualize(com_pos_w, com_quat_w)

    """
    Internal helpers.
    """

    def _resolve_x_velocity_to_arrow(self, x_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a 1D body-X velocity into an arrow marker scale + orientation."""
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(x_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.abs(x_velocity) * 3.0
        # The arrow mesh points +X by default -- rotate 180° about Z for negative (reverse) velocity.
        heading_angle = torch.where(x_velocity < 0, torch.full_like(x_velocity, math.pi), torch.zeros_like(x_velocity))
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # body frame -> world frame, following the robot's current heading
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        return arrow_scale, arrow_quat


@configclass
class UniformVelocityCommandCfg(CommandTermCfg):
    """Configuration for :class:`UniformVelocityCommand`."""

    class_type: type = UniformVelocityCommand

    asset_name: str = MISSING
    """Name of the robot asset in the scene, e.g. "robot"."""

    ranges: tuple[float, float] = MISSING
    """Sampling range for the target linear velocity (m/s), positive = forward, negative = backward."""

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """Green arrow showing the TARGET velocity when ``debug_vis=True``."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """Blue arrow showing the ACTUAL velocity when ``debug_vis=True``."""

    com_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/com")
    """Frame marker showing base_link's true world CoM position/orientation when ``debug_vis=True``."""
    com_visualizer_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
