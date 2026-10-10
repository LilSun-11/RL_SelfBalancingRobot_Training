"""Command terms: a forward velocity and a heading for the robot to follow."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Debug arrows (m): a cylinder shaft scaled to the arrow length, plus a fixed-size cone head.
ARROW_SHAFT_RADIUS = 0.008
ARROW_HEAD_RADIUS = 0.02
ARROW_HEAD_LENGTH = 0.04


def arrow_marker_cfg(prim_path: str, color: tuple[float, float, float]) -> VisualizationMarkersCfg:
    """Solid 3D arrow along +X: marker 0 = shaft (unit length), marker 1 = head."""
    material = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "shaft": sim_utils.CylinderCfg(radius=ARROW_SHAFT_RADIUS, height=1.0, axis="X", visual_material=material),
            "head": sim_utils.ConeCfg(radius=ARROW_HEAD_RADIUS, height=ARROW_HEAD_LENGTH, axis="X", visual_material=material),
        },
    )


def draw_arrows(markers: VisualizationMarkers, origin: torch.Tensor, quat: torch.Tensor, length: torch.Tensor):
    """Draw one arrow per env starting at ``origin``, pointing along the X axis of ``quat``."""
    n = origin.shape[0]
    unit_x = torch.zeros(n, 3, device=origin.device)
    unit_x[:, 0] = 1.0
    direction = math_utils.quat_apply(quat, unit_x)
    shaft_pos = origin + direction * (0.5 * length).unsqueeze(1)
    head_pos = origin + direction * (length + 0.5 * ARROW_HEAD_LENGTH).unsqueeze(1)
    shaft_scale = torch.ones(n, 3, device=origin.device)
    shaft_scale[:, 0] = length
    head_scale = (length > 1e-3).float().unsqueeze(1).repeat(1, 3)  # hide the head of a zero-length arrow
    markers.visualize(
        translations=torch.cat([shaft_pos, head_pos]),
        orientations=torch.cat([quat, quat]),
        scales=torch.cat([shaft_scale, head_scale]),
        marker_indices=torch.cat(
            [torch.zeros(n, dtype=torch.int32, device=origin.device), torch.ones(n, dtype=torch.int32, device=origin.device)]
        ),
    )


class UniformVelocityCommand(CommandTerm):
    """Target forward velocity (m/s) along the robot's body X axis, + = forward.

    Sampled uniformly from ``cfg.ranges`` every ``cfg.resampling_time_range`` seconds. With probability
    ``cfg.rel_standing_envs`` an env gets exactly 0 instead ("stand still").
    """

    cfg: UniformVelocityCommandCfg

    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.vel_command_b = torch.zeros(self.num_envs, 1, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.metrics["error_vel"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return f"UniformVelocityCommand: range {self.cfg.ranges} m/s, resampled every {self.cfg.resampling_time_range} s"

    @property
    def command(self) -> torch.Tensor:
        """Target velocity (m/s). Shape (num_envs, 1)."""
        return self.vel_command_b

    def _update_metrics(self):
        # running mean of |target - actual| over the episode so far (m/s)
        error = torch.abs(self.vel_command_b[:, 0] - self.robot.data.root_lin_vel_b[:, 0])
        n = self._env.episode_length_buf.clamp(min=1)
        self.metrics["error_vel"] += (error - self.metrics["error_vel"]) / n

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges)
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) < self.cfg.rel_standing_envs

    def _update_command(self):
        self.vel_command_b[self.is_standing_env, 0] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_visualizer_cfg)
                self.current_visualizer = VisualizationMarkers(self.cfg.current_visualizer_cfg)
            self.goal_visualizer.set_visibility(True)
            self.current_visualizer.set_visibility(True)
        elif hasattr(self, "goal_visualizer"):
            self.goal_visualizer.set_visibility(False)
            self.current_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        origin = self.robot.data.root_pos_w.clone()
        origin[:, 2] += 0.3
        # green = target velocity, blue = actual velocity; both drawn along the robot's heading
        # (flipped by 180 deg when the velocity is negative)
        yaw_only = math_utils.yaw_quat(self.robot.data.root_quat_w)
        flip = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 4)  # 180 deg about Z (w, x, y, z)
        for markers, vel in (
            (self.goal_visualizer, self.vel_command_b[:, 0]),
            (self.current_visualizer, self.robot.data.root_lin_vel_b[:, 0]),
        ):
            quat = torch.where((vel < 0).unsqueeze(1), math_utils.quat_mul(yaw_only, flip), yaw_only)
            length = (vel.abs() * self.cfg.arrow_length_per_mps).clamp(max=self.cfg.arrow_max_length)
            draw_arrows(markers, origin, quat, length)


@configclass
class UniformVelocityCommandCfg(CommandTermCfg):
    """Configuration for :class:`UniformVelocityCommand`."""

    class_type: type = UniformVelocityCommand

    asset_name: str = MISSING
    """Name of the robot in the scene."""

    ranges: tuple[float, float] = MISSING
    """Sampling range of the target velocity (m/s)."""

    rel_standing_envs: float = 0.0
    """Probability (0-1) that a resampled env gets a zero command (stand still)."""

    arrow_length_per_mps: float = 1.0
    """Arrow length (m) per 1 m/s."""

    arrow_max_length: float = 0.5
    """Maximum arrow length (m)."""

    goal_visualizer_cfg: VisualizationMarkersCfg = arrow_marker_cfg("/Visuals/Command/velocity_goal", (0.0, 1.0, 0.0))
    current_visualizer_cfg: VisualizationMarkersCfg = arrow_marker_cfg("/Visuals/Command/velocity_current", (0.0, 0.0, 1.0))


class UniformYawCommand(CommandTerm):
    """Target heading (rad) relative to the heading the robot spawned with, + = left, in [-pi, pi].

    The spawn heading of each env is stored at reset and counts as yaw 0, like a real robot that
    integrates its gyro from 0 at start-up. Every ``cfg.resampling_time_range`` seconds the target moves
    by at most ``cfg.max_step`` from the previous one. Envs that the velocity command
    ``cfg.standing_command_name`` tells to stand still keep the heading they have.
    """

    cfg: UniformYawCommandCfg

    def __init__(self, cfg: UniformYawCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.yaw_command = torch.zeros(self.num_envs, 1, device=self.device)
        self.spawn_heading_w = torch.zeros(self.num_envs, device=self.device)
        self._was_standing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.metrics["error_yaw"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return f"UniformYawCommand: range {self.cfg.ranges} rad, max step {self.cfg.max_step} rad"

    @property
    def command(self) -> torch.Tensor:
        """Target yaw relative to the spawn heading (rad). Shape (num_envs, 1)."""
        return self.yaw_command

    @property
    def relative_yaw(self) -> torch.Tensor:
        """Current yaw relative to the spawn heading (rad, [-pi, pi]). Shape (num_envs,)."""
        return math_utils.wrap_to_pi(self.robot.data.heading_w - self.spawn_heading_w)

    @property
    def yaw_error(self) -> torch.Tensor:
        """Shortest signed angle from the current yaw to the target (rad). Shape (num_envs,)."""
        return math_utils.wrap_to_pi(self.yaw_command[:, 0] - self.relative_yaw)

    def _standing(self) -> torch.Tensor:
        return self._env.command_manager.get_term(self.cfg.standing_command_name).is_standing_env

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        # commands are reset after the reset events, so heading_w is already the new spawn heading
        self.spawn_heading_w[env_ids] = self.robot.data.heading_w[env_ids]
        self.yaw_command[env_ids, 0] = 0.0  # the "previous target" of a new episode is the spawn heading
        extras = super().reset(env_ids)
        standing = self._standing()
        self.yaw_command[env_ids, 0] = torch.where(standing[env_ids], 0.0, self.yaw_command[env_ids, 0])
        self._was_standing[env_ids] = standing[env_ids]
        return extras

    def _update_metrics(self):
        n = self._env.episode_length_buf.clamp(min=1)
        self.metrics["error_yaw"] += (torch.abs(self.yaw_error) - self.metrics["error_yaw"]) / n

    def _resample_command(self, env_ids: Sequence[int]):
        old = self.yaw_command[env_ids, 0].clone()
        step = torch.empty(len(env_ids), device=self.device).uniform_(-self.cfg.max_step, self.cfg.max_step)
        new = math_utils.wrap_to_pi(old + step).clamp(*self.cfg.ranges)
        # standing envs keep their current target
        self.yaw_command[env_ids, 0] = torch.where(self._standing()[env_ids], old, new)

    def _update_command(self):
        # an env that just started standing holds the heading it has right now
        standing = self._standing()
        started = standing & ~self._was_standing
        self.yaw_command[started, 0] = self.relative_yaw[started].clamp(*self.cfg.ranges)
        self._was_standing[:] = standing

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_visualizer_cfg)
            self.goal_visualizer.set_visibility(True)
        elif hasattr(self, "goal_visualizer"):
            self.goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        # orange = target heading (world yaw = spawn heading + target)
        heading = self.spawn_heading_w + self.yaw_command[:, 0]
        zeros = torch.zeros_like(heading)
        quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading)
        origin = self.robot.data.root_pos_w.clone()
        origin[:, 2] += 0.2
        draw_arrows(self.goal_visualizer, origin, quat, torch.full_like(heading, self.cfg.arrow_length))


@configclass
class UniformYawCommandCfg(CommandTermCfg):
    """Configuration for :class:`UniformYawCommand`."""

    class_type: type = UniformYawCommand

    asset_name: str = MISSING
    """Name of the robot in the scene."""

    ranges: tuple[float, float] = MISSING
    """Allowed range of the target yaw (rad), within [-pi, pi]."""

    max_step: float = MISSING
    """Maximum change of the target at each resample (rad)."""

    standing_command_name: str = MISSING
    """Name of the UniformVelocityCommand term whose standing envs hold their heading."""

    arrow_length: float = 0.25
    """Length (m) of the target-heading arrow."""

    goal_visualizer_cfg: VisualizationMarkersCfg = arrow_marker_cfg("/Visuals/Command/yaw_goal", (1.0, 0.5, 0.0))
