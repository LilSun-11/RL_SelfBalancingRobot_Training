# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Solid 3D arrow (m): a cylinder shaft whose length follows the velocity, capped by a fixed-size cone.
ARROW_SHAFT_RADIUS = 0.008
ARROW_HEAD_RADIUS = 0.02
ARROW_HEAD_LENGTH = 0.04


def arrow_3d_marker_cfg(prim_path: str, color: tuple[float, float, float]) -> VisualizationMarkersCfg:
    """Two prototypes along +X: "shaft" (cylinder, unit length, scaled in X to the arrow length) and
    "head" (cone, apex at +X)."""
    material = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "shaft": sim_utils.CylinderCfg(radius=ARROW_SHAFT_RADIUS, height=1.0, axis="X", visual_material=material),
            "head": sim_utils.ConeCfg(
                radius=ARROW_HEAD_RADIUS, height=ARROW_HEAD_LENGTH, axis="X", visual_material=material
            ),
        },
    )

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
        # envs told to stand still (vx = 0, and wz = 0 via UniformYawRateCommandCfg.standing_command_name)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
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
        # Running mean of |error| over the episode so far (m/s). Dividing a whole-episode sum by the
        # steps of ONE resampling interval (as IsaacLab's built-in velocity command does) inflates it by
        # episode_length / resampling_time (e.g. ~40x for 200 s episodes with 5 s resampling).
        error = torch.abs(self.vel_command_b[:, 0] - self.robot.data.root_lin_vel_b[:, 0])
        n = self._env.episode_length_buf.clamp(min=1)
        self.metrics["error_vel"] += (error - self.metrics["error_vel"]) / n

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges)
        # a fraction of the resampled envs gets an exact zero command for this interval
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) < self.cfg.rel_standing_envs

    def _update_command(self):
        self.vel_command_b[self.is_standing_env, 0] = 0.0

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
        # Green arrow = target (vx, wz), blue = actual -- it leans toward the turn direction (see
        # _draw_velocity_arrow) and follows the actual path when reversing.
        if self.cfg.yaw_rate_command_name is not None:
            target_wz = self._env.command_manager.get_command(self.cfg.yaw_rate_command_name)[:, 0]
        else:
            target_wz = torch.zeros_like(self.vel_command_b[:, 0])
        self._draw_velocity_arrow(self.goal_vel_visualizer, base_pos_w, self.vel_command_b[:, 0], target_wz)
        self._draw_velocity_arrow(
            self.current_vel_visualizer,
            base_pos_w,
            self.robot.data.root_lin_vel_b[:, 0],
            self.robot.data.root_ang_vel_b[:, 2],
        )

        # True CoM frame (body_com_pos_w/quat_w already includes the URDF <inertial> offset plus any
        # runtime randomize_com shift), to see how far/which way the CoM lands per env.
        com_pos_w = self.robot.data.body_com_pos_w[:, self._base_link_body_id, :]
        com_quat_w = self.robot.data.body_com_quat_w[:, self._base_link_body_id, :]
        self.com_visualizer.visualize(com_pos_w, com_quat_w)

    """
    Internal helpers.
    """

    def _draw_velocity_arrow(
        self, markers: VisualizationMarkers, origin_w: torch.Tensor, x_velocity: torch.Tensor, yaw_rate: torch.Tensor
    ):
        """Draw a solid 3D arrow (see arrow_3d_marker_cfg) from origin_w showing the velocity of a point
        cfg.turn_lookahead away from the robot along its direction of travel (ahead when driving
        forward, behind when reversing): (vx, wz * lookahead * sign(vx)) in the body XY plane. So it
        follows the path the robot will actually take: straight ahead/back when driving straight,
        leaning toward the side the path curves to (a left yaw while reversing curves the path to the
        right), and sideways when spinning in place. Shaft length = |that velocity| *
        cfg.arrow_length_per_mps."""
        n = x_velocity.shape[0]
        # small deadband so the arrow doesn't flip sides from noise when the actual vx hovers around 0
        travel_dir = torch.where(x_velocity < -0.01, -1.0, 1.0)
        lateral = yaw_rate * self.cfg.turn_lookahead * travel_dir
        length = (torch.sqrt(x_velocity**2 + lateral**2) * self.cfg.arrow_length_per_mps).clamp(
            max=self.cfg.arrow_max_length
        )
        # heading of that planar velocity in the body frame, then body -> world frame
        heading = torch.atan2(lateral, x_velocity)
        zeros = torch.zeros_like(heading)
        quat = math_utils.quat_mul(self.robot.data.root_quat_w, math_utils.quat_from_euler_xyz(zeros, zeros, heading))
        unit_x = torch.zeros(n, 3, device=self.device)
        unit_x[:, 0] = 1.0
        direction = math_utils.quat_apply(quat, unit_x)

        # shaft centered halfway along the arrow; head just past the shaft's tip
        shaft_pos = origin_w + direction * (0.5 * length).unsqueeze(1)
        head_pos = origin_w + direction * (length + 0.5 * ARROW_HEAD_LENGTH).unsqueeze(1)
        shaft_scale = torch.ones(n, 3, device=self.device)
        shaft_scale[:, 0] = length
        # hide the head when there's essentially no velocity (the shaft already has zero length)
        head_scale = (length > 1e-3).float().unsqueeze(1).repeat(1, 3)

        markers.visualize(
            translations=torch.cat([shaft_pos, head_pos]),
            orientations=torch.cat([quat, quat]),
            scales=torch.cat([shaft_scale, head_scale]),
            marker_indices=torch.cat(
                [torch.zeros(n, dtype=torch.int32, device=self.device), torch.ones(n, dtype=torch.int32, device=self.device)]
            ),
        )


@configclass
class UniformVelocityCommandCfg(CommandTermCfg):
    """Configuration for :class:`UniformVelocityCommand`."""

    class_type: type = UniformVelocityCommand

    asset_name: str = MISSING
    """Name of the robot asset in the scene, e.g. "robot"."""

    ranges: tuple[float, float] = MISSING
    """Sampling range for the target linear velocity (m/s), positive = forward, negative = backward."""

    rel_standing_envs: float = 0.0
    """Probability (0-1) that an env gets vx = 0 at each resample, held until its next resample, so the
    policy also practices standing still. A yaw-rate command with ``standing_command_name`` pointing
    here is zeroed for the same envs."""

    arrow_length_per_mps: float = 1.0
    """Arrow shaft length (m) per 1 m/s of velocity."""

    arrow_max_length: float = 0.5
    """Cap on the shaft length (m), so a falling or pushed robot doesn't draw a huge arrow."""

    yaw_rate_command_name: str | None = None
    """Name of the yaw-rate command term (e.g. "target_yaw_rate"). When set, the target arrow also
    shows the commanded turn; otherwise it shows forward velocity only."""

    turn_lookahead: float = 0.5
    """Distance (m) along the direction of travel whose velocity the arrows show -- the larger, the more a given
    yaw rate tilts the arrow sideways."""

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = arrow_3d_marker_cfg(
        "/Visuals/Command/velocity_goal", color=(0.0, 1.0, 0.0)
    )
    """Solid green 3D arrow showing the TARGET velocity when ``debug_vis=True``."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = arrow_3d_marker_cfg(
        "/Visuals/Command/velocity_current", color=(0.0, 0.0, 1.0)
    )
    """Solid blue 3D arrow showing the ACTUAL velocity when ``debug_vis=True``."""

    com_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/com")
    """Frame marker showing base_link's true world CoM position/orientation when ``debug_vis=True``."""
    com_visualizer_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)


class UniformYawRateCommand(CommandTerm):
    """Target yaw rate (rad/s) about the robot's body Z axis -- positive = turn left (counter-clockwise
    seen from above), negative = turn right.

    Sampled uniformly from ``cfg.ranges`` every ``cfg.resampling_time_range`` seconds. Kept as a
    separate term from :class:`UniformVelocityCommand` so the forward-velocity command and its rewards
    stay unchanged; together they form the (vx, wz) command of a differential-drive robot. Tracked
    against the body's true yaw rate (``root_ang_vel_b[:, 2]``).
    """

    cfg: UniformYawRateCommandCfg

    def __init__(self, cfg: UniformYawRateCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.yaw_rate_command_b = torch.zeros(self.num_envs, 1, device=self.device)
        self.metrics["error_yaw_rate"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformYawRateCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tYaw rate range: {self.cfg.ranges}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """Target yaw rate about body Z (rad/s). Shape (num_envs, 1)."""
        return self.yaw_rate_command_b

    def _update_metrics(self):
        # running mean of |error| over the episode so far (rad/s), see UniformVelocityCommand
        error = torch.abs(self.yaw_rate_command_b[:, 0] - self.robot.data.root_ang_vel_b[:, 2])
        n = self._env.episode_length_buf.clamp(min=1)
        self.metrics["error_yaw_rate"] += (error - self.metrics["error_yaw_rate"]) / n

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.yaw_rate_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges)

    def _update_command(self):
        # standing envs of the velocity command also get wz = 0. Done here (every step) rather than
        # at resample, since the two commands resample on independent timers.
        if self.cfg.standing_command_name is not None:
            standing = self._env.command_manager.get_term(self.cfg.standing_command_name).is_standing_env
            self.yaw_rate_command_b[standing, 0] = 0.0


@configclass
class UniformYawRateCommandCfg(CommandTermCfg):
    """Configuration for :class:`UniformYawRateCommand`."""

    class_type: type = UniformYawRateCommand

    asset_name: str = MISSING
    """Name of the robot asset in the scene, e.g. "robot"."""

    ranges: tuple[float, float] = MISSING
    """Sampling range for the target yaw rate (rad/s), positive = turn left, negative = turn right."""

    standing_command_name: str | None = None
    """Name of a :class:`UniformVelocityCommand` term (e.g. "target_velocity"). When set, the yaw-rate
    command is forced to 0 for that term's standing envs (see its ``rel_standing_envs``)."""


class UniformYawCommand(CommandTerm):
    """Target yaw angle (rad) relative to the heading the robot spawned with -- positive = rotated left
    (counter-clockwise seen from above), wrapped to [-pi, pi].

    Each env's spawn heading is recorded at reset and counts as yaw = 0, so the target and the yaw
    observation (:attr:`relative_yaw`, see mdp.relative_yaw) are both relative to it -- like a real
    robot that integrates its gyro Z from 0 at power-up. Sampled uniformly from ``cfg.ranges`` every
    ``cfg.resampling_time_range`` seconds. Replaces :class:`UniformYawRateCommand` (kept for reference).
    """

    cfg: UniformYawCommandCfg

    def __init__(self, cfg: UniformYawCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.yaw_command = torch.zeros(self.num_envs, 1, device=self.device)
        # world heading at spawn (= yaw 0). Zeros until the first reset, which is fine: the observation
        # manager only evaluates terms before that to infer their shapes.
        self.spawn_heading_w = torch.zeros(self.num_envs, device=self.device)
        # for standing_command_name: which envs were standing last step, to catch the moment one starts
        self._was_standing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.metrics["error_yaw"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformYawCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tYaw range: {self.cfg.ranges}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """Target yaw relative to the spawn heading (rad). Shape (num_envs, 1)."""
        return self.yaw_command

    @property
    def relative_yaw(self) -> torch.Tensor:
        """Current yaw relative to the spawn heading (rad, wrapped to [-pi, pi]). Shape (num_envs,)."""
        return math_utils.wrap_to_pi(self.robot.data.heading_w - self.spawn_heading_w)

    @property
    def yaw_error(self) -> torch.Tensor:
        """Shortest signed angle from the current yaw to the target (rad, [-pi, pi]). Shape (num_envs,)."""
        return math_utils.wrap_to_pi(self.yaw_command[:, 0] - self.relative_yaw)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        # The command manager resets after the "reset" events, so heading_w is already the new spawn pose.
        self.spawn_heading_w[env_ids] = self.robot.data.heading_w[env_ids]
        # the previous target of a new episode is the spawn heading, so with max_step the first target
        # is within +-max_step of it
        self.yaw_command[env_ids, 0] = 0.0
        extras = super().reset(env_ids)
        if self.cfg.standing_command_name is not None:
            # a standing env holds its spawn heading
            standing = self._env.command_manager.get_term(self.cfg.standing_command_name).is_standing_env
            self.yaw_command[env_ids, 0] = torch.where(standing[env_ids], 0.0, self.yaw_command[env_ids, 0])
            self._was_standing[env_ids] = standing[env_ids]
        return extras

    def _update_metrics(self):
        # running mean of |error| over the episode so far (rad), see UniformVelocityCommand
        error = torch.abs(self.yaw_error)
        n = self._env.episode_length_buf.clamp(min=1)
        self.metrics["error_yaw"] += (error - self.metrics["error_yaw"]) / n

    def _resample_command(self, env_ids: Sequence[int]):
        old = self.yaw_command[env_ids, 0].clone()
        r = torch.empty(len(env_ids), device=self.device)
        if self.cfg.max_step is None:
            self.yaw_command[env_ids, 0] = r.uniform_(*self.cfg.ranges)
        else:
            # step from the previous target by at most max_step (turns of <= max_step at a time), wrapped
            # to [-pi, pi] (so +170 deg + 20 deg = -170 deg) and kept inside cfg.ranges
            new = math_utils.wrap_to_pi(old + r.uniform_(-self.cfg.max_step, self.cfg.max_step))
            self.yaw_command[env_ids, 0] = new.clamp(*self.cfg.ranges)
        if self.cfg.standing_command_name is not None:
            # standing envs keep the heading they're holding instead of being told to turn
            standing = self._env.command_manager.get_term(self.cfg.standing_command_name).is_standing_env
            self.yaw_command[env_ids, 0] = torch.where(standing[env_ids], old, self.yaw_command[env_ids, 0])

    def _update_command(self):
        # When an env starts standing (target_velocity resampled it to vx = 0), freeze the target at
        # the yaw it has right now: "stand still" = no driving AND no turning.
        if self.cfg.standing_command_name is not None:
            standing = self._env.command_manager.get_term(self.cfg.standing_command_name).is_standing_env
            started = standing & ~self._was_standing
            # (clamped to cfg.ranges, so a (0, 0) range from a curriculum really keeps the target at 0)
            self.yaw_command[started, 0] = self.relative_yaw[started].clamp(*self.cfg.ranges)
            self._was_standing[:] = standing

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_yaw_visualizer"):
                self.goal_yaw_visualizer = VisualizationMarkers(self.cfg.goal_yaw_visualizer_cfg)
            self.goal_yaw_visualizer.set_visibility(True)
        elif hasattr(self, "goal_yaw_visualizer"):
            self.goal_yaw_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        # Fixed-length, flat arrow pointing along the target heading (world yaw = spawn + target),
        # drawn a bit lower than the velocity arrows so they don't overlap.
        n = self.num_envs
        length = self.cfg.arrow_length
        heading = self.spawn_heading_w + self.yaw_command[:, 0]
        zeros = torch.zeros_like(heading)
        quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading)
        direction = torch.stack([torch.cos(heading), torch.sin(heading), zeros], dim=1)
        origin = self.robot.data.root_pos_w.clone()
        origin[:, 2] += 0.2
        shaft_pos = origin + direction * (0.5 * length)
        head_pos = origin + direction * (length + 0.5 * ARROW_HEAD_LENGTH)
        shaft_scale = torch.ones(n, 3, device=self.device)
        shaft_scale[:, 0] = length
        self.goal_yaw_visualizer.visualize(
            translations=torch.cat([shaft_pos, head_pos]),
            orientations=torch.cat([quat, quat]),
            scales=torch.cat([shaft_scale, torch.ones(n, 3, device=self.device)]),
            marker_indices=torch.cat(
                [torch.zeros(n, dtype=torch.int32, device=self.device), torch.ones(n, dtype=torch.int32, device=self.device)]
            ),
        )


@configclass
class UniformYawCommandCfg(CommandTermCfg):
    """Configuration for :class:`UniformYawCommand`."""

    class_type: type = UniformYawCommand

    asset_name: str = MISSING
    """Name of the robot asset in the scene, e.g. "robot"."""

    ranges: tuple[float, float] = MISSING
    """Sampling range for the target yaw relative to the spawn heading (rad), within [-pi, pi]."""

    max_step: float | None = None
    """Max change (rad) of the target at each resample, relative to the previous target (the spawn
    heading at episode start): new = wrap(previous + U(-max_step, max_step)), clamped to ``ranges``.
    None = sample uniformly from ``ranges`` independently of the previous target."""

    standing_command_name: str | None = None
    """Name of a :class:`UniformVelocityCommand` term (e.g. "target_velocity"). When set, its standing
    envs hold the yaw they had when they started standing instead of following new targets."""

    arrow_length: float = 0.25
    """Shaft length (m) of the target-heading arrow."""

    goal_yaw_visualizer_cfg: VisualizationMarkersCfg = arrow_3d_marker_cfg(
        "/Visuals/Command/yaw_goal", color=(1.0, 0.5, 0.0)
    )
    """Solid orange 3D arrow showing the TARGET heading when ``debug_vis=True``."""
