# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass

from .observations import wheel_distance

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class UniformDistanceCommand(CommandTerm):
    """Lệnh quãng đường tiến/lùi mục tiêu (m) cho robot 2 bánh tự cân bằng.

    Quãng đường mục tiêu được lấy mẫu đều trong ``cfg.ranges`` và giữ cố định trong suốt
    ``cfg.resampling_time_range`` (đặt bằng episode_length_s để chỉ lấy mẫu 1 lần/episode).
    Dương = tiến (+X_robot), âm = lùi. Sai số bám theo được đo qua quãng đường thực tế đọc trực
    tiếp từ góc quay khớp bánh xe (:func:`~.observations.wheel_distance`).
    """

    cfg: UniformDistanceCommandCfg

    def __init__(self, cfg: UniformDistanceCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # CommandTermCfg không tự resolve SceneEntityCfg như Observation/Reward/EventTermCfg
        # (không đi qua ManagerBase._process_term_cfg_at_play), nên phải tự resolve joint_ids ở đây.
        self.cfg.asset_cfg.resolve(env.scene)
        self.distance_command = torch.zeros(self.num_envs, 1, device=self.device)
        self.metrics["error_distance"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformDistanceCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tDistance range: {self.cfg.ranges}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Quãng đường tiến/lùi mục tiêu (m). Shape (num_envs, 1)."""
        return self.distance_command

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        traveled = wheel_distance(self._env, self.cfg.asset_cfg, self.cfg.wheel_radius)
        max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        self.metrics["error_distance"] += torch.abs(self.distance_command[:, 0] - traveled[:, 0]) / max_command_step

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.distance_command[env_ids, 0] = r.uniform_(*self.cfg.ranges)

    def _update_command(self):
        pass

    """
    Debug visualization.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        # tạo marker (trục toạ độ) lần đầu tiên nếu chưa có
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_visualizer_cfg)
            self.goal_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_visualizer"):
                self.goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # vị trí mục tiêu (world) = gốc env + quãng đường mục tiêu dọc trục X (robot spawn tại
        # x=0, y=0 so với gốc env, xem TwoWheel_CFG.init_state trong twip.py), nhấc lên khỏi mặt
        # đất 1 chút cho dễ nhìn.
        target_pos_w = self._env.scene.env_origins.clone()
        target_pos_w[:, 0] += self.distance_command[:, 0]
        target_pos_w[:, 2] += 0.1
        self.goal_visualizer.visualize(target_pos_w)


@configclass
class UniformDistanceCommandCfg(CommandTermCfg):
    """Cấu hình cho :class:`UniformDistanceCommand`."""

    class_type: type = UniformDistanceCommand

    asset_cfg: SceneEntityCfg = MISSING
    """Scene entity trỏ tới 1 hoặc nhiều khớp bánh xe cần theo dõi vị trí (nhiều khớp -> lấy trung
    bình, xem :func:`~.observations.wheel_distance`)."""

    wheel_radius: float = 0.034
    """Bán kính bánh xe (m)."""

    ranges: tuple[float, float] = MISSING
    """Khoảng lấy mẫu quãng đường mục tiêu (m), dương = tiến, âm = lùi."""

    goal_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/target_distance"
    )
    """Marker (trục toạ độ X/Y/Z) hiển thị vị trí mục tiêu khi ``debug_vis=True``."""
    goal_visualizer_cfg.markers["frame"].scale = (0.15, 0.15, 0.15)
