# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class SymmetricWheelEffortAction(ActionTerm):
    """1 action (action_dim == 1) áp CÙNG 1 giá trị torque cho mọi khớp bánh xe được truyền vào.

    Khác với JointEffortActionCfg thông thường (mỗi khớp bánh xe nhận 1 action riêng, 2 action độc
    lập cho robot 2 bánh) -- action term này chỉ có 1 action duy nhất, đảm bảo 2 bánh LUÔN đồng bộ
    tuyệt đối (cùng tiến/cùng lùi với cùng độ lớn torque), không cần dựa vào reward shaping (vốn chỉ
    là ràng buộc mềm, không đảm bảo policy tuân theo). Đánh đổi: robot mất khả năng chủ động sửa lệch
    yaw bằng cách chỉnh lệch torque 2 bánh (cơ chế duy nhất robot differential-drive có để chống xoay).

    Mỗi step: raw (N, 1) trong [-1, 1] -> nhân scale (Nm) -> clamp [-scale, scale] -> broadcast ra
    mọi khớp trong joint_names.
    """

    cfg: SymmetricWheelEffortActionCfg
    _asset: Articulation

    def __init__(self, cfg: SymmetricWheelEffortActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        # scale sang torque (Nm), clamp về giới hạn động cơ thật (giữ raw action ~[-1, 1])
        torque = torch.clamp(self._raw_actions * self.cfg.scale, -self.cfg.scale, self.cfg.scale)
        # cùng 1 torque cho mọi bánh -> chỉ có thể tiến/lùi cân bằng, không thể tự rẽ/lệch hướng
        self._processed_actions = torque.repeat(1, self._num_joints)

    def apply_actions(self) -> None:
        self._asset.set_joint_effort_target(self._processed_actions, joint_ids=self._joint_ids)


@configclass
class SymmetricWheelEffortActionCfg(ActionTermCfg):
    """Cấu hình cho :class:`SymmetricWheelEffortAction`."""

    class_type: type[ActionTerm] = SymmetricWheelEffortAction

    joint_names: list[str] = ["joint_L", "joint_R"]
    """Danh sách khớp bánh xe cùng nhận 1 lệnh torque giống hệt nhau. Mặc định 2 khớp bánh xe của
    RobotTwoWheel.urdf (joint_L, joint_R)."""

    scale: float = 1.0
    """Raw action trong [-1, 1] nhân với số này ra torque (Nm), rồi clamp về [-scale, scale]."""

