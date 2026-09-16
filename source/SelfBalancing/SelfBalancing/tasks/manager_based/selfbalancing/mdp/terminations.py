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
    """Huỷ episode ngay nếu quãng đường thực tế (vị trí bánh xe, đo qua encoder) lệch khỏi quãng
    đường mục tiêu (lệnh) quá ``threshold`` (m) -- xe đi quá xa điểm đặt.

    Lưu ý: dựa trên encoder (giả định bánh xe lăn không trượt) nên có thể bị "lừa" nếu bánh xe trượt
    trên sàn -- dùng kèm :func:`base_position_error_exceeded` (dựa trên vị trí thật) làm lưới an toàn.
    """
    command = env.command_manager.get_command(command_name)[:, 0]
    traveled = wheel_distance(env, asset_cfg, wheel_radius)[:, 0]
    return torch.abs(command - traveled) > threshold


def base_position_error_exceeded(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Huỷ episode ngay nếu vị trí THẬT (root_pos_w, ground-truth mô phỏng -- không qua encoder) của
    thân xe theo trục X lệch khỏi điểm reset quá ``threshold`` (m).

    Lưới an toàn bổ sung cho :func:`distance_command_error_exceeded`: nếu bánh xe trượt trên sàn thay
    vì lăn, encoder (wheel_distance) có thể báo sai (gần 0) trong khi robot đã trôi thật -- termination
    này dùng vị trí world thật nên không bị exploit theo cách đó. Vì dùng thông tin privileged
    (không phải cảm biến thật robot có), chỉ hợp lý dùng cho termination/giám sát, không dùng làm
    observation cho policy.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pos_x = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    return torch.abs(pos_x) > threshold
