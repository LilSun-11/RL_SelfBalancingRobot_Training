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
    """
    Phạt xe khi bị nghiêng. Dựa vào projected_gravity_b (vector Z của thế giới chiếu vào thân xe).
    Khi xe đứng thẳng hoàn hảo, vector này là [0, 0, 1]. Ta sẽ phạt 2 trục x (Roll) và y (Pitch).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    proj_grav = asset.data.projected_gravity_b
    # Tính tổng bình phương của trục x và trục y
    return torch.sum(torch.square(proj_grav[:, :2]), dim=1)


def base_upright_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float = 0.2) -> torch.Tensor:
    """
    Thưởng xe đứng thẳng bằng exponential kernel: bằng 1.0 khi đứng thẳng tuyệt đối,
    giảm dần về 0 khi nghiêng. Bổ sung cho base_upright_penalty (dạng phạt không chặn trên).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    proj_grav = asset.data.projected_gravity_b
    tilt_sq = torch.sum(torch.square(proj_grav[:, :2]), dim=1)
    return torch.exp(-tilt_sq / std**2)


def ang_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Phạt xe khi quay quanh trục Z (yaw), tính theo bình phương vận tốc góc yaw."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])


def yaw_angle_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Phạt bình phương góc yaw (quay quanh trục Z) hiện tại của thân xe.

    Bổ sung cho ang_vel_z_l2 (chỉ phạt vận tốc quay tức thời): 2 bánh có thể lệch vận tốc thoáng
    qua rồi bù lại nên vận tốc yaw trung bình có thể ~0, nhưng nếu góc yaw TÍCH LUỸ đã lệch thì xe
    đang hướng sai hướng ban đầu thật -- term này bắt được sai lệch tuyệt đối đó.

    Không cần lưu trạng thái "yaw lúc spawn" riêng vì luôn = 0: EventCfg.reset_base chỉ random pose
    "roll" (nghiêng cân bằng), không random "yaw" -- nên spawn/reset lúc nào yaw cũng bắt đầu ở 0,
    có thể so sánh thẳng góc yaw hiện tại với hằng số 0.

    euler_xyz_from_quat trả về roll-pitch-yaw theo quy ước XYZ extrinsic, giá trị yaw đã tự động nằm
    trong (-π, π] nên không cần wrap_to_pi thêm.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    _, _, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.square(yaw)


def lin_vel_x_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Phạt xe khi di chuyển theo trục X (body frame), tính theo bình phương vận tốc dài."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 0])


def lin_vel_x_normalized_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, norm_scale: float = 0.05
) -> torch.Tensor:
    """Giống lin_vel_x_l2 nhưng chuẩn hoá vận tốc theo ``norm_scale`` (m/s) trước khi bình phương.

    root_lin_vel_b tính bằng m/s luôn rất nhỏ (vài mm/s -> vài cm/s) so với các reward khác như
    upright (dựa trên proj_gravity, range tự nhiên ~[0,1]) -- bình phương trực tiếp khiến vận tốc
    trôi chậm gần như không bị phạt dù nhân weight lớn. Chia cho norm_scale trước khi bình phương đưa
    reward về cùng thang O(1) (norm_scale = "vận tốc coi là đáng phạt ngang upright"), giúp weight dễ
    cân chỉnh/so sánh giữa các reward term.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 0] / norm_scale)


def distance_command_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    wheel_radius: float = 0.034,
    norm_scale: float = 0.05,
) -> torch.Tensor:
    """Phạt bình phương sai lệch (đã chuẩn hoá theo ``norm_scale``, đơn vị m) giữa quãng đường mục
    tiêu (lệnh) và quãng đường thực tế (vị trí bánh xe).

    Lý do chuẩn hoá: xem :func:`lin_vel_x_normalized_l2` -- sai số vị trí tính bằng mét cũng nhỏ y hệt
    vấn đề đó. norm_scale mặc định khớp ngưỡng huỷ episode "out_of_range" (0.05 m) nên sai số = ngưỡng
    huỷ tương ứng với giá trị chuẩn hoá = 1.0.
    """
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
    """Thưởng exponential (bổ sung cho distance_command_error_l2) khi quãng đường thực tế (vị trí bánh xe)
    gần đúng quãng đường mục tiêu (lệnh)."""
    command = env.command_manager.get_command(command_name)[:, 0]
    traveled = wheel_distance(env, asset_cfg, wheel_radius)[:, 0]
    return torch.exp(-torch.square(command - traveled) / std**2)


def true_position_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    norm_scale: float = 0.05,
) -> torch.Tensor:
    """Phạt bình phương sai lệch (đã chuẩn hoá theo ``norm_scale``, đơn vị m) giữa quãng đường mục
    tiêu (lệnh) và vị trí THẬT (root_pos_w, ground-truth mô phỏng) của thân xe theo trục X.

    Bổ sung cho distance_command_error_l2 (dựa trên encoder wheel_distance, giả định bánh xe lăn
    không trượt): nếu bánh xe trượt trên sàn, encoder có thể báo sai số ~0 trong khi robot đã trôi
    thật -- term này dùng root_pos_w nên không bị "lừa" theo cách đó. Vì root_pos_w là thông tin
    privileged (không phải cảm biến thật robot có), chỉ hợp lý dùng cho reward lúc train, không đưa
    vào observation của policy (giống lý do của base_position_error_exceeded trong terminations.py).
    """
    command = env.command_manager.get_command(command_name)[:, 0]
    asset: Articulation = env.scene[asset_cfg.name]
    true_pos = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    return torch.square((command - true_pos) / norm_scale)


def wheel_vel_diff_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Phạt khi 2 bánh xe quay khác chiều/khác tốc độ nhau (bình phương hiệu vận tốc góc 2 bánh).

    asset_cfg.joint_ids phải trỏ đúng 2 khớp bánh xe. Quay ngược chiều nhau -> hiệu số lớn -> phạt nặng.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.square(wheel_vel[:, 0] - wheel_vel[:, 1])


def wheel_pos_diff_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, max_diff: float = 3.14) -> torch.Tensor:
    """Phạt khi góc quay (tích luỹ từ lúc reset) của 2 bánh xe lệch nhau (bình phương hiệu góc, đã
    chặn trong [-max_diff, max_diff] trước khi bình phương).

    asset_cfg.joint_ids phải trỏ đúng 2 khớp bánh xe. Bổ sung cho wheel_vel_diff_l2 (chỉ phạt lệch
    vận tốc tức thời): 2 bánh có thể lệch vận tốc thoáng qua rồi bù lại nhau nên vận tốc trung bình
    có thể giống nhau, nhưng nếu góc quay TÍCH LUỸ đã lệch (1 bánh quay nhiều/ít hơn bánh kia theo
    thời gian) thì xe đang rẽ/lệch hướng thật -- term này bắt được sai lệch tích luỹ đó.

    joint_pos là góc quay tích luỹ KHÔNG GIỚI HẠN (quay vô hạn vòng) -- nếu không chặn, chỉ cần 1 lệch
    vận tốc hệ thống rất nhỏ giữa 2 bánh (vd do randomize_com) cũng khiến hiệu góc tăng dần theo thời
    gian sống, làm phạt tăng BÌNH PHƯƠNG theo thời gian sống -- episode sống lâu luôn bị phạt nặng hơn
    bất kể hành vi tốt xấu, tạo động lực ngã sớm để tránh cộng dồn phạt (đã xảy ra thật, xem log
    training fell_over nhảy lên 87% sau khi thêm term này lần đầu, chưa có clamp). Sau khi clamp, phạt
    mỗi step có trần, tổng phạt cả episode chỉ tăng tuyến tính theo thời gian -- an toàn như các reward
    per-step khác (upright, pitch_rate, ...), không còn bùng nổ theo thời gian sống nữa.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    diff = torch.clamp(wheel_pos[:, 0] - wheel_pos[:, 1], -max_diff, max_diff)
    return torch.square(diff)
