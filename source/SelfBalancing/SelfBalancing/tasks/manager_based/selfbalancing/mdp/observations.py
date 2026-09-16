from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_G = 9.81


def imu_lin_acc(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Accelerometer reading from simulated IMU.

    Unlike projected_gravity (which is R^T * g — a perfect, noise-free tilt measurement),
    this returns R^T * (a_body + gravity_bias), i.e. what a real accelerometer sees.
    When the robot accelerates, body dynamics corrupt the tilt estimate — exactly as on hardware.

    Output shape: (N, 3), normalized by g so values are dimensionless (~[-1, 1] when near-level).
    """
    imu = env.scene["imu"]
    return imu.data.lin_acc_b / _G


def imu_ang_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Gyroscope reading from simulated IMU, expressed in IMU (body) frame.

    Shape: (N, 3), units rad/s.
    """
    imu = env.scene["imu"]
    return imu.data.ang_vel_b


def imu_pitch_angle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Góc nghiêng thân xe (rad) quanh trục bánh xe (Y_robot -- RobotTwoWheel.urdf: joint_L/joint_R
    có axis="0 1 0", KHÁC với robot TWIP cũ dùng trục X, xem lưu ý bên dưới), tính từ
    projected_gravity_b (giá trị SẠCH, không nhiễu do gia tốc thân xe -- khác với bản trước đây dùng
    accelerometer thô).

    Dùng projected_gravity_b thay vì accelerometer thô để chuyển động không làm nhiễu góc đo, khớp
    với pitch ĐÃ QUA BỘ LỌC complementary/Kalman mà firmware thật đưa vào policy (firmware không gửi
    thẳng accelerometer thô, mà tự tính pitch sạch trước khi feed cho policy). Đứng thẳng ->
    projected_gravity_b = [0, 0, -1] -> pitch = 0.

    LƯU Ý: robot TWIP cũ (Twip_Rsl_v2) có trục bánh xe = X_robot nên dùng atan2(-pg[:,1], -pg[:,2])
    (mặt phẳng Y-Z, quay quanh X). RobotTwoWheel này có trục bánh xe = Y_robot (chuẩn "pitch" thông
    thường), nên phải đổi sang mặt phẳng X-Z (quay quanh Y): atan2(-pg[:,0], -pg[:,2]).

    Shape: (N, 1).
    """
    robot = env.scene["robot"]
    pg = robot.data.projected_gravity_b  # (N, 3), = [0, 0, -1] khi đứng thẳng
    return torch.atan2(-pg[:, 0], -pg[:, 2]).unsqueeze(1)


def imu_pitch_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Tốc độ góc nghiêng (rad/s) quanh trục bánh xe (Y_robot, xem lưu ý ở imu_pitch_angle), đọc
    trực tiếp từ root_ang_vel_b của robot thay vì qua object IMU -- đồng nhất nguồn với
    imu_pitch_angle (cùng dùng dữ liệu robot, không qua hệ toạ độ đã xoay của IMU), giúp pitch_rate
    luôn nhất quán = d(pitch_angle)/dt.

    Shape: (N, 1).
    """
    robot = env.scene["robot"]
    return robot.data.root_ang_vel_b[:, 1].unsqueeze(1)


def wheel_distance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, wheel_radius: float = 0.034) -> torch.Tensor:
    """Quãng đường đã lăn (m), tính trực tiếp từ góc quay khớp (joint_pos) của 1 hoặc nhiều bánh xe.

    asset_cfg.joint_ids trỏ tới 1 hoặc nhiều khớp bánh xe -- nếu nhiều, lấy trung bình góc quay các
    khớp (giả định 2 bánh quay cùng chiều dấu khi xe đi thẳng, giống quy ước wheel_vel_diff_l2) để
    giảm ảnh hưởng khi 1 bánh bị trượt cục bộ. joint_pos của khớp bánh xe là góc quay liên tục,
    không giới hạn/không wrap về [-π, π] (quay được vô hạn vòng, từ -∞ đến +∞), nên dùng thẳng làm
    phép đo vị trí. Bánh xe lăn không trượt: dist = angle * wheel_radius. Mốc 0 là vị trí khớp lúc
    reset episode (reset_wheels đưa về 0).

    Shape: (N, 1).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_angle = asset.data.joint_pos[:, asset_cfg.joint_ids].mean(dim=1)
    return (wheel_angle * wheel_radius).unsqueeze(-1)
