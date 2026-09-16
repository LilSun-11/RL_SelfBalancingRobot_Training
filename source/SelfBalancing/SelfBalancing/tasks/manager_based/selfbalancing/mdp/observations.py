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
    """Body tilt angle (rad) about the wheel axis (Y_robot -- RobotTwoWheel.urdf's joint_L/joint_R
    use axis="0 1 0", unlike the old TWIP robot which used X), from projected_gravity_b -- a CLEAN
    signal, unlike a raw accelerometer.

    projected_gravity_b isn't corrupted by body motion, matching the already-filtered
    complementary/Kalman pitch a real firmware would feed the policy (not raw accelerometer).
    Upright -> projected_gravity_b = [0, 0, -1] -> pitch = 0.

    Note: the old TWIP robot (Twip_Rsl_v2, wheel axis = X_robot) reads atan2(-pg[:,1], -pg[:,2])
    (Y-Z plane, rotation about X). This robot's wheel axis is Y_robot (standard "pitch"), so it
    reads the X-Z plane instead (rotation about Y): atan2(-pg[:,0], -pg[:,2]).

    Shape: (N, 1).
    """
    robot = env.scene["robot"]
    pg = robot.data.projected_gravity_b  # (N, 3), = [0, 0, -1] when upright
    return torch.atan2(-pg[:, 0], -pg[:, 2]).unsqueeze(1)


def imu_pitch_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Tilt rate (rad/s) about the wheel axis (Y_robot, see imu_pitch_angle), read directly from
    root_ang_vel_b instead of the IMU sensor object — keeps it consistent with imu_pitch_angle (same
    data source, no rotated IMU frame), so pitch_rate always equals d(pitch_angle)/dt.

    Shape: (N, 1).
    """
    robot = env.scene["robot"]
    return robot.data.root_ang_vel_b[:, 1].unsqueeze(1)


def wheel_distance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, wheel_radius: float = 0.034) -> torch.Tensor:
    """Distance traveled (m), computed directly from wheel joint angle (joint_pos).

    If asset_cfg.joint_ids selects multiple joints, their angles are averaged (assumes both wheels
    spin the same sign when driving straight, matching wheel_vel_diff_l2's convention) to reduce
    the effect of one wheel slipping locally. Wheel joint_pos is a continuous, unbounded angle (not
    wrapped to [-pi, pi]), so it's used directly as a position measurement. No-slip rolling:
    dist = angle * wheel_radius. Zero is the joint position at episode reset (reset_wheels).

    Shape: (N, 1).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_angle = asset.data.joint_pos[:, asset_cfg.joint_ids].mean(dim=1)
    return (wheel_angle * wheel_radius).unsqueeze(-1)
