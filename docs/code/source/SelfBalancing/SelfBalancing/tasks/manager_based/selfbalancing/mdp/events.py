"""Event terms: randomization and disturbances."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def push_by_external_force_local_x(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    body_offset_z: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["base_link"]),
) -> None:
    """Kick the chassis forward/backward along its own X axis for one physics step, at
    ``body_offset_z`` above its center of mass (like a hit on the upper body)."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    num_bodies = len(body_ids) if isinstance(body_ids, list) else 1

    forces = torch.zeros(len(env_ids), num_bodies, 3, device=asset.device)
    forces[:, :, 0] = sample_uniform(*force_range, (len(env_ids), num_bodies), asset.device)
    positions = asset.data.body_com_pos_b[env_ids][:, body_ids, :].clone()
    positions[:, :, 2] += body_offset_z

    # the instantaneous composer clears itself after one physics step -> a short kick
    asset.instantaneous_wrench_composer.set_forces_and_torques(
        forces=forces, positions=positions, body_ids=body_ids, env_ids=env_ids, is_global=False
    )


def reset_joints_by_offset_symmetric(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Like the built-in reset_joints_by_offset, but with ONE random offset per env shared by all
    selected joints, so both wheels start in the same state."""
    asset: Articulation = env.scene[asset_cfg.name]
    iter_env_ids = env_ids[:, None] if asset_cfg.joint_ids != slice(None) else env_ids

    joint_pos = asset.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_pos += sample_uniform(*position_range, (len(env_ids), 1), joint_pos.device)
    joint_vel += sample_uniform(*velocity_range, (len(env_ids), 1), joint_vel.device)

    pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(pos_limits[..., 0], pos_limits[..., 1])
    vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-vel_limits, vel_limits)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def randomize_wheel_motor_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    friction_range: tuple[float, float],
    asymmetry: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Randomize the gearbox friction of the wheel joints (Nm in Isaac Sim 5.x).

    One value per env is sampled from ``friction_range``; each wheel is then multiplied by its own
    factor in [1 - asymmetry, 1 + asymmetry], so the left and right motors differ like real ones.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:  # mode="startup" passes None = all envs
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    joint_ids = asset_cfg.joint_ids
    num_joints = asset.num_joints if isinstance(joint_ids, slice) else len(joint_ids)

    static = sample_uniform(*friction_range, (len(env_ids), 1), asset.device)
    dynamic = torch.minimum(sample_uniform(*friction_range, (len(env_ids), 1), asset.device), static)
    viscous = sample_uniform(*friction_range, (len(env_ids), 1), asset.device)
    per_wheel = sample_uniform(1.0 - asymmetry, 1.0 + asymmetry, (len(env_ids), num_joints), asset.device)

    asset.write_joint_friction_coefficient_to_sim(
        joint_friction_coeff=static * per_wheel,
        joint_dynamic_friction_coeff=dynamic * per_wheel,
        joint_viscous_friction_coeff=viscous * per_wheel,
        joint_ids=joint_ids,
        env_ids=env_ids,
    )
