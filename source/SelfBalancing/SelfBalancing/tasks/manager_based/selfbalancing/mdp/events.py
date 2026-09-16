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
    body_offset_z: float = 0.079,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["base_link"]),
) -> None:
    """Push the robot with a FORCE (not a velocity set) along the robot's own body X axis
    (is_global=False) rather than a fixed world axis -- keeps the push aligned with the robot's
    current heading even after yaw drift (from asymmetric wheel friction), so it always reads as
    "push along the direction of travel" instead of injecting an unwanted turning moment.

    Body X is the rolling direction the wheels can actively correct; the wheel axis itself
    (joint_L/joint_R, Y_robot) is not a useful push direction -- that would just tip the robot
    sideways with no way to recover.

    Applied at +Z above base_link's true CoM (body_offset_z) to simulate a hit on the upper body
    rather than right at the CoM/wheel axis -- the same force produces a bigger tip-over moment the
    higher it's applied. Since is_global=False, both "forces" and "positions" are in the body's local
    frame, so body_com_pos_b (already includes the URDF <inertial> offset and any runtime
    randomize_com shift) is used directly as the origin.

    Uses instantaneous_wrench_composer (auto-clears every physics step), so this is a brief kick, not
    a sustained force held for the whole interval_range_s.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    num_bodies = len(body_ids) if isinstance(body_ids, list) else 1

    forces = torch.zeros(len(env_ids), num_bodies, 3, device=asset.device)
    forces[:, :, 0] = sample_uniform(*force_range, (len(env_ids), num_bodies), asset.device)

    positions = asset.data.body_com_pos_b[env_ids][:, body_ids, :].clone()
    positions[:, :, 2] += body_offset_z

    asset.instantaneous_wrench_composer.set_forces_and_torques(
        forces=forces,
        positions=positions,
        body_ids=body_ids,
        env_ids=env_ids,
        is_global=False,
    )


def randomize_wheel_motor_friction_symmetric(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    friction_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Randomize motor joint friction (static/dynamic/viscous), giving BOTH wheels the SAME value per
    env (broadcast (E,1) -> (E, num_joints)) -- unlike the built-in mdp.randomize_joint_parameters,
    which samples independently per joint and could make the two wheels' friction diverge.
    Static/dynamic/viscous are still sampled independently of each other, dynamic clamped <= static
    as expected physically.

    Isaac Sim >=5.0 treats this value as an effort unit (Nm), not a unitless coefficient -- keep
    friction_range small relative to the actuator's effort_limit.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    # mode="startup" calls apply() with env_ids=None, meaning "all envs" -- resolve it the same way
    # the built-in mdp.randomize_joint_parameters does internally.
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    static = sample_uniform(*friction_range, (len(env_ids), 1), asset.device)
    dynamic = torch.minimum(sample_uniform(*friction_range, (len(env_ids), 1), asset.device), static)
    viscous = sample_uniform(*friction_range, (len(env_ids), 1), asset.device)

    asset.write_joint_friction_coefficient_to_sim(
        joint_friction_coeff=static,
        joint_dynamic_friction_coeff=dynamic,
        joint_viscous_friction_coeff=viscous,
        joint_ids=joint_ids,
        env_ids=env_ids,
    )
