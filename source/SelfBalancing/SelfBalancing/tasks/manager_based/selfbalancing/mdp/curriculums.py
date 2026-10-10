# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class enable_command_when_proficient(ManagerTermBase):
    """Keep a command's range at its initial value (e.g. yaw = 0) until the policy has learned to
    balance and track the forward-velocity command, then switch it to ``enabled_range`` once and keep
    it there.

    "Proficient" means all of, at the same time:
      - ``env.common_step_counter >= min_steps`` (e.g. the velocity curriculum has fully opened, so
        velocity tracking is being judged on the final range, not on a near-zero one);
      - fall rate (fraction of resetting envs that ended by a termination other than time-out,
        exponentially averaged) ``<= max_fall_rate``;
      - mean |target - actual| body-X velocity over all envs (exponentially averaged)
        ``<= max_velocity_error``.

    Called on every reset, like other curriculum terms. Returns its state so it is logged to
    TensorBoard as ``Curriculum/<term>/{enabled, fall_rate, velocity_error}``. Like the other
    curricula here, the state is per-process: a resumed run (``--resume``) starts gated again.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._enabled = False
        self._fall_rate: float | None = None
        self._velocity_error: float | None = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        command_name: str,
        velocity_command_name: str,
        enabled_range: tuple[float, float],
        min_steps: int,
        max_fall_rate: float,
        max_velocity_error: float,
        ema_alpha: float = 0.01,
        asset_name: str = "robot",
    ) -> dict[str, float]:
        # skip the initial reset at env creation: nothing has fallen yet, which would seed the
        # averages with an optimistic fall rate of 0
        if not self._enabled and env.common_step_counter > 0:
            fall_rate = env.termination_manager.terminated[env_ids].float().mean().item()
            target = env.command_manager.get_command(velocity_command_name)[:, 0]
            actual = env.scene[asset_name].data.root_lin_vel_b[:, 0]
            velocity_error = (target - actual).abs().mean().item()
            self._fall_rate = fall_rate if self._fall_rate is None else self._ema(self._fall_rate, fall_rate, ema_alpha)
            self._velocity_error = (
                velocity_error
                if self._velocity_error is None
                else self._ema(self._velocity_error, velocity_error, ema_alpha)
            )

            if (
                env.common_step_counter >= min_steps
                and self._fall_rate <= max_fall_rate
                and self._velocity_error <= max_velocity_error
            ):
                env.command_manager.get_term(command_name).cfg.ranges = enabled_range
                self._enabled = True
                print(
                    f"[INFO] Curriculum: enabled '{command_name}' range {enabled_range} at step"
                    f" {env.common_step_counter} (fall rate {self._fall_rate:.3f}, velocity error"
                    f" {self._velocity_error:.4f} m/s)"
                )

        return {
            "enabled": float(self._enabled),
            "fall_rate": self._fall_rate if self._fall_rate is not None else 1.0,
            "velocity_error": self._velocity_error if self._velocity_error is not None else 0.0,
        }

    @staticmethod
    def _ema(old: float, new: float, alpha: float) -> float:
        return (1.0 - alpha) * old + alpha * new


def linear_range_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    data: tuple[float, float],
    start_range: tuple[float, float],
    end_range: tuple[float, float],
    num_steps: int,
) -> tuple[float, float]:
    """Linearly interpolate a 2-tuple config value from ``start_range`` to ``end_range`` as
    ``env.common_step_counter`` goes from 0 to ``num_steps``, then holds at ``end_range``.
    ``start_range`` can be narrower or wider than ``end_range`` -- the direction is just whichever
    way those two arguments point.

    Meant for :class:`isaaclab.envs.mdp.curriculums.modify_term_cfg`, targeting any 2-tuple field on
    a term's config, e.g. ``"commands.<command_name>.ranges"`` or
    ``"commands.<command_name>.resampling_time_range"``. Used here to both widen target_velocity's
    sampling range and narrow how often it resamples, so the policy first learns to hold a steady,
    near-zero velocity for long stretches before it has to track a wide range of speeds that also
    change every few seconds.

    ``env.common_step_counter`` resets to 0 at the start of every training process, including a
    resumed run (``--resume``) -- the ramp restarts from ``start_range`` each time train.py is
    launched; it does not carry over progress across separate invocations.
    """
    progress = min(env.common_step_counter / num_steps, 1.0)
    low = start_range[0] + progress * (end_range[0] - start_range[0])
    high = start_range[1] + progress * (end_range[1] - start_range[1])
    return (low, high)


def step_value_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    data,
    start_value,
    end_value,
    num_steps: int,
):
    """Hold a config value at ``start_value`` until ``env.common_step_counter`` reaches ``num_steps``,
    then switch it to ``end_value`` for the rest of the run (a hard switch, not a ramp).

    Meant for :class:`isaaclab.envs.mdp.curriculums.modify_term_cfg`, e.g.
    ``"commands.target_yaw.ranges"`` with start (0, 0) so the target heading stays at the spawn heading
    while the policy first learns to balance and track velocity. Like linear_range_curriculum, the
    counter restarts at 0 on every train.py launch, including ``--resume``.
    """
    from isaaclab.envs.mdp import modify_term_cfg

    new_value = start_value if env.common_step_counter < num_steps else end_value
    return modify_term_cfg.NO_CHANGE if new_value == data else new_value
