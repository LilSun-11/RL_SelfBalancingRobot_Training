# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def widen_velocity_range(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    data: tuple[float, float],
    start_range: tuple[float, float],
    end_range: tuple[float, float],
    num_steps: int,
) -> tuple[float, float]:
    """Linearly widen a command's ``ranges`` from ``start_range`` to ``end_range`` as
    ``env.common_step_counter`` goes from 0 to ``num_steps``, then holds at ``end_range``.

    Meant for :class:`isaaclab.envs.mdp.curriculums.modify_term_cfg`, targeting
    ``"commands.<command_name>.ranges"``: lets the policy first learn to balance (near-zero
    velocity command) before it has to learn the lean-forward-to-accelerate coupling needed to track
    a nonzero body velocity target, instead of facing the full range from step 0.

    ``env.common_step_counter`` resets to 0 at the start of every training process, including a
    resumed run (``--resume``) -- the ramp restarts from ``start_range`` each time train.py is
    launched; it does not carry over progress across separate invocations.
    """
    progress = min(env.common_step_counter / num_steps, 1.0)
    low = start_range[0] + progress * (end_range[0] - start_range[0])
    high = start_range[1] + progress * (end_range[1] - start_range[1])
    return (low, high)
