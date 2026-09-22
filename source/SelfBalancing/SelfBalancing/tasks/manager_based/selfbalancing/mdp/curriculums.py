# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
