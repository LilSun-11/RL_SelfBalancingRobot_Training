"""Curriculum terms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs.mdp import modify_term_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def staged_value_curriculum(env: ManagerBasedRLEnv, env_ids: Sequence[int], data, stages: list):
    """For ``modify_term_cfg``: set a config value from a list of ``(start_step, value)`` stages, sorted
    by start_step. The value of the last stage that has started is used (a step change, not a ramp).
    The step counter restarts at 0 every time train.py is launched (also with --resume)."""
    new_value = stages[0][1]
    for start_step, value in stages:
        if env.common_step_counter >= start_step:
            new_value = value
    return modify_term_cfg.NO_CHANGE if new_value == data else new_value
