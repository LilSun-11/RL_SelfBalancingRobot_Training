"""Curriculum terms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs.mdp import modify_term_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def step_value_curriculum(env: ManagerBasedRLEnv, env_ids: Sequence[int], data, start_value, end_value, num_steps: int):
    """For ``modify_term_cfg``: keep a config value at ``start_value`` until ``num_steps`` environment
    steps have passed, then switch it to ``end_value``. The step counter restarts at 0 every time
    train.py is launched (also with --resume)."""
    new_value = start_value if env.common_step_counter < num_steps else end_value
    return modify_term_cfg.NO_CHANGE if new_value == data else new_value
