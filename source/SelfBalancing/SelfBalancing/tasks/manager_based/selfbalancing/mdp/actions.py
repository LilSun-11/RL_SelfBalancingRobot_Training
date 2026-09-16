# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

# SymmetricWheelEffortAction (forced both wheels to the same torque, one action for the pair) has
# been replaced by the built-in mdp.JointEffortActionCfg (independent torque per wheel, see
# ActionsCfg in selfbalancing_env_cfg.py) so the policy can compensate for wheel-to-wheel friction
# mismatch. See git history for the old single-action implementation.
