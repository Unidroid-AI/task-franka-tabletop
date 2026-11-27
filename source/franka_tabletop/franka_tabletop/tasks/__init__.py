# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration. Imported via the isaaclab.tasks entry point."""

import gymnasium as gym

from ..eval_compat import wrap_for_eval

__all__ = ["wrap_for_eval"]

gym.register(
    id="Unidroid-FrankaTabletop-Envhub-Play-v0",
    entry_point="franka_tabletop.core.eval_env:TabletopEvalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "franka_tabletop.env_cfg:TabletopBenchmarkEnvCfg",
    },
)
