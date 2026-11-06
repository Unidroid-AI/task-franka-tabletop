# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registrations. Imported via the isaaclab.tasks entry point."""

import gymnasium as gym

from ..spec import FAMILIES

gym.register(
    id="Nepher-FrankaTabletop-Envhub-Play-v0",
    entry_point="franka_tabletop.core.eval_env:TabletopEvalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "franka_tabletop.env_cfg:TabletopBenchmarkEnvCfg",
    },
)

for _family in FAMILIES:
    _title = "".join(part.capitalize() for part in _family.split("_"))
    gym.register(
        id=f"Nepher-FrankaTabletop-{_title}-Play-v0",
        entry_point="franka_tabletop.core.eval_env:TabletopEvalEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"franka_tabletop.env_cfg:play_{_family}",
        },
    )
