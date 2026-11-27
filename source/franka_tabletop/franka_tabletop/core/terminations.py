# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode terminations of the tabletop evaluation environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .eval_env import TabletopEvalEnv


def task_success(env: TabletopEvalEnv) -> torch.Tensor:
    """True once the success expression has held on the current state for 0.5 s."""
    env.update_task_state()
    return env.success_flags()
