# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adapt a tabletop env to the evaluation worker."""

from __future__ import annotations


def wrap_for_eval(env):
    """Return the env when it already exposes the benchmark methods.

    The methods are ``task_completed``, ``task_failed``, ``instructions``,
    ``get_raw_state``, and ``get_obs``.
    """
    required = ("task_completed", "task_failed", "instructions", "get_raw_state", "get_obs")
    target = env.unwrapped if hasattr(env, "unwrapped") else env
    missing = [name for name in required if not hasattr(target, name)]
    if missing:
        raise TypeError(f"environment is missing benchmark methods: {missing}")
    return target
