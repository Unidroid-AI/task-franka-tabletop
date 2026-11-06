# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Zero-shot GR00T N1.7 brain. The policy package is provided by the brain image."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nepher_brain_comm.brain import Brain

from adapt import relative_chunk_to_absolute, resize_hwc


class Gr00tN17Brain(Brain):
    """Load a GR00T N1.7 checkpoint and return absolute joint actions."""

    def setup(self, device: str, weights_dir: Path, config: dict) -> None:
        self.horizon = int(config["horizon"])
        self._tag = config.get("embodiment_tag")
        self._policy = _load_policy(weights_dir, self._tag, device)
        self._joint_pos = np.zeros((1, 7), dtype=np.float32)
        self._instructions: list[str] = []

    def begin_episode(self, episode_ids: list[str], seeds: list[int], instructions: list[str]) -> None:
        self._instructions = list(instructions)

    def act(self, obs: dict[str, np.ndarray], step: int) -> np.ndarray:
        self._joint_pos = np.asarray(obs["joint_pos"], dtype=np.float32)
        if self._policy is None:
            count = self._joint_pos.shape[0]
            return np.zeros((count, self.horizon, 8), dtype=np.float32)
        model_obs = {
            "video": {
                "exterior": resize_hwc(obs["cam_exterior_left"]),
                "wrist": resize_hwc(obs["cam_wrist"]),
            },
            "state": {
                "joint_pos": self._joint_pos,
                "gripper_pos": np.asarray(obs["gripper_pos"], dtype=np.float32),
                "eef_pose": np.asarray(obs["eef_pose"], dtype=np.float32),
            },
            "language": self._instructions,
        }
        chunk = np.asarray(self._policy.get_action(model_obs), dtype=np.float32)
        return relative_chunk_to_absolute(self._joint_pos, chunk[..., :8])


def _load_policy(weights_dir: Path, embodiment_tag: str | None, device: str):
    if not any(weights_dir.glob("config.json")):
        return None
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    return Gr00tPolicy(model_path=str(weights_dir), embodiment_tag=embodiment_tag, device=device)
