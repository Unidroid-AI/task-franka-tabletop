#!/usr/bin/env python3
# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export a fine-tuning dataset from the scripted expert on public seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from franka_tabletop.expert import script_episode
from franka_tabletop.instructions import phrase
from franka_tabletop.scenes.bundle import read_catalog, read_stage, table_span
from franka_tabletop.scenes.catalog import PUBLIC_SEEDS
from franka_tabletop.scenes.composer import compose
from franka_tabletop.spec import FAMILIES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, help="Benchmark directory that declares the object catalog")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    catalog = read_catalog(args.bundle)
    table_m, center_xy, table_z_m = table_span(read_stage(args.bundle))
    out = Path(args.output)
    (out / "meta").mkdir(parents=True)
    (out / "episodes").mkdir()
    episodes = []
    for task_id in FAMILIES:
        for seed in PUBLIC_SEEDS:
            scene = compose(
                seed,
                task_id,
                catalog,
                table_m=table_m,
                center_xy=center_xy,
                table_z_m=table_z_m,
                archetype="kitchen_counter",
            )
            actions = script_episode(scene)
            name = f"{task_id}-{seed}"
            np.savez(
                out / "episodes" / f"{name}.npz",
                action=np.stack(actions),
                language=np.array(phrase(task_id, 0, scene.names(), side=scene.side)),
            )
            episodes.append(name)
    (out / "meta" / "modality.json").write_text(
        json.dumps(
            {
                "embodiment": "franka_2f85_jointpos",
                "action": {"absolute_joint_position": {"start": 0, "end": 7}, "gripper": {"start": 7, "end": 8}},
                "state": {"joint_position": {"start": 0, "end": 7}, "gripper": {"start": 7, "end": 8}},
                "video": {"cam_exterior_left": {}, "cam_exterior_right": {}, "cam_wrist": {}},
                "annotation": {"language": {}},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "meta" / "info.json").write_text(
        json.dumps({"episodes": episodes, "fps": 20, "format": "npz-starter"}, indent=2),
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()
