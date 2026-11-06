#!/usr/bin/env python3
# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Write a hidden phase bundle. The license sidecar stays outside the bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from franka_tabletop.instructions import phrase
from franka_tabletop.scenes.bundle import read_catalog, read_stage, table_span
from franka_tabletop.scenes.composer import compose

FAMILIES_IN_PHASE = (
    "place_in_container",
    "place_relative",
    "stack_on",
    "sort_by_attribute",
    "multi_place",
)


def write_layouts(
    bundle: Path,
    tasks: tuple[str, ...],
    scene_count: int,
    episodes: int,
    salt: str,
) -> None:
    """Write scene files and the instruction pool. Existing benchmark metadata is left in place."""
    catalog = read_catalog(bundle)
    table_m, center_xy, table_z_m = table_span(read_stage(bundle))
    pools: dict[str, list[str]] = {task_id: [] for task_id in tasks}
    scenes = bundle / "scenes"
    scenes.mkdir(parents=True, exist_ok=True)
    for task_id in tasks:
        layouts = []
        for index in range(scene_count):
            scene_id = f"kitchen_counter-{index}"
            instance = compose(
                10_000 + index,
                task_id,
                catalog,
                table_m=table_m,
                center_xy=center_xy,
                table_z_m=table_z_m,
                archetype="kitchen_counter",
            )
            layouts.append(instance)
            payload = {
                "scene_id": scene_id,
                "task_id": task_id,
                "archetype": instance.archetype,
                "side": instance.side,
                "placements": [
                    {
                        "slot": item.slot,
                        "name": item.asset.name,
                        "usd": item.asset.bundle_usd(),
                        "size_m": list(item.asset.size_m),
                        "position_m": list(item.position_m),
                    }
                    for item in instance.placements
                ],
            }
            text = json.dumps(payload, indent=2)
            (scenes / f"{task_id}-{scene_id}.json").write_text(text, encoding="utf-8")
        for episode in range(episodes):
            sample = layouts[episode % len(layouts)]
            pools[task_id].append(phrase(task_id, episode, sample.names(), salt=salt, side=sample.side))
    pool_dir = bundle / "instruction_pools"
    pool_dir.mkdir(parents=True, exist_ok=True)
    (pool_dir / "phase1_hidden.json").write_text(json.dumps(pools, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Bundle directory")
    parser.add_argument("--licenses", required=True, help="Internal license sidecar, not inside the bundle")
    parser.add_argument("--version", default="franka-tabletop-v1")
    parser.add_argument("--salt", default="phase1-hidden")
    parser.add_argument("--episodes", type=int, default=8)
    args = parser.parse_args()

    bundle = Path(args.output)
    write_layouts(bundle, FAMILIES_IN_PHASE, scene_count=3, episodes=args.episodes, salt=args.salt)
    catalog = read_catalog(bundle)
    benchmark = {
        "version": args.version,
        "shard_size": 4,
        "seed_salt": args.salt,
        "tasks": [
            {
                "task_id": task_id,
                "scenes": [{"composer": "kitchen_counter", "count": 3}],
                "variants": [{"name": "nominal"}, {"name": "jitter10", "pose_jitter_m": 0.10}],
                "instruction_pool": "phase1_hidden",
                "episodes": args.episodes,
            }
            for task_id in FAMILIES_IN_PHASE
        ],
    }
    (bundle / "benchmark.yaml").write_text(yaml.safe_dump(benchmark, sort_keys=False), encoding="utf-8")
    manifest = {
        "id": args.version,
        "name": "Tabletop phase",
        "category": "manipulation",
        "version": args.version,
        "benchmark": True,
        "preset_scenes": [{"scene_id": "0", "preset": "benchmark.yaml"}],
    }
    (bundle / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    licenses = Path(args.licenses)
    licenses.parent.mkdir(parents=True, exist_ok=True)
    licenses.write_text(
        json.dumps(
            {
                "note": "Internal only. Do not ship this file in the bundle.",
                "objects": {name: asset.bundle_usd() for name, asset in catalog.items()},
                "robot": "Franka arm with a 2F-85 gripper.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(bundle)


if __name__ == "__main__":
    main()
