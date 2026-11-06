#!/usr/bin/env python3
# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run one public seed against a local brain socket.

Example:
    python scripts/play_brain.py --bundle <benchmark> --task place_in_container --socket /tmp/brain/brain-0.sock
"""

from __future__ import annotations

import argparse

from franka_tabletop.instructions import phrase
from franka_tabletop.scenes.bundle import read_catalog, read_stage, table_span
from franka_tabletop.scenes.catalog import PUBLIC_SEEDS
from franka_tabletop.scenes.composer import compose


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, help="Downloaded benchmark directory")
    parser.add_argument("--task", default="place_in_container")
    parser.add_argument("--seed", type=int, default=PUBLIC_SEEDS[0])
    parser.add_argument("--archetype", default="kitchen_counter")
    parser.add_argument("--socket", required=True)
    args = parser.parse_args()
    catalog = read_catalog(args.bundle)
    table_m, center_xy, table_z_m = table_span(read_stage(args.bundle))
    scene = compose(
        args.seed,
        args.task,
        catalog,
        table_m=table_m,
        center_xy=center_xy,
        table_z_m=table_z_m,
        archetype=args.archetype,
    )
    text = phrase(args.task, 0, scene.names(), side=scene.side)
    print(f"instruction: {text}")
    print(f"objects: {scene.names()}")
    print(f"brain socket: {args.socket}")
    print("Start the simulator with isaaclab.sh -p scripts/evaluate.py --config <eval yaml>.")


if __name__ == "__main__":
    main()
