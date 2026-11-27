<p align="center">
  <strong>UNIDROID</strong><br>
  Training general and custom robot brains through challenges.
</p>

# Franka tabletop manipulation in NVIDIA Isaac Lab

**Franka · Robotiq 2F-85 · 8-D joint positions**

[![Isaac Sim](https://img.shields.io/badge/IsaacSim-6.1-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://docs.python.org/3/whatsnew/3.12.html)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

**task-franka-tabletop** is a [Unidroid](https://unidroid.ai) challenge environment for [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab) and [Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html). A **Franka** arm with a **Robotiq 2F-85** gripper follows a language instruction on a table. The action is an **8-D absolute joint position**: seven arm joints in radians and a gripper command.

Use this repository for the Isaac Lab environment a Unidroid Franka tabletop submission is scored on.

| | |
|---|---|
| **Robot** | Franka with a Robotiq 2F-85 gripper |
| **Simulator** | NVIDIA Isaac Sim 6.1 + Isaac Lab 3.0 |
| **Action** | 8 values: 7 absolute arm joints [rad] and a gripper command |
| **Rates** | Physics 120 Hz. Commands every 8 steps (15 Hz) |
| **Cameras sent to the brain** | `cam_exterior_left`, `cam_exterior_right`, `cam_wrist` at 720×1280 |
| **Success** | The task predicate holds on the current state for 0.5 s |
| **Eval** | [isaac-eval](https://github.com/Unidroid-AI/isaac-eval) with `configs/task-franka-tabletop.yaml` |
| **License** | BSD 3-Clause. Copyright (c) 2026 Unidroid |

## Observation

| Key | Shape | Unit |
| --- | --- | --- |
| `cam_exterior_left`, `cam_exterior_right`, `cam_wrist` | `[N, 720, 1280, 3]` uint8 | image |
| `joint_pos` | `[N, 7]` | rad |
| `gripper_pos` | `[N, 1]` | 0 open, 1 closed |
| `eef_pose` | `[N, 7]` | m and wxyz quaternion in the robot-root frame |
| instruction | string per env | passed at episode start |

`cam_exterior_left` and `cam_exterior_right` are the same over-shoulder view from opposite sides. `cam_wrist` looks out from the gripper. `cam_front` looks at the table from the front and is recorded for review. The observation sent to the brain is the two exterior cameras and the wrist.

The reference brain resizes the left exterior and the wrist to 180×320 and converts a relative action chunk to absolute joint targets. Horizon is 8. `cam_exterior_right` is there for a policy trained on two exterior cameras.

Mounting `/output/videos` writes three files per shard: `front_<job_id>.mp4`, `exterior_left_<job_id>.mp4`, and `wrist_<job_id>.mp4`. Each frame is 720×1280, with the shard's environments in a 4×4 grid and env 0 at the top left.

## Task

`Unidroid-FrankaTabletop-Envhub-Play-v0` is the Gym id. `./isaaclab.sh` discovers it through the `isaaclab.tasks` entry point. The phase bundle supplies the scenes, and an episode succeeds when its predicate stays true for 0.5 s.

The phase bundle supplies each scene, including its instruction. A later phase uses another bundle, with different seeds, layouts, and wording. The bundle is the `bundle_path` argument or `ISAAC_EVAL_BUNDLE_DIR` when one of those is set, otherwise `~/.unidroid/bench-cache/<env_id>/bundle`, and otherwise the envhub download.

## Install

Install inside an Isaac Lab 3.0 environment:

```bash
pip install -e source/franka_tabletop
```

## License

BSD-3-Clause. Copyright (c) 2026 Unidroid. See [LICENSE](LICENSE).

**Keywords:** Franka, Robotiq 2F-85, tabletop manipulation, pick and place, language instruction, joint position, NVIDIA Isaac Lab, Isaac Sim, Unidroid, Unidroid-FrankaTabletop.
