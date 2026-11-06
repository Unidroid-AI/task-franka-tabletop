# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted joint-space expert for public seeds.

The arm model is a 7-DoF numerical inverse kinematics solver. It is starter
data for fine-tuning, not a claim about a particular simulator's kinematics.
"""

from __future__ import annotations

import numpy as np

# Modified Denavit-Hartenberg parameters for a 7-DoF Franka-like arm: a [m], d [m], alpha [rad].
_A = np.array([0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088])
_D = np.array([0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.0])
_ALPHA = np.array([0.0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2])
HOME = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741], dtype=np.float64)


def fk(joints: np.ndarray) -> np.ndarray:
    """Tool position [m] of a 7-vector of joint angles [rad]."""
    transform = np.eye(4)
    for index, angle in enumerate(np.asarray(joints, dtype=np.float64)):
        transform = transform @ _dh(angle, _A[index], _D[index], _ALPHA[index])
    return transform[:3, 3]


def ik(target_m: np.ndarray, seed: np.ndarray | None = None, steps: int = 80) -> np.ndarray:
    """Damped least-squares inverse kinematics. Returns 7 joint angles [rad]."""
    joints = HOME.copy() if seed is None else np.asarray(seed, dtype=np.float64).copy()
    target = np.asarray(target_m, dtype=np.float64)
    for _step in range(steps):
        position = fk(joints)
        error = target - position
        if float(np.linalg.norm(error)) < 1e-3:
            break
        jacobian = _jacobian(joints)
        damping = 1e-3
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping * np.eye(3), error)
        joints = joints + 0.5 * delta
    return joints


def script_episode(scene, horizon: int = 8) -> list[np.ndarray]:
    """Return absolute 8-D actions: 7 joint positions [rad] and a gripper command in 0..1."""
    item = next(placement for placement in scene.placements if placement.role in ("item", "top", "first", "keep"))
    target = np.array(item.position_m, dtype=np.float64)
    above = target + np.array([0.0, 0.0, 0.12])
    waypoints = [HOME, ik(above), ik(target, seed=ik(above)), ik(above)]
    gripper = [0.0, 0.0, 1.0, 1.0]
    actions = []
    for joints, grip in zip(waypoints, gripper):
        action = np.concatenate([joints, [grip]]).astype(np.float32)
        actions.extend([action.copy() for _repeat in range(horizon)])
    return actions


def _dh(theta: float, a: float, d: float, alpha: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _jacobian(joints: np.ndarray) -> np.ndarray:
    epsilon = 1e-5
    base = fk(joints)
    columns = []
    for index in range(7):
        bumped = joints.copy()
        bumped[index] += epsilon
        columns.append((fk(bumped) - base) / epsilon)
    return np.stack(columns, axis=1)
