# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contact readings. A force component above ``CONTACT_FORCE_N`` counts as touching, as in the reference benchmark.

Finger sensors filter every object entity, in entity order. Each object that a task reads a pair
for has its own sensor, filtered on the bodies it is paired with. PhysX reports the normal force.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from ..geometry import SURFACE_SLOT
from ..spec import TabletopTaskSpec
from ..success import CONTACT_FORCE_N, required_pairs

FINGER_SENSORS = ("gripper_left", "gripper_right")


def pair_sensor_name(entity: str) -> str:
    """Scene name of the sensor on object entity ``entity``."""
    return f"contact_{entity}"


def contact_partners(
    envs: Iterable[tuple[TabletopTaskSpec, Iterable[str], dict[str, str]]],
) -> dict[str, tuple[str, ...]]:
    """Per object entity, the bodies whose contact force on it any env reads.

    Args:
        envs: One ``(spec, anchor slots, slot to entity)`` per env. An anchor slot has no body, so the
            work surface stands in for it.
    """
    partners: dict[str, list[str]] = {}
    for spec, anchors, entities in envs:
        for item, partner in required_pairs(spec, frozenset(anchors)):
            body = entities.get(item)
            other = partner if partner == SURFACE_SLOT else entities.get(partner)
            if body is None or other is None:
                continue
            found = partners.setdefault(body, [])
            if other not in found:
                found.append(other)
    return {body: tuple(found) for body, found in partners.items()}


def fingers_touching(scene, num_envs: int, entities: Sequence[str]) -> dict[str, np.ndarray]:
    """Per entity, whether either finger touches it, as bool arrays of length ``num_envs``.

    ``entities`` is in finger-sensor filter order.
    """
    touching = {entity: np.zeros(num_envs, dtype=bool) for entity in entities}
    if not entities:
        return touching
    for name in FINGER_SENSORS:
        matrix = scene[name].data.normal_force_matrix_w
        if matrix is None:
            continue
        forces = _array(matrix)
        if forces.shape[2] != len(entities):
            raise ValueError(f"contact sensor {name} has {forces.shape[2]} filters, expected {len(entities)}")
        for index, entity in enumerate(entities):
            touching[entity] |= np.any(np.abs(forces[:, :, index]) > CONTACT_FORCE_N, axis=(1, 2))
    return touching


def pair_forces(scene, num_envs: int, partners: dict[str, tuple[str, ...]]) -> dict[tuple[str, str], np.ndarray]:
    """Force [N] on each ``(entity, partner)`` body from its partner, shape ``[num_envs, 3]``."""
    found: dict[tuple[str, str], np.ndarray] = {}
    for entity, others in partners.items():
        matrix = scene[pair_sensor_name(entity)].data.normal_force_matrix_w
        if matrix is None:
            continue
        forces = _array(matrix).sum(axis=1)
        if forces.shape[1] != len(others):
            raise ValueError(f"contact sensor of {entity} has {forces.shape[1]} filters, expected {len(others)}")
        for index, other in enumerate(others):
            found[(entity, other)] = forces[:, index]
    return found


def _array(matrix) -> np.ndarray:
    """Force matrix on the host, shape ``[num_envs, bodies, filters, 3]``."""
    values = matrix.torch if hasattr(matrix, "torch") else matrix
    return values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)
