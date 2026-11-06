# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Map tabletop observations to a GR00T N1.7 policy and back to absolute joints."""

from __future__ import annotations

import numpy as np

IMAGE_HW = (180, 320)


def resize_hwc(images: np.ndarray, height: int = IMAGE_HW[0], width: int = IMAGE_HW[1]) -> np.ndarray:
    """Nearest-neighbor resize of uint8 HWC images.

    Args:
        images: Shape [H, W, C] or [N, H, W, C].
    """
    array = np.asarray(images)
    batched = array.ndim == 4
    if not batched:
        array = array[None]
    count, src_h, src_w, channels = array.shape
    row = np.clip((np.arange(height) * src_h / height).astype(np.int64), 0, src_h - 1)
    col = np.clip((np.arange(width) * src_w / width).astype(np.int64), 0, src_w - 1)
    resized = array[:, row][:, :, col]
    assert resized.shape == (count, height, width, channels)
    return resized if batched else resized[0]


def relative_chunk_to_absolute(joint_pos: np.ndarray, chunk: np.ndarray) -> np.ndarray:
    """Add relative joint deltas to the current joint position.

    Args:
        joint_pos: Current arm joints [rad], shape [N, 7].
        chunk: Relative chunk, shape [N, H, 8]. The last column is the gripper command.

    Returns:
        Absolute actions, shape [N, H, 8].
    """
    joints = np.asarray(joint_pos, dtype=np.float32)
    actions = np.asarray(chunk, dtype=np.float32).copy()
    actions[..., :7] = joints[:, None, :] + actions[..., :7]
    return actions
