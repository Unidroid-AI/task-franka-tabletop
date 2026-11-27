# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One front video, one left-exterior video, and one wrist video per shard.

A shard runs up to 16 environments together. Each video is a single 720x1280
frame [pixels]: a 4x4 grid, env 0 at the top left, then across. An empty cell
stays black. Frames are appended as the shard steps, so the file is playable
once the shard ends.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

VIDEO_DIR = "/output/videos"
# Output frame [pixels], height then width. 4x4 cells of 180x320 fill it exactly.
GRID_ROWS = 4
GRID_COLS = 4
GRID_HW = (720, 1280)
TILE_HW = (GRID_HW[0] // GRID_ROWS, GRID_HW[1] // GRID_COLS)
# The right exterior camera is not written out.
RECORD_CAMERAS = (
    ("cam_front", "front"),
    ("cam_exterior_left", "exterior_left"),
    ("cam_wrist", "wrist"),
)
# Older callers imported the recorded cameras under this name.
CAMERAS = RECORD_CAMERAS

_WRITERS: dict[str, object] = {}


def video_directory() -> str | None:
    """Mounted output directory, or None when this run is not recording."""
    return VIDEO_DIR if Path(VIDEO_DIR).is_dir() else None


def clip_filename(prefix: str, task_id: str, scene: str, variant: str, episode_index: int) -> str:
    """File name for one camera and one episode.

    Shard recordings use :func:`shard_clip_filename` instead. This name remains for callers
    that still address a single episode.
    """
    return f"{prefix}_{task_id}_{scene}_{variant}_{int(episode_index):04d}.mp4"


def shard_clip_filename(
    prefix: str, jobs, task_id: str = "task", scene: str = "scene", variant: str = "nominal"
) -> str:
    """File name for one camera and one shard. The first job's id keeps workers from colliding."""
    first = jobs[0]
    job_id = str(getattr(first, "job_id", "") or "")
    if not job_id:
        job_id = "_".join(
            (
                str(getattr(first, "task_id", None) or task_id),
                str(getattr(first, "scene_id", None) or scene),
                str(getattr(first, "variant", None) or variant),
                f"{int(getattr(first, 'episode_index', 0)):04d}",
            )
        )
    return f"{prefix}_{job_id}.mp4"


def tile_grid(
    images,
    rows: int = GRID_ROWS,
    cols: int = GRID_COLS,
    grid_hw: tuple[int, int] = GRID_HW,
) -> np.ndarray:
    """Pack environment images into one RGB frame, row-major.

    Args:
        images: uint8 images ``[N, tile_h, tile_w, 3]``. ``N`` may be smaller than the grid.
        rows: Grid rows.
        cols: Grid columns.
        grid_hw: Output size [pixels] as ``(height, width)``.

    Returns:
        One uint8 frame of shape ``(height, width, 3)``. Cells past ``N`` are black.

    Raises:
        ValueError: ``images`` is not ``[N, H, W, 3]``, ``N`` exceeds the grid, or a tile
            does not match ``grid_hw`` divided into ``rows`` by ``cols``.
    """
    frames = np.asarray(images)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"images must be [N, H, W, 3], got {frames.shape}")
    count, height, width, _channels = frames.shape
    if count > rows * cols:
        raise ValueError(f"{count} environments do not fit in a {rows}x{cols} grid")
    tile_h, tile_w = grid_hw[0] // rows, grid_hw[1] // cols
    if (height, width) != (tile_h, tile_w):
        raise ValueError(f"tile is {height}x{width}, expected {tile_h}x{tile_w}")
    if (rows * tile_h, cols * tile_w) != grid_hw:
        raise ValueError(f"grid {grid_hw} is not {rows}x{cols} cells of {tile_h}x{tile_w}")
    packed = np.zeros((rows * cols, tile_h, tile_w, 3), dtype=np.uint8)
    packed[:count] = np.asarray(frames, dtype=np.uint8)
    packed = packed.reshape(rows, cols, tile_h, tile_w, 3)
    packed = packed.transpose(0, 2, 1, 3, 4)
    return np.ascontiguousarray(packed.reshape(grid_hw[0], grid_hw[1], 3))


class EpisodeClips:
    """A front grid, a left-exterior grid, and a wrist grid for the current shard."""

    def __init__(self, directory: str | None, write, grid_hw: tuple[int, int] = GRID_HW) -> None:
        self._directory = directory
        self._write = write
        self._grid_hw = grid_hw
        self._names: dict[str, str] = {}
        self._fps = 15
        self.active = False

    def begin(self, jobs, task_id: str, scene: str, variant: str, fps: int = 15) -> None:
        """Close the previous shard's files and start this shard.

        ``task_id``, ``scene``, and ``variant`` name the file when the first job has no id.
        """
        self.end()
        if not self._directory or not jobs:
            return
        self._fps = int(fps)
        self._names = {
            prefix: shard_clip_filename(prefix, jobs, task_id, scene, variant) for _sensor, prefix in RECORD_CAMERAS
        }
        self.active = True

    def add(self, prefix: str, images, active: np.ndarray | None = None) -> None:
        """Append one grid frame for this camera.

        ``images`` is indexed by environment, one tile per env. ``active`` is ignored: a
        finished environment stays in its cell for the rest of the shard.
        """
        del active
        if not self.active or prefix not in self._names:
            return
        self._write(self._directory, self._names[prefix], tile_grid(images, grid_hw=self._grid_hw), self._fps)

    def end(self, fps: int = 15) -> None:
        """Close the shard's mp4 files. ``fps`` was fixed when the shard began."""
        del fps
        if not self.active:
            return
        self.active = False
        close_clips()
        self._names = {}


def write_clip(directory: str, name: str, frame: np.ndarray, fps: int) -> None:
    """Append one RGB frame to an mp4. The file is opened on the first frame.

    The encoder is imported here so a scoring run does not need it.
    """
    path = str(Path(directory) / name)
    writer = _WRITERS.get(path)
    if writer is None:
        os.makedirs(directory, exist_ok=True)
        import imageio.v2 as imageio

        writer = imageio.get_writer(
            path,
            fps=int(fps),
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=1,
        )
        _WRITERS[path] = writer
        print(f"[INFO] Recording {path}", flush=True)
    writer.append_data(np.asarray(frame, dtype=np.uint8))


def close_clips() -> None:
    """Finish every mp4 opened by :func:`write_clip`."""
    writers = list(_WRITERS.values())
    _WRITERS.clear()
    for writer in writers:
        writer.close()
