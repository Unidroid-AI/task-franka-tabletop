# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation environment. Finished environments stay frozen until the shard ends."""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import QhullError

from isaaclab.envs import ManagerBasedRLEnv

from .. import geometry
from ..geometry import SURFACE_SLOT
from ..predicates import task_progress
from ..scenes.composer import load_scene, with_jitter, world_position
from ..scenes.mixed import PARK_POSITION_M, SURFACE_ENTITY
from ..success import CONTACT_FORCE_N, Contacts, note_placements, required_pairs, task_satisfied
from .clips import RECORD_CAMERAS, TILE_HW, EpisodeClips, video_directory, write_clip
from .clone import planned_worlds
from .contact import fingers_touching, pair_forces
from .robot import GRIPPER_CLOSED_RAD

# A finished pose has to hold this long before the episode ends, so a one-step bump does not count.
_SUCCESS_HOLD_S = 0.5


class TabletopEvalEnv(ManagerBasedRLEnv):
    """Manager environment that records per-env success and holds finished envs."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        # A mixed stage has a different mesh per env at each object position. ``None`` for copies of one scene.
        self._plan = getattr(cfg, "mixed_plan", None)
        if self._plan is None:
            super().__init__(cfg, render_mode, **kwargs)
        else:
            with planned_worlds(self._plan):
                super().__init__(cfg, render_mode, **kwargs)
        if not getattr(self, "step_dt", None):
            self.step_dt = float(self.cfg.decimation) * float(self.cfg.sim.dt)
        count = self.num_envs
        self._success = torch.zeros(count, dtype=torch.bool, device=self.device)
        self._failed = torch.zeros(count, dtype=torch.bool, device=self.device)
        self._frozen = torch.zeros(count, dtype=torch.bool, device=self.device)
        self._jobs = []
        self._scenes = []
        self._instructions: list[str] = []
        self._hand_xyz: list[list[np.ndarray]] = [[] for _ in range(count)]
        self._slots = tuple(getattr(cfg, "object_slots", ()))
        self._own_slots = {slot: slot for slot in self._slots}
        self._partners = dict(getattr(cfg, "pair_partners", None) or {})
        self._pairs: list[tuple[tuple[str, str], ...]] = []
        self._geometry: dict[tuple[str, int], geometry.AssetGeometry] = {}
        self._surface_geometry: dict[int, geometry.AssetGeometry] = {}
        self._updated_at = -1
        self._latched = [{} for _ in range(count)]
        self._entered = [[] for _ in range(count)]
        self._first_placed: list[dict[str, int]] = [{} for _ in range(count)]
        # Slot to the first pose [m] and wxyz quaternion read after reset. A guarded reference compares to it.
        self._initial: list[dict[str, tuple[np.ndarray, np.ndarray]]] = [{} for _ in range(count)]
        self._hold_s = np.zeros(count, dtype=np.float64)
        self._success_hold_s = np.zeros(count, dtype=np.float64)
        self._progress = np.zeros(count, dtype=np.float64)
        self._videos = EpisodeClips(video_directory(), write_clip)
        # Lockstep leaves cameras off between brain reads. A recording run needs every step.
        self.record_video = video_directory() is not None
        # Env origins never move, so they are copied to the host once.
        self._origins: np.ndarray | None = None

    def set_episode_jobs(self, jobs) -> None:
        """Store the shard's jobs. ``reset`` composes one scene per job."""
        self._jobs = list(jobs)

    def close(self):
        self._videos.end(self._control_fps())
        super().close()

    def reset(self, seed: int | None = None, options: dict | None = None):
        self._videos.end(self._control_fps())
        self._success.zero_()
        self._failed.zero_()
        self._frozen.zero_()
        self._scenes = []
        self._instructions = []
        self._hand_xyz = [[] for _ in range(self.num_envs)]
        self._pairs = []
        self._updated_at = -1
        self._latched = [{} for _ in range(self.num_envs)]
        self._entered = [[] for _ in range(self.num_envs)]
        self._first_placed = [{} for _ in range(self.num_envs)]
        self._initial = [{} for _ in range(self.num_envs)]
        self._hold_s.fill(0.0)
        self._success_hold_s.fill(0.0)
        self._progress.fill(0.0)
        result = super().reset(seed=seed, options=options)
        jobs = self._jobs or [None] * self.num_envs
        for index in range(self.num_envs):
            job = jobs[index] if index < len(jobs) else jobs[-1]
            episode_seed = int(getattr(job, "seed", seed or 1))
            task_id = getattr(job, "task_id", self.cfg.task_id)
            scene_id = getattr(job, "scene_id", self.cfg.scene_id)
            if self._plan is not None:
                # The stage was built for these scenes, env by env. Another job would find the wrong meshes.
                planned = self.cfg.mixed_scenes[index]
                if job is not None and (task_id, scene_id) != planned:
                    raise ValueError(f"env {index} was built for {planned}, not for {(task_id, scene_id)}")
                task_id, scene_id = planned
            published = load_scene(self.cfg.bundle_path, task_id, scene_id) if self.cfg.bundle_path else None
            if published is None:
                raise FileNotFoundError(
                    f"benchmark scene {task_id}-{scene_id} is required under {self.cfg.bundle_path or '(no bundle)'}"
                )
            scene = with_jitter(published, episode_seed, float(self.cfg.pose_jitter_m))
            self._scenes.append(scene)
            anchors = {item.slot for item in scene.placements if item.asset.kind == "anchor"}
            if scene.spec is None:
                raise ValueError(f"scene {scene.task_id} has no success block")
            self._pairs.append(required_pairs(scene.spec, anchors))
            spoken = scene.instruction or getattr(job, "instruction", "") or ""
            self._instructions.append(spoken)
            self._write_poses(index, scene)
        self._videos.begin(
            self._jobs, self.cfg.task_id, self.cfg.scene_id, self.cfg.variant, fps=self._control_fps()
        )
        return result

    def step(self, action):
        # The lockstep loop passes a numpy action. Lab 3 applies it as a tensor.
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(np.asarray(action), dtype=torch.float32, device=self.device)
        action = _snap_gripper(action)
        result = super().step(action)
        self._capture_videos()
        self._record_hand()
        self.update_task_state()
        return result

    def _control_dt(self) -> float:
        step_dt = float(getattr(self, "step_dt", 0.0) or 0.0)
        if step_dt <= 0.0:
            return 1.0 / 15.0
        return step_dt

    def _control_fps(self) -> int:
        return max(1, round(1.0 / self._control_dt()))

    def _capture_videos(self) -> None:
        if not self._videos.active:
            return
        from .observations import camera_rgb, resize_rgb

        # One cell of the 4x4 grid. The left and wrist cameras still render at full size for the brain.
        for sensor, prefix in RECORD_CAMERAS:
            image = resize_rgb(camera_rgb(self, sensor), TILE_HW)
            self._videos.add(prefix, image.detach().cpu().numpy())

    def get_obs(self) -> dict[str, np.ndarray]:
        from .observations import pack_numpy

        return pack_numpy(self)

    def task_completed(self) -> np.ndarray:
        return self._success.detach().cpu().numpy()

    def task_failed(self) -> np.ndarray:
        return self._failed.detach().cpu().numpy()

    def task_progress(self) -> np.ndarray:
        """Latched subtask progress in ``[0, 1]``, one value per environment."""
        return np.asarray(self._progress, dtype=np.float64)

    def instructions(self) -> list[str]:
        return list(self._instructions)

    def hand_positions(self) -> list[np.ndarray]:
        """Hand position [m] in the robot base frame up to the success step, one [T, 3] array per env."""
        return [np.stack(path) if path else np.zeros((0, 3)) for path in self._hand_xyz]

    def _record_hand(self) -> None:
        from .observations import eef_pose

        pose = eef_pose(self).detach().cpu().numpy()
        frozen = self._frozen.detach().cpu().numpy()
        for index in range(self.num_envs):
            if frozen[index]:
                continue
            self._hand_xyz[index].append(np.asarray(pose[index, :3], dtype=np.float64).copy())

    def get_raw_state(self) -> np.ndarray:
        """Live root positions [m] for each placed object, padded to a common width.

        An anchor is a fixed reference in the scene file. It is not a spawned body,
        so its authored position [m] is recorded instead of a simulator root.
        """
        rows = []
        for index, scene in enumerate(self._scenes):
            row = []
            origin = self._origin_xyz(index)
            for placement in scene.placements:
                if placement.asset.kind == "anchor":
                    row.append([float(value) for value in placement.position_m])
                    continue
                entity = self.scene[self._entity_name(index, placement.slot)]
                pos = entity.data.root_pos_w[index].detach().cpu().numpy().reshape(3) - origin
                row.append(pos.tolist())
            rows.append(row)
        if not rows:
            return np.zeros((self.num_envs, 1, 3), dtype=np.float64)
        width = max(len(row) for row in rows)
        padded = [row + [[0.0, 0.0, 0.0]] * (width - len(row)) for row in rows]
        return np.asarray(padded, dtype=np.float64)

    def success_flags(self) -> torch.Tensor:
        """Bool tensor, true where the success expression has held for 0.5 s."""
        return self._success

    def update_task_state(self) -> None:
        """Score every running env on the current state. Runs once per control step.

        Success is stateless: the expression has to hold on the current state, then keep holding for
        :data:`_SUCCESS_HOLD_S`. The latched progress is scored on the side and never decides success.
        """
        step = int(self.common_step_counter)
        if step == self._updated_at or not self._scenes:
            return
        self._updated_at = step
        touching_all = fingers_touching(self.scene, self.num_envs, self._slots)
        forces = pair_forces(self.scene, self.num_envs, self._partners)
        frozen = self._frozen.detach().cpu().numpy()
        roots = self._root_poses()
        dt = self._control_dt()
        for index, scene in enumerate(self._scenes):
            if frozen[index]:
                continue
            if scene.spec is None:
                raise ValueError(f"scene {scene.task_id} has no success block")
            spec = scene.spec
            poses = self._read_poses(index, scene, roots)
            names = self._slot_names(index)
            touching = {slot: bool(touching_all[entity][index]) for slot, entity in names.items()}
            contacts = self._contacts(index, touching, forces)
            score, latched, entered = task_progress(
                spec, poses, contacts, self._latched[index], self._entered[index], side=scene.side
            )
            self._progress[index] = score
            # Latch a finished pose only after it has held, so a one-step bump is not stored.
            if score >= 1.0 - 1e-9:
                self._hold_s[index] += dt
                if self._hold_s[index] >= _SUCCESS_HOLD_S:
                    self._latched[index] = latched
                    self._entered[index] = entered
            else:
                self._hold_s[index] = 0.0
                self._latched[index] = latched
                self._entered[index] = entered
            step_index = int(self.episode_length_buf[index])
            note_placements(spec, poses, contacts, self._first_placed[index], step_index, side=scene.side)
            satisfied = task_satisfied(spec, poses, contacts, self._first_placed[index], side=scene.side)
            self._success_hold_s[index] = self._success_hold_s[index] + dt if satisfied else 0.0
            if self._success_hold_s[index] >= _SUCCESS_HOLD_S:
                self._success[index] = True
                self._frozen[index] = True

    def _reset_idx(self, env_ids):
        """Reset the running envs. A finished env keeps its final state until ``reset`` composes new scenes."""
        frozen = getattr(self, "_frozen", None)
        if frozen is None:
            super()._reset_idx(env_ids)
            return
        ids = torch.as_tensor(env_ids, device=self.device).reshape(-1)
        ids = ids[~frozen[ids]]
        if ids.numel():
            super()._reset_idx(ids)

    def _contacts(self, index: int, touching: dict[str, bool], forces: dict[tuple[str, str], np.ndarray]) -> Contacts:
        """Finger contact and the pair forces [N] that env ``index`` reads."""
        names = self._slot_names(index)
        touch: dict[tuple[str, str], bool] = {}
        force: dict[tuple[str, str], np.ndarray] = {}
        for item, partner in self._pairs[index]:
            other = SURFACE_SLOT if partner == SURFACE_SLOT else names.get(partner)
            found = forces.get((names.get(item), other))
            if found is None:
                continue
            force[(item, partner)] = found[index]
            touch[(item, partner)] = bool(np.any(np.abs(found[index]) > CONTACT_FORCE_N))
        return Contacts(touching=touching, touch=touch, force=force)

    def _origin_xyz(self, index: int) -> np.ndarray:
        if self._origins is None:
            self._origins = self.scene.env_origins.detach().cpu().numpy().reshape(-1, 3)
        return self._origins[index]

    def _root_poses(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Root position [m] and xyzw quaternion of every object slot, one host copy per slot."""
        roots = {}
        for slot in self._slots:
            data = self.scene[slot].data
            roots[slot] = (_host(data.root_pos_w).reshape(-1, 3), _host(data.root_quat_w).reshape(-1, 4))
        return roots

    def _slot_names(self, index: int) -> dict[str, str]:
        """Scene slot of each body in env ``index`` mapped to the scene entity that holds it."""
        if self._plan is None:
            return self._own_slots
        return self._plan.slot_entities[index]

    def _entity_name(self, index: int, slot: str) -> str:
        return self._slot_names(index)[slot]

    def _write_poses(self, index: int, scene) -> None:
        origin = tuple(float(value) for value in self._origin_xyz(index))
        for placement in scene.placements:
            if placement.asset.kind == "anchor":
                continue
            entity = self.scene[self._entity_name(index, placement.slot)]
            self._write_root_pose(entity, index, world_position(placement.position_m, origin), placement.quat_wxyz)
        if self._plan is None:
            return
        # Unused positions wait above the robot. The work surface takes this scene's own pose.
        for name in self._plan.parked(index):
            self._write_root_pose(
                self.scene[name], index, world_position(PARK_POSITION_M, origin), (1.0, 0.0, 0.0, 0.0)
            )
        position, quat = self._plan.surface_poses[index]
        self._write_root_pose(self.scene[SURFACE_ENTITY], index, world_position(position, origin), quat)

    def _write_root_pose(self, entity, index: int, position: tuple[float, float, float], quat_wxyz) -> None:
        w, x, y, z = quat_wxyz
        pose = torch.tensor([[*position, x, y, z, w]], device=self.device, dtype=torch.float32)
        ids = torch.tensor([index], device=self.device)
        if hasattr(entity, "write_root_pose_to_sim"):
            entity.write_root_pose_to_sim(pose, env_ids=ids)
        # Drop velocity left by the reset settle, so a posed object does not leave the table.
        if hasattr(entity, "write_root_velocity_to_sim"):
            entity.write_root_velocity_to_sim(
                torch.zeros((1, 6), device=self.device, dtype=torch.float32),
                env_ids=ids,
            )

    def _read_poses(self, index: int, scene, roots: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
        """Pose [m], wxyz quaternion, and geometry per slot. An anchor and ``surface`` share the work surface."""
        surface = self._surface_pose(index)
        poses = {SURFACE_SLOT: surface}
        origin = self._origin_xyz(index)
        for placement in scene.placements:
            if placement.asset.kind == "anchor":
                poses[placement.slot] = surface
                continue
            root_pos, root_quat = roots[self._entity_name(index, placement.slot)]
            pos = (root_pos[index] - origin).astype(np.float64).reshape(1, 3)
            quat = _as_wxyz(root_quat[index]).reshape(1, 4)
            start = self._initial[index].setdefault(placement.slot, (pos.copy(), quat.copy()))
            poses[placement.slot] = {
                "pos": pos,
                "quat": quat,
                "geometry": self._body_geometry(index, placement),
                "initial_pos": start[0],
                "initial_quat": start[1],
            }
        return poses

    def _surface_pose(self, index: int) -> dict:
        """Work-surface pose [m] in the env frame, with its geometry."""
        if self._plan is None:
            init = self.cfg.scene.surface.init_state
            pos = init.pos
            x, y, z, w = init.rot
            quat = (w, x, y, z)
        else:
            pos, quat = self._plan.surface_poses[index]
        return {
            "pos": np.asarray(pos, dtype=np.float64).reshape(1, 3),
            "quat": np.asarray(quat, dtype=np.float64).reshape(1, 4),
            "geometry": self._surface(index),
            "anchor": True,
        }

    def _body_geometry(self, index: int, placement) -> geometry.AssetGeometry:
        """Geometry of the mesh that env ``index`` spawned for ``placement``. Built once per mesh."""
        entity = self._entity_name(index, placement.slot)
        if self._plan is None:
            choice, sample = 0, 0
        else:
            choices = self._plan.choices[self._plan.entities.index(entity)]
            choice = choices[index]
            sample = next(env for env in range(self.num_envs) if choices[env] == choice)
        key = (entity, choice)
        if key not in self._geometry:
            path = f"/World/envs/env_{sample}/{entity}"
            try:
                self._geometry[key] = geometry.from_points(geometry.read_local_points(_usd_stage(), path))
            except (ValueError, QhullError):  # no mesh, or a mesh without volume
                self._geometry[key] = geometry.from_box(placement.asset.size_m, warn=f"{placement.asset.name} ({path})")
        return self._geometry[key]

    def _surface(self, index: int) -> geometry.AssetGeometry:
        """Geometry of the work surface env ``index`` spawned. Built once per surface mesh."""
        if self._plan is None:
            choice, sample = 0, 0
        else:
            choices = self._plan.surface_choices
            choice = choices[index]
            sample = next(env for env in range(self.num_envs) if choices[env] == choice)
        if choice not in self._surface_geometry:
            points = geometry.read_local_points(_usd_stage(), f"/World/envs/env_{sample}/table")
            self._surface_geometry[choice] = geometry.from_points(points)
        return self._surface_geometry[choice]


def _snap_gripper(action: torch.Tensor) -> torch.Tensor:
    """Map a 0..1 gripper command to open (0) or closed (pi/4). Values above 0.5 close."""
    if action.shape[-1] != 8:
        return action
    action = action.clone()
    closed = action[..., 7] > 0.5
    action[..., 7] = torch.where(
        closed,
        torch.full_like(action[..., 7], GRIPPER_CLOSED_RAD),
        torch.zeros_like(action[..., 7]),
    )
    return action


def _usd_stage():
    import omni.usd

    return omni.usd.get_context().get_stage()


def _host(values) -> np.ndarray:
    """Copy a torch tensor, or a Warp-backed array that exposes ``.torch``, to a numpy array."""
    values = values.torch if hasattr(values, "torch") else values
    return values.detach().cpu().numpy()


def _as_wxyz(quat: np.ndarray) -> np.ndarray:
    """Convert an Isaac Lab 3 xyzw root quaternion to wxyz."""
    x, y, z, w = np.asarray(quat, dtype=np.float64).reshape(-1)
    return np.array([w, x, y, z], dtype=np.float64)
