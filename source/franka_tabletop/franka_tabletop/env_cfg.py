# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for the tabletop benchmark."""

from __future__ import annotations

import json
import os
from pathlib import Path

from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.cloner import CloneCfg, InclusionSet
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .core.clips import TILE_HW, video_directory
from .core.observation_spec import OBSERVATION_CAMERAS, requested_cameras, requested_image_hw, unused_cameras
from .core.observations import camera_rgb, eef_pose, gripper_pos, joint_pos
from .core.robot import franka_2f85, joint_position_action, with_position_iterations
from .core.sensors import exterior_left_camera, exterior_right_camera, finger_contact, front_camera, wrist_camera
from .scenes.bundle import (
    read_stage,
    resolve_background,
    resolve_bundle_path,
    resolve_surface,
    surface_pose,
    xyzw_from_wxyz,
)
from .scenes.mixed import SURFACE_ENTITY, ObjectVariant, plan_mixed

# Replaced from the benchmark scene before the simulation starts.
_PENDING_USD = "assets/objects/pending.usd"
# The brain is commanded at this rate whatever physics rate the benchmark declares.
CONTROL_HZ = 15


def _object(
    slot: str,
    position: tuple[float, float, float],
    rotation: tuple[float, float, float, float],
) -> RigidObjectCfg:
    """One contact slot. The benchmark scene supplies the mesh, including its rigid body.

    ``rotation`` is xyzw. Reset settles from this pose before the episode writes
    positions again, so a yawed fixture must not spawn at identity.
    """
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + slot,
        spawn=sim_utils.UsdFileCfg(
            usd_path=_PENDING_USD,
            activate_contact_sensors=True,
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=rotation),
    )


@configclass
class TabletopSceneCfg(InteractiveSceneCfg):
    """Robot, table, and cameras. Object bodies are added from the scene file."""

    robot = franka_2f85()
    # Mount under the arm. The benchmark replaces the mesh and the pose.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table_fixture",
        spawn=sim_utils.UsdFileCfg(usd_path="assets/fixtures/franka_table.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.087, 0.0, 0.0),
            # 180 degrees about +Z. The benchmark stores that turn as wxyz.
            rot=(0.0, 0.0, 1.0, 0.0),
        ),
    )
    # Work surface the objects rest on. The benchmark chooses which table mesh.
    # Those payloads are dynamic rigid bodies. Hold the surface kinematic so the collider
    # stays on the visual top and placed objects do not fall through it.
    surface = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.UsdFileCfg(
            usd_path="assets/fixtures/pending.usd",
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    cam_exterior_left = exterior_left_camera()
    cam_exterior_right = exterior_right_camera()
    cam_wrist = wrist_camera()
    cam_front = front_camera()
    gripper_left = finger_contact("left_inner_finger")
    gripper_right = finger_contact("right_inner_finger")
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(visible=False),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.697)),
    )
    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/sphere",
        spawn=sim_utils.SphereLightCfg(intensity=5000.0),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.6, 0.7)),
    )
    background = AssetBaseCfg(
        prim_path="/World/background",
        spawn=sim_utils.DomeLightCfg(
            texture_file="assets/backgrounds/home_office.exr",
            intensity=500.0,
            visible_in_primary_ray=True,
            texture_format="latlong",
        ),
    )


@configclass
class ActionsCfg:
    arm = joint_position_action()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=joint_pos)
        gripper_pos = ObsTerm(func=gripper_pos)
        eef_pose = ObsTerm(func=eef_pose)
        cam_exterior_left = ObsTerm(func=camera_rgb, params={"sensor_name": "cam_exterior_left"})
        cam_exterior_right = ObsTerm(func=camera_rgb, params={"sensor_name": "cam_exterior_right"})
        cam_wrist = ObsTerm(func=camera_rgb, params={"sensor_name": "cam_wrist"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """No training reward. Episode success is scored by the evaluator."""


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class TabletopBenchmarkEnvCfg(ManagerBasedRLEnvCfg):
    """Benchmark environment. Use :func:`build_cfg` to fill a phase job."""

    scene: TabletopSceneCfg = TabletopSceneCfg(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    task_id: str = "place_in_container"
    scene_id: str = "kitchen_counter-0"
    variant: str = "nominal"
    composer: str = "kitchen_counter"
    pose_jitter_m: float = 0.0
    env_id: str | None = None
    bundle_path: str = ""
    object_slots: tuple[str, ...] = ()
    mixed_scenes: tuple[tuple[str, str], ...] = ()
    """(task, scene) of each env when one stage holds different objects per env. Empty otherwise."""
    mixed_plan: object | None = None
    """:class:`~franka_tabletop.scenes.mixed.MixedPlan` of a mixed stage. :func:`build_cfg` fills it."""
    observation_cameras: tuple[str, ...] = OBSERVATION_CAMERAS
    """Cameras sent to the brain. :func:`build_cfg` narrows this to what the brain declares."""
    observation_image_hw: tuple[int, int] | None = None
    """Image size [pixels] sent to the brain as (height, width). ``None`` keeps the rendered size."""

    def __post_init__(self) -> None:
        # 120 Hz physics, command every 8 steps: 15 Hz. One step is 1/15 s.
        self.episode_length_s = 40.0
        set_physics_rate(self, 120)
        self.sim.physics = PhysxAutoCfg(isaacsim_physx=PhysxCfg())
        self.sim.physics.isaacsim_physx.enable_enhanced_determinism = True
        # GPU dynamics has no CCD, so the enableCCD authored on the pads and objects is inert.
        self.sim.physics.isaacsim_physx.enable_ccd = False
        # Sixteen envs share one PhysX scene. The default patch buffer is 163840, and a
        # 7-env tabletop already asked for about 194000. Both pair buffers are sized for a full shard.
        physx = self.sim.physics.isaacsim_physx
        physx.gpu_max_rigid_patch_count = 2**21
        physx.gpu_found_lost_pairs_capacity = 2**23


def set_physics_rate(cfg: TabletopBenchmarkEnvCfg, rate_hz: int) -> None:
    """Step physics at ``rate_hz`` [Hz] and keep the brain at :data:`CONTROL_HZ`.

    Raises:
        ValueError: ``rate_hz`` is not a positive multiple of the control rate.
    """
    if rate_hz < CONTROL_HZ or rate_hz % CONTROL_HZ:
        raise ValueError(f"physics rate_hz must be a multiple of {CONTROL_HZ}, got {rate_hz}")
    cfg.decimation = rate_hz // CONTROL_HZ
    cfg.sim.dt = 1.0 / rate_hz
    cfg.sim.render_interval = cfg.decimation


def _apply_benchmark_physics(cfg: TabletopBenchmarkEnvCfg, physics: dict) -> None:
    """Apply the stage's optional ``physics`` block. Absent keys keep the task default.

    Keys: ``rate_hz`` [Hz], ``solver_position_iterations`` for the robot,
    ``enhanced_determinism``, and ``gpu_max_rigid_patch_count`` for stages with many envs.
    """
    if "gpu_max_rigid_patch_count" in physics:
        cfg.sim.physics.isaacsim_physx.gpu_max_rigid_patch_count = int(physics["gpu_max_rigid_patch_count"])
    if "rate_hz" in physics:
        set_physics_rate(cfg, int(physics["rate_hz"]))
    if "solver_position_iterations" in physics:
        cfg.scene.robot = with_position_iterations(cfg.scene.robot, int(physics["solver_position_iterations"]))
    if "enhanced_determinism" in physics:
        cfg.sim.physics.isaacsim_physx.enable_enhanced_determinism = bool(physics["enhanced_determinism"])


def play_cfg(task_id: str) -> TabletopBenchmarkEnvCfg:
    """Cfg for a local play gym id. The eval worker uses :func:`build_cfg` instead."""
    cfg = TabletopBenchmarkEnvCfg()
    cfg.task_id = task_id
    return cfg


def play_place_in_container() -> TabletopBenchmarkEnvCfg:
    return play_cfg("place_in_container")


def play_place_relative() -> TabletopBenchmarkEnvCfg:
    return play_cfg("place_relative")


def play_stack_on() -> TabletopBenchmarkEnvCfg:
    return play_cfg("stack_on")


def play_sort_by_attribute() -> TabletopBenchmarkEnvCfg:
    return play_cfg("sort_by_attribute")


def play_multi_place() -> TabletopBenchmarkEnvCfg:
    return play_cfg("multi_place")


def _choice(payload: dict, key: str) -> str | None:
    """Scene override for a stage list. Absent means the stage default."""
    value = payload.get(key)
    return value if isinstance(value, str) else None


def _bundle_file(bundle: Path, relative: str) -> str:
    """Absolute path of one file inside a downloaded benchmark."""
    path = bundle / relative
    if not path.is_file():
        raise FileNotFoundError(f"benchmark is missing {relative} under {bundle}")
    return str(path)


def _apply_benchmark_assets(cfg: TabletopBenchmarkEnvCfg) -> None:
    """Point the office, table, and objects at whatever the benchmark declares."""
    if not cfg.bundle_path:
        return
    bundle = Path(cfg.bundle_path)
    stage = read_stage(bundle)
    scene_file = bundle / "scenes" / f"{cfg.task_id}-{cfg.scene_id}.json"
    if not scene_file.is_file():
        raise FileNotFoundError(f"benchmark has no scene {cfg.task_id}-{cfg.scene_id} under {bundle}")
    payload = json.loads(scene_file.read_text(encoding="utf-8"))
    _apply_benchmark_physics(cfg, dict(stage.get("physics") or {}))
    background = resolve_background(stage, _choice(payload, "background"))
    cfg.scene.background.spawn = cfg.scene.background.spawn.replace(
        texture_file=_bundle_file(bundle, background["texture"]),
        intensity=float(background.get("intensity", 500.0)),
        visible_in_primary_ray=bool(background.get("visible_in_primary_ray", True)),
        texture_format=str(background.get("texture_format", "latlong")),
    )
    fixture = stage["fixture"]
    cfg.scene.table.spawn = cfg.scene.table.spawn.replace(usd_path=_bundle_file(bundle, fixture["usd"]))
    cfg.scene.table.init_state = cfg.scene.table.init_state.replace(
        pos=tuple(fixture.get("pos", (0.0, 0.0, 0.0))),
        rot=xyzw_from_wxyz(tuple(fixture.get("rot", (1.0, 0.0, 0.0, 0.0)))),
    )
    surface = resolve_surface(stage, _choice(payload, "surface"))
    pos, rot = surface_pose(surface, payload)
    cfg.scene.surface.spawn = cfg.scene.surface.spawn.replace(usd_path=_bundle_file(bundle, surface["usd"]))
    cfg.scene.surface.init_state = cfg.scene.surface.init_state.replace(pos=pos, rot=xyzw_from_wxyz(rot))
    if "light" in stage:
        light = stage["light"]
        cfg.scene.light.spawn = cfg.scene.light.spawn.replace(intensity=float(light["intensity"]))
        cfg.scene.light.init_state = cfg.scene.light.init_state.replace(pos=tuple(light["pos"]))
    if "ground" in stage:
        ground = stage["ground"]
        cfg.scene.plane.spawn = cfg.scene.plane.spawn.replace(visible=bool(ground.get("visible", False)))
        cfg.scene.plane.init_state = cfg.scene.plane.init_state.replace(pos=tuple(ground["pos"]))
    if cfg.mixed_scenes:
        first = (cfg.task_id, cfg.scene_id)
        payloads = [payload if pair == first else _scene_payload(bundle, *pair) for pair in cfg.mixed_scenes]
        _mount_mixed(cfg, bundle, stage, payloads)
    else:
        _mount_objects(cfg, bundle, payload["placements"])


def _scene_payload(bundle: Path, task_id: str, scene_id: str) -> dict:
    scene_file = bundle / "scenes" / f"{task_id}-{scene_id}.json"
    if not scene_file.is_file():
        raise FileNotFoundError(f"benchmark has no scene {task_id}-{scene_id} under {bundle}")
    return json.loads(scene_file.read_text(encoding="utf-8"))


def _tune_spawn(spawn: sim_utils.UsdFileCfg, scale, kinematic: bool) -> sim_utils.UsdFileCfg:
    """Apply a placement's instance scale and kinematic flag to an object spawner."""
    # An instance scale shrinks an asset that is larger than the table. size_m is already
    # that scaled extent, so the score box matches the mesh that is spawned.
    if scale:
        spawn = spawn.replace(scale=tuple(float(value) for value in scale))
    # A shelf is a fixed collider in the source scene. A dynamic copy falls into the table.
    if kinematic:
        spawn.rigid_props = sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True)
    return spawn


def _mount_objects(cfg: TabletopBenchmarkEnvCfg, bundle: Path, placements: list[dict]) -> None:
    """Spawn one rigid body per placement and point both finger filters at those prims."""
    if not placements:
        raise ValueError(f"scene {cfg.task_id}-{cfg.scene_id} places no objects")
    slots: list[str] = []
    filters: list[str] = []
    for item in placements:
        if str(item.get("kind")) == "anchor":
            continue
        slot = str(item["slot"])
        if slot in slots:
            raise ValueError(f"scene {cfg.task_id}-{cfg.scene_id} places {slot} more than once")
        position = tuple(float(value) for value in item["position_m"])
        quat = tuple(float(value) for value in item.get("quat_wxyz", (1.0, 0.0, 0.0, 0.0)))
        entity = _object(slot, position, xyzw_from_wxyz(quat))
        entity.spawn = _tune_spawn(
            entity.spawn.replace(usd_path=_bundle_file(bundle, str(item["usd"]))),
            item.get("scale"),
            bool(item.get("kinematic")),
        )
        setattr(cfg.scene, slot, entity)
        slots.append(slot)
        filters.append("{ENV_REGEX_NS}/" + slot)
    cfg.scene.gripper_left.filter_prim_paths_expr = filters
    cfg.scene.gripper_right.filter_prim_paths_expr = filters
    cfg.object_slots = tuple(slots)


def _parked_spawn() -> sim_utils.CuboidCfg:
    """Stand-in for an unused position: a small kinematic cube that is parked out of the way."""
    return sim_utils.CuboidCfg(
        size=(0.02, 0.02, 0.02),
        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
    )


def _variant_spawn(bundle: Path, variant: ObjectVariant | None):
    if variant is None:
        return _parked_spawn()
    spawn = sim_utils.UsdFileCfg(
        usd_path=_bundle_file(bundle, variant.usd),
        activate_contact_sensors=True,
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
    )
    return _tune_spawn(spawn, variant.scale, variant.kinematic)


def _mount_mixed(cfg: TabletopBenchmarkEnvCfg, bundle: Path, stage: dict, payloads: list[dict]) -> None:
    """Mount one body per object position and one for the work surface, each with a mesh per env.

    The finger filters name the positions, so every env has the same filter count. The cloner
    gets one world per env from :class:`~franka_tabletop.core.clone.planned_worlds`.
    """
    plan = plan_mixed(payloads, stage)
    for index, name in enumerate(plan.entities):
        position, quat = plan.start_poses[index]
        setattr(
            cfg.scene,
            name,
            RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                # The wrapper replaces each variant's contact-sensor flag, so it is set here.
                spawn=sim_utils.MultiAssetSpawnerCfg(
                    assets_cfg=[_variant_spawn(bundle, variant) for variant in plan.variants[index]],
                    activate_contact_sensors=True,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=xyzw_from_wxyz(quat)),
            ),
        )
    surfaces = [resolve_surface(stage, name) for name in plan.surfaces]
    position, quat = plan.surface_poses[0]
    # The surface is a kinematic body, so each env can move it to its own scene's pose.
    setattr(
        cfg.scene,
        SURFACE_ENTITY,
        RigidObjectCfg(
            prim_path=cfg.scene.surface.prim_path,
            spawn=sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=[
                    cfg.scene.surface.spawn.replace(usd_path=_bundle_file(bundle, str(record["usd"])))
                    for record in surfaces
                ],
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=xyzw_from_wxyz(quat)),
        ),
    )
    filters = ["{ENV_REGEX_NS}/" + name for name in plan.entities]
    cfg.scene.gripper_left.filter_prim_paths_expr = filters
    cfg.scene.gripper_right.filter_prim_paths_expr = filters
    cfg.object_slots = plan.entities
    cfg.mixed_plan = plan
    # Every env owns its meshes, so physics cannot copy env 0. The claim only keeps the cloner on
    # its heterogeneous path; planned_worlds supplies the actual worlds.
    cfg.scene.replicate_physics = False
    cfg.scene.clone_cfg = CloneCfg(clone_combinations=[InclusionSet(assets=[*plan.entities, SURFACE_ENTITY])])


def _drop_unused_cameras(cfg: TabletopBenchmarkEnvCfg, observation: dict | None) -> None:
    """Do not spawn, render, or observe a camera that nothing reads.

    A recording run keeps the front, left exterior, and wrist cameras. The front camera
    renders one cell of the shard grid. The left and wrist cameras stay at full size,
    because the brain reads them.
    """
    if observation is not None:
        # A brain reads images through pack_numpy on capture steps only. The observation
        # manager would otherwise fetch every camera after every control step.
        for name in OBSERVATION_CAMERAS:
            setattr(cfg.observations.policy, name, None)
    recording = video_directory() is not None
    if not recording:
        cfg.scene.cam_front = None
    for name in unused_cameras(observation):
        if recording and name in ("cam_exterior_left", "cam_wrist"):
            continue
        setattr(cfg.scene, name, None)
    if recording:
        cfg.scene.cam_front.height = TILE_HW[0]
        cfg.scene.cam_front.width = TILE_HW[1]


def build_cfg(
    task_id: str = "place_in_container",
    scene: str = "kitchen_counter-0",
    variant: str = "nominal",
    env_id: str | None = None,
    num_envs: int = 1,
    composer: str = "kitchen_counter",
    pose_jitter_m: float = 0.0,
    observation: dict | None = None,
    scenes: list[tuple[str, str]] | None = None,
    bundle_path: str | Path | None = None,
) -> TabletopBenchmarkEnvCfg:
    """Build a cfg for one group of episodes that build the same stage.

    The first job names the scene. Every other job in the group must have the same
    background, surface, and object meshes. Their poses, instructions, and success
    tests are applied per environment at reset. With ``scenes`` the jobs may differ in
    surface and objects too, and only the background must match.

    Args:
        scenes: (task, scene) of each env, in env order. Give it to put different objects
            in different envs of one stage; it must list ``num_envs`` pairs, and ``task_id``
            and ``scene`` must name the first. ``None`` builds copies of one scene.
        observation: The brain's ``observation`` block: ``cameras`` and ``image_hw``.
            ``None`` renders and sends every observation camera at full size and keeps
            the images in the ``policy`` observation group for an in-process policy.
        bundle_path: Extracted bundle directory. When given (or ``$ISAAC_EVAL_BUNDLE_DIR``
            is set, or ``~/.unidroid/bench-cache/<env_id>/bundle`` exists), assets are read
            directly without invoking ``nepher.load_env``.
    """
    cfg = TabletopBenchmarkEnvCfg()
    cfg.observation_cameras = requested_cameras(observation)
    cfg.observation_image_hw = requested_image_hw(observation)
    _drop_unused_cameras(cfg, observation)
    cfg.task_id = task_id
    cfg.scene_id = scene
    cfg.variant = variant
    cfg.env_id = env_id
    cfg.composer = composer
    cfg.pose_jitter_m = 0.10 if variant == "jitter10" else pose_jitter_m
    cfg.scene.num_envs = num_envs
    if scenes:
        pairs = tuple((str(task), str(name)) for task, name in scenes)
        if len(pairs) != num_envs or pairs[0] != (task_id, scene):
            raise ValueError("scenes must list one (task, scene) per env, starting with task_id and scene")
        cfg.mixed_scenes = pairs
    cfg.bundle_path = resolve_bundle_path(env_id, bundle_path)
    _apply_benchmark_assets(cfg)
    return cfg
