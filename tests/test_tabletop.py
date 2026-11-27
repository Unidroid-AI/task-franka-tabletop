# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "franka_tabletop"))
sys.path.insert(0, str(ROOT / "examples" / "gr00t_n17_zero_shot" / "brain"))

from adapt import relative_chunk_to_absolute, resize_hwc
from franka_tabletop.geometry import SURFACE_SLOT, from_box, from_points, read_local_points
from franka_tabletop.predicates import task_progress
from franka_tabletop.scenes.bundle import (
    read_catalog,
    read_stage,
    resolve_background,
    resolve_surface,
    surface_pose,
    xyzw_from_wxyz,
)
from franka_tabletop.core.clips import EpisodeClips, tile_grid
from franka_tabletop.scenes.composer import load_scene, world_position
from franka_tabletop.spec import spec_from_success
from franka_tabletop.success import Contacts, note_placements, required_pairs, task_satisfied


# The fixtures use dyadic lengths [m], so every boundary below is exact in floating point.
def _body(position, size, quat=(1.0, 0.0, 0.0, 0.0), anchor=False) -> dict:
    pose = {
        "pos": np.asarray([position], dtype=np.float64),
        "quat": np.asarray([quat], dtype=np.float64),
        "geometry": from_box(size),
    }
    # A pose starts at rest: its reset pose is the pose itself.
    pose["initial_pos"], pose["initial_quat"] = pose["pos"], pose["quat"]
    if anchor:
        pose["anchor"] = True
    return pose


def _met(block, poses, contacts=None, first_placed=None) -> bool:
    spec = spec_from_success("t", block, 40.0, {name: name for name in poses})
    return task_satisfied(spec, poses, contacts or Contacts(), first_placed or {})


CUBE = (0.0625,) * 3
# Open-top bin: x and y in [0.375, 0.625] m, floor at z = 0.
BIN = _body((0.5, 0.0, 0.0625), (0.25, 0.25, 0.125))
IN_BIN = {"op": "in_container", "object": "cube", "container": "bin"}


def _scene_poses(scene) -> dict:
    """Authored poses of a scene on box geometry. An anchor stands for the work surface."""
    poses = {}
    for placement in scene.placements:
        poses[placement.slot] = {
            "pos": np.asarray([placement.position_m], dtype=np.float64),
            "quat": np.asarray([placement.quat_wxyz], dtype=np.float64),
            "geometry": from_box(placement.asset.size_m),
        }
        poses[placement.slot]["initial_pos"] = poses[placement.slot]["pos"]
        poses[placement.slot]["initial_quat"] = poses[placement.slot]["quat"]
        if placement.asset.kind == "anchor":
            poses[placement.slot]["anchor"] = True
    return poses


def _starts_solved(scene) -> bool:
    """Whether the authored layout already satisfies the task, with every pair resting on its partner."""
    anchors = {placement.slot for placement in scene.placements if placement.asset.kind == "anchor"}
    pairs = required_pairs(scene.spec, anchors)
    contacts = Contacts(
        touching={placement.slot: False for placement in scene.placements},
        touch={pair: True for pair in pairs},
        force={pair: np.array([0.0, 0.0, 1.0]) for pair in pairs},
    )
    poses = _scene_poses(scene)
    poses[SURFACE_SLOT] = _body((0.5, 0.0, -0.0625), (1.0, 1.0, 0.125), anchor=True)
    return task_satisfied(scene.spec, poses, contacts, {}, side=scene.side)


def test_in_container_uses_the_hull_center_and_an_open_top():
    def at(x, z=0.09375):
        return {"cube": _body((x, 0.0, z), CUBE), "bin": BIN}

    assert _met(IN_BIN, at(0.5))
    # The hull center on the wall plane still counts. A hair beyond it does not.
    assert _met(IN_BIN, at(0.625))
    assert not _met(IN_BIN, at(0.640625))
    # The open top reaches up without bound. Below the floor is outside.
    assert _met(IN_BIN, at(0.5, z=1.0))
    assert not _met(IN_BIN, at(0.5, z=-0.03125))
    assert not _met({**IN_BIN, "op": "enclosed"}, at(0.5, z=1.0))


def _tilt_from_vertical(degrees: float) -> tuple[float, float, float, float]:
    """wxyz rotation about y that sends local +x to ``degrees`` off world up, toward +x."""
    half = 0.5 * np.deg2rad(degrees - 90.0)
    return (float(np.cos(half)), 0.0, float(np.sin(half)), 0.0)


def test_a_seated_rod_counts_when_its_center_is_outside_the_wall():
    # 0.5 m rod, 0.25 m opening. The center is past the wall and 0.2 m of the shaft is inside.
    rod = (0.5, 0.03125, 0.03125)
    poses = {"rod": _body((0.16, 0.0, 0.0), rod), "bin": _body((0.0, 0.0, 0.0), (0.25, 0.25, 0.25))}
    block = {"op": "in_container", "object": "rod", "container": "bin"}
    assert _met(block, poses)
    assert not _met({**block, "op": "enclosed"}, poses)
    assert not _met({**block, "op": "outside"}, poses)


def test_a_short_tip_and_a_shaft_above_the_lid_do_not_count():
    rod = (0.5, 0.03125, 0.03125)
    holder = _body((0.0, 0.0, 0.0), (0.25, 0.25, 0.25))
    block = {"op": "in_container", "object": "rod", "container": "bin"}
    tip = {"rod": _body((0.34, 0.0, 0.0), rod), "bin": holder}
    above = {"rod": _body((0.16, 0.0, 0.25), rod), "bin": holder}
    assert not _met(block, tip)
    assert not _met(block, above)


def test_a_slender_center_above_the_rim_does_not_use_the_open_column():
    from franka_tabletop.geometry import centerline_inside_length, rotation

    rod = (0.5, 0.03125, 0.03125)
    quat = _tilt_from_vertical(0.0)
    assert np.allclose(rotation(quat)[:, 0], (0.0, 0.0, 1.0), atol=1e-6)
    poses = {"rod": _body((0.0, 0.0, 0.4), rod, quat), "bin": _body((0.0, 0.0, 0.0), (0.25, 0.25, 0.25))}
    block = {"op": "in_container", "object": "rod", "container": "bin"}
    seated = centerline_inside_length(
        poses["rod"]["geometry"],
        poses["rod"]["pos"],
        poses["rod"]["quat"],
        poses["bin"]["pos"],
        poses["bin"]["quat"],
        poses["bin"]["geometry"].planes_full,
    )
    assert seated == 0.0
    assert not _met(block, poses)
    assert not _met({**block, "op": "enclosed"}, poses)


def test_a_slender_center_inside_a_short_cavity_is_not_enough():
    # The center sits in the cavity, but the cavity is shorter than 35% of the rod.
    rod = (1.0, 0.02, 0.02)
    poses = {"rod": _body((0.0, 0.0, 0.0), rod), "bin": _body((0.0, 0.0, 0.0), (0.25, 0.25, 0.25))}
    assert not _met({"op": "in_container", "object": "rod", "container": "bin"}, poses)


def test_a_stubby_object_keeps_the_hull_center_test():
    holder = _body((0.0, 0.0, 0.0), (0.25, 0.25, 0.25))
    block = {"op": "in_container", "object": "item", "container": "bin"}
    # Longest edge is 2.4 times the next, under the 3x slender cut, and past the wall.
    stubby = {"item": _body((0.2, 0.0, 0.0), (0.3, 0.125, 0.125)), "bin": holder}
    assert not _met(block, stubby)
    assert _met(block, {"item": _body((0.0, 0.0, 0.0), (0.3, 0.125, 0.125)), "bin": holder})


def test_inside_requires_every_hull_vertex_in_the_closed_hull():
    holder = _body((0.0, 0.0, 0.0), (0.25, 0.25, 0.25))
    block = {"op": "inside", "object": "carton", "container": "bin"}
    # The carton fills the cavity. Corners on the faces still count.
    assert _met(block, {"carton": _body((0.0, 0.0, 0.0), (0.25, 0.25, 0.25)), "bin": holder})
    # The center is inside, but the top corners are past the lid.
    poking = {"carton": _body((0.0, 0.0, 0.1), (0.1, 0.1, 0.1)), "bin": holder}
    assert not _met(block, poking)
    assert _met({**block, "op": "enclosed"}, poking)
    assert not _met({**block, "tolerance": 0.5}, poking)
    # Same crate orientation as recycle_cartons_vertical_crate: the mouth faces world -X.
    crate = _body((0.0, 0.0, 0.0), (0.2, 0.2, 0.2), (0.5, -0.5, -0.5, 0.5))
    seated = {"carton": _body((-0.02, 0.0, 0.0), (0.04, 0.04, 0.04)), "bin": crate}
    proud = {"carton": _body((-0.15, 0.0, 0.0), (0.04, 0.04, 0.04)), "bin": crate}
    assert _met(block, seated)
    assert not _met(block, proud)


def test_in_container_follows_the_container_yaw():
    """A quarter turn swaps the box axes. The long local side is then the scene y axis."""
    yaw = (np.cos(np.pi / 4.0), 0.0, 0.0, np.sin(np.pi / 4.0))
    tray = _body((0.0, 0.0, 0.0), (0.5, 0.25, 0.125), yaw)
    inside = {"cube": _body((0.0, 0.1875, 0.0), CUBE), "bin": tray}
    assert _met(IN_BIN, inside)
    turned = {"cube": inside["cube"], "bin": _body((0.0, 0.0, 0.0), (0.5, 0.25, 0.125))}
    assert not _met(IN_BIN, turned)


def test_success_needs_the_release_the_stage_asks_for():
    poses = {"cube": _body((0.5, 0.0, 0.09375), CUBE), "bin": BIN}
    detach = {**IN_BIN, "require_gripper_detached": True}
    assert _met(detach, poses, Contacts(touching={"cube": False}))
    assert not _met(detach, poses, Contacts(touching={"cube": True}))
    # A missing finger reading cannot pass a release check.
    assert not _met(detach, poses)
    assert _met(IN_BIN, poses, Contacts(touching={"cube": True}))


def test_grey_bin_spawn_is_outside_the_mesh_box():
    """The bin extent is the mesh, so resting on the table is not already inside it."""
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    root = Path(__file__).resolve().parents[2] / "envhub" / "environments"
    bundle = root / "franka-tabletop-v0"
    if not bundle.is_dir():
        pytest.skip("envhub bundle is not next to this package")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    stage = Usd.Stage.Open(str(bundle / "assets" / "fixtures" / "grey_bin.usd"))
    bounds = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    extent = np.array(bounds.GetMax()) - np.array(bounds.GetMin())
    np.testing.assert_allclose(extent, (0.42, 0.28, 0.105), atol=1e-3)
    for task_id, scene_id in (
        ("bowl_in_bin", "bin_mug_mustard_marker_bowl"),
        ("bigger_object", "butter_raisin_box_grey_bin"),
    ):
        scene = load_scene(bundle, task_id, scene_id)
        if scene is None:
            continue
        assert not _starts_solved(scene)


def test_catalog_sizes_match_the_object_prim():
    """A size is the object's own extent. A ground plane in the file must not set it."""
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    root = Path(__file__).resolve().parents[2] / "envhub" / "environments"
    if not root.is_dir():
        pytest.skip("envhub bundle is not next to this package")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    for env_id in ("franka-tabletop-v0", "franka-tabletop-v1", "franka-tabletop-v2", "franka-tabletop-v10"):
        bundle = root / env_id
        if not bundle.is_dir():
            continue
        catalog = json.loads((bundle / "assets" / "catalog.json").read_text(encoding="utf-8"))
        for record in catalog["objects"]:
            stage = Usd.Stage.Open(str(bundle / record["usd"]))
            bounds = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
            extent = np.array(bounds.GetMax()) - np.array(bounds.GetMin())
            recorded = np.asarray(record["size_m"], dtype=np.float64)
            np.testing.assert_allclose(recorded, extent, rtol=0.5, atol=0.01, err_msg=record["name"])


def test_published_tasks_do_not_start_inside_the_container():
    """Grabbing an object and letting go is not success when it never entered the container."""
    root = Path(__file__).resolve().parents[2] / "envhub" / "environments"
    if not root.is_dir():
        pytest.skip("envhub bundle is not next to this package")
    for env_id in (
        "franka-tabletop-v0",
        "franka-tabletop-v1",
        "franka-tabletop-v2",
        "franka-tabletop-v8",
        "franka-tabletop-v9",
        "franka-tabletop-v10",
    ):
        bundle = root / env_id
        if not bundle.is_dir():
            continue
        for scene_path in sorted((bundle / "scenes").glob("*.json")):
            payload = json.loads(scene_path.read_text(encoding="utf-8"))
            success = payload.get("success") or {}
            published = env_id in ("franka-tabletop-v10", "franka-tabletop-v9", "franka-tabletop-v8")
            if not published and success.get("kind") not in ("item_in_container", "both_in_container", "sorted_one"):
                continue
            if published and not success:
                continue
            scene = load_scene(bundle, payload["task_id"], payload["scene_id"])
            assert not _starts_solved(scene), scene_path.name


def test_a_pickup_counts_only_while_a_finger_holds_it_above_the_work_surface():
    """Picking an object up is not placing it. The credit ends when the gripper lets go."""
    surface = _body((0.5, 0.0, -0.0625), (1.0, 1.0, 0.125), anchor=True)
    block = {"op": "picked_up", "object": "cube", "distance": 0.25}

    def met(z, grabbed=True):
        poses = {"cube": _body((0.5, 0.0, z), CUBE), SURFACE_SLOT: surface}
        return _met(block, poses, Contacts(touching={"cube": grabbed}))

    assert met(0.3125)
    # The lowest corner may sit exactly at the distance.
    assert met(0.28125)
    assert not met(0.25)
    assert not met(0.3125, grabbed=False)


def test_a_touched_object_resting_on_a_published_table_is_not_picked_up():
    """The oak, maple, and bamboo tables are cubes and cylinders, not meshes. Their top must still be read."""
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom

    bundle = Path(__file__).resolve().parents[2] / "envhub" / "environments" / "franka-tabletop-v13"
    if not bundle.is_dir():
        pytest.skip("envhub bundle is not next to this package")
    block = {"op": "picked_up", "object": "cube", "distance": 0.05}
    for record in json.loads((bundle / "assets" / "stage.json").read_text(encoding="utf-8"))["surfaces"]:
        stage = Usd.Stage.CreateInMemory()
        table = stage.DefinePrim("/table", "Xform")
        table.GetReferences().AddReference(str(bundle / record["usd"]))
        UsdGeom.Xformable(table).AddTranslateOp().Set(Gf.Vec3d(*record["pos"]))
        surface = _body(record["pos"], (1.0, 1.0, 1.0), record["rot"], anchor=True)
        surface["geometry"] = from_points(read_local_points(stage, "/table"))
        top = float(record["table_z_m"])

        def met(z):
            poses = {"cube": _body((record["pos"][0], 0.0, z), CUBE), SURFACE_SLOT: surface}
            return _met(block, poses, Contacts(touching={"cube": True}))

        assert not met(top + CUBE[2] / 2.0), record["name"]
        assert met(top + CUBE[2] / 2.0 + 0.0625), record["name"]


def test_v0_and_v1_split_v13_into_disjoint_background_halves():
    """v0 and v1 are disjoint 60-task slices of v13.

    Each slice sets 30 scenes to home_office and 30 to brown_photostudio. Poses,
    instructions, and success checks stay the ones in v13. The meshes live in the slice.
    """
    root = Path(__file__).resolve().parents[2] / "envhub" / "environments"
    source = root / "franka-tabletop-v13"
    if not source.is_dir():
        pytest.skip("franka-tabletop-v13 is not next to this package")
    chosen: list[set[str]] = []
    for env_id in ("franka-tabletop-v0", "franka-tabletop-v1"):
        bundle = root / env_id
        if not bundle.is_dir():
            pytest.skip(f"{env_id} is not next to this package")
        text = (bundle / "benchmark.yaml").read_text(encoding="utf-8")
        ids = [line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("- task_id:")]
        assert len(ids) == 60
        assert len(set(ids)) == 60
        chosen.append(set(ids))
        assets = bundle / "assets"
        assert assets.is_dir()
        assert not assets.is_symlink()
        assert (assets / "stage.json").stat().st_ino != (source / "assets" / "stage.json").stat().st_ino
        stage = json.loads((assets / "stage.json").read_text(encoding="utf-8"))
        assert [record["name"] for record in stage["backgrounds"]] == ["home_office", "brown_photostudio"]
        backgrounds = []
        referenced = [stage["fixture"]["usd"]]
        referenced.extend(record["texture"] for record in stage["backgrounds"])
        referenced.extend(record["usd"] for record in stage["surfaces"])
        for task_id in ids:
            copied = json.loads(next((bundle / "scenes").glob(f"{task_id}-*.json")).read_text(encoding="utf-8"))
            original = json.loads(next((source / "scenes").glob(f"{task_id}-*.json")).read_text(encoding="utf-8"))
            assert copied["background"] in ("home_office", "brown_photostudio")
            backgrounds.append(copied["background"])
            original["background"] = copied["background"]
            assert copied == original
            scene = load_scene(bundle, task_id, copied["scene_id"])
            assert scene is not None and scene.spec is not None
            for placement in scene.placements:
                referenced.append(placement.asset.bundle_usd())
        assert backgrounds.count("home_office") == 30
        assert backgrounds.count("brown_photostudio") == 30
        for relative in referenced:
            assert (bundle / relative).is_file(), relative
    assert chosen[0].isdisjoint(chosen[1])
    assert len(chosen[0] | chosen[1]) == 120


def test_v8_is_the_sixteen_scored_successes():
    """v8 is the sixteen v10 episodes the previous scorer marked successful."""
    bundle = Path(__file__).resolve().parents[2] / "envhub" / "environments" / "franka-tabletop-v8"
    if not bundle.is_dir():
        pytest.skip("franka-tabletop-v8 is not next to this package")
    text = (bundle / "benchmark.yaml").read_text(encoding="utf-8")
    ids = [line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("- task_id:")]
    expected = (
        ("unstack_rubiks_cube", "mug_banana_ketchup_bowl_rubiks3_bin"),
        ("keyboard_out_of_bin", "workdesk_bin"),
        ("rubiks_cube_or_banana", "rubiks_cube_banana_bowl"),
        ("banana_on_plate", "bagel_plate_banana_bowl"),
        ("bananas_in_bin_three", "bananas_5_grey_bin"),
        ("mustard_above_raisin", "mustard_raisin_box"),
        ("one_bottle_in_square_pail", "shelf_with_cleaning_products"),
        ("tools_picking_hammer", "tools_picking"),
        ("cooking_orange_object", "cooking_table"),
        ("rubiks_cube_and_banana", "rubiks_cube_banana_bowl"),
        ("wood_spatula_to_bowl", "fruits_in_basket"),
        ("banana_then_rubiks_cube", "rubiks_cube_banana_bowl"),
        ("banana_in_bowl", "banana_bowl"),
        ("rubiks_cube", "rubiks_cube_bowl"),
        ("bigger_object", "butter_raisin_box_grey_bin"),
        ("bananas_in_bin_one_more", "bananas_5_grey_bin"),
    )
    assert ids == [task_id for task_id, _scene in expected]
    scenes = sorted(path.name for path in (bundle / "scenes").glob("*.json"))
    assert scenes == sorted(f"{task_id}-{scene_id}.json" for task_id, scene_id in expected)
    assets = bundle / "assets"
    v10_assets = bundle.parent / "franka-tabletop-v10" / "assets"
    assert assets.is_dir()
    assert not assets.is_symlink()
    assert (assets / "stage.json").stat().st_ino != (v10_assets / "stage.json").stat().st_ino
    stage = json.loads((assets / "stage.json").read_text(encoding="utf-8"))
    referenced = [stage["fixture"]["usd"]]
    referenced.extend(record["texture"] for record in stage["backgrounds"])
    referenced.extend(record["usd"] for record in stage["surfaces"])
    for task_id, scene_id in expected:
        scene = load_scene(bundle, task_id, scene_id)
        assert scene is not None
        assert scene.spec is not None
        assert scene.instruction
        for placement in scene.placements:
            referenced.append(placement.asset.bundle_usd())
    for relative in referenced:
        assert (bundle / relative).is_file(), relative


def test_v9_rechecks_only_the_four_false_successes():
    """v9 is the four tasks whose scored clips finished while the object was still in place."""
    bundle = Path(__file__).resolve().parents[2] / "envhub" / "environments" / "franka-tabletop-v9"
    if not bundle.is_dir():
        pytest.skip("franka-tabletop-v9 is not next to this package")
    text = (bundle / "benchmark.yaml").read_text(encoding="utf-8")
    ids = [line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("- task_id:")]
    expected = (
        ("unstack_rubiks_cube", "mug_banana_ketchup_bowl_rubiks3_bin"),
        ("keyboard_out_of_bin", "workdesk_bin"),
        ("tools_picking_hammer", "tools_picking"),
        ("cooking_orange_object", "cooking_table"),
    )
    assert ids == [task_id for task_id, _scene in expected]
    scenes = sorted(path.name for path in (bundle / "scenes").glob("*.json"))
    assert scenes == sorted(f"{task_id}-{scene_id}.json" for task_id, scene_id in expected)
    assets = bundle / "assets"
    v10_assets = bundle.parent / "franka-tabletop-v10" / "assets"
    assert assets.is_dir()
    assert not assets.is_symlink()
    assert (assets / "stage.json").stat().st_ino != (v10_assets / "stage.json").stat().st_ino
    stage = json.loads((assets / "stage.json").read_text(encoding="utf-8"))
    referenced = [stage["fixture"]["usd"]]
    referenced.extend(record["texture"] for record in stage["backgrounds"])
    referenced.extend(record["usd"] for record in stage["surfaces"])
    for task_id, scene_id in expected:
        scene = load_scene(bundle, task_id, scene_id)
        assert scene is not None
        assert scene.spec is not None
        assert scene.instruction
        for placement in scene.placements:
            referenced.append(placement.asset.bundle_usd())
    for relative in referenced:
        assert (bundle / relative).is_file(), relative


def test_a_touch_does_not_finish_a_place_or_an_unstack():
    """The scored clips left the object where it started. Those poses do not succeed."""
    root = Path(__file__).resolve().parents[2] / "envhub" / "environments" / "franka-tabletop-v10"
    if not root.is_dir():
        pytest.skip("franka-tabletop-v10 is not next to this package")
    for task_id, scene_id in (
        ("keyboard_out_of_bin", "workdesk_bin"),
        ("unstack_rubiks_cube", "mug_banana_ketchup_bowl_rubiks3_bin"),
        ("tools_picking_hammer", "tools_picking"),
    ):
        assert not _starts_solved(load_scene(root, task_id, scene_id)), task_id


def test_a_sideways_pickup_uses_the_world_bottom():
    """Raising the center of a rotated cup is not a lift while a corner is still down."""
    surface = _body((0.5, 0.0, -0.05), (1.0, 1.0, 0.1), anchor=True)
    block = {"op": "picked_up", "object": "cup", "distance": 0.053}
    # A quarter turn about y stands the 0.1734 m long side on end.
    quat = (0.7071, 0.0, 0.7071, 0.0)

    def met(z):
        poses = {"cup": _body((0.36, -0.26, z), (0.1734, 0.0952, 0.052), quat), SURFACE_SLOT: surface}
        return _met(block, poses, Contacts(touching={"cup": True}))

    # The lowest corner is z - 0.0867 m.
    assert not met(0.09)
    assert met(0.16)


def test_resize_and_absolute_actions():
    image = np.zeros((40, 80, 3), dtype=np.uint8)
    image[0, 0] = 255
    resized = resize_hwc(image, 20, 40)
    assert resized.shape == (20, 40, 3)
    joints = np.zeros((2, 7), dtype=np.float32)
    chunk = np.ones((2, 8, 8), dtype=np.float32)
    absolute = relative_chunk_to_absolute(joints, chunk)
    assert absolute.shape == (2, 8, 8)
    assert np.allclose(absolute[0, 0, :7], 1.0)


def test_load_scene_uses_the_benchmark_record(tmp_path: Path):
    scene_dir = tmp_path / "scenes"
    scene_dir.mkdir()
    payload = {
        "side": "right",
        "success": {
            "op": "in_container",
            "objects": ["custom_widget"],
            "container": "custom_bin",
        },
        "placements": [
            {
                "slot": "item_a",
                "name": "custom_widget",
                "usd": "assets/objects/custom/widget.usd",
                "size_m": [0.2, 0.1, 0.05],
                "position_m": [0.4, 0.0, 0.025],
            },
            {
                "slot": "container",
                "name": "custom_bin",
                "usd": "assets/objects/custom/bin.usd",
                "size_m": [0.3, 0.3, 0.1],
                "position_m": [0.5, 0.2, 0.05],
            },
        ],
    }
    (scene_dir / "place_in_container-kitchen_counter-0.json").write_text(json.dumps(payload), encoding="utf-8")
    scene = load_scene(tmp_path, "place_in_container", "kitchen_counter-0")
    assert scene is not None
    item = scene.by_slot()["item_a"]
    assert scene.names()["item_a"] == "custom widget"
    assert item.asset.size_m == (0.2, 0.1, 0.05)
    assert item.asset.bundle_usd() == "assets/objects/custom/widget.usd"
    assert scene.side == "right"


def test_scene_pose_includes_the_env_origin():
    """A 4-env shard sits on a centered 2.5 m grid. No table is at the world origin."""
    origins = (
        (1.25, -1.25, 0.0),
        (1.25, 1.25, 0.0),
        (-1.25, -1.25, 0.0),
        (-1.25, 1.25, 0.0),
    )
    banana = (0.509727, -0.115807, 0.021177)
    for origin in origins:
        placed = world_position(banana, origin)
        assert placed == pytest.approx(
            (banana[0] + origin[0], banana[1] + origin[1], banana[2] + origin[2])
        )
        assert abs(placed[0] - banana[0]) > 1.0
        assert abs(placed[1] - banana[1]) > 1.0


def test_a_shard_writes_one_grid_for_front_left_and_wrist():
    """Every control step appends one 4x4 frame to each camera, named for the first job."""
    written = []

    def _write(directory, name, frame, fps):
        written.append((directory, name, frame.shape, fps))

    clips = EpisodeClips("/output/videos", _write, grid_hw=(8, 8))
    jobs = [type("Job", (), {"job_id": "abc", "episode_index": index})() for index in range(3)]
    clips.begin(jobs, "banana_in_bowl", "banana_bowl", "nominal", fps=15)
    frame = np.zeros((3, 2, 2, 3), dtype=np.uint8)
    for _ in range(4):
        clips.add("front", frame)
        clips.add("exterior_left", frame)
        clips.add("wrist", frame)
    clips.end(fps=15)

    names = [item[1] for item in written]
    assert names == ["front_abc.mp4", "exterior_left_abc.mp4", "wrist_abc.mp4"] * 4
    assert {item[0] for item in written} == {"/output/videos"}
    assert {item[2] for item in written} == {(8, 8, 3)}
    assert {item[3] for item in written} == {15}


def test_a_finished_environment_stays_in_the_grid():
    """The shard video keeps every cell. A finished environment is not dropped from later frames."""
    frames = []

    def _write(directory, name, frame, fps):
        del directory, fps
        if name.startswith("front"):
            frames.append(frame.copy())

    clips = EpisodeClips("/output/videos", _write, grid_hw=(8, 8))
    jobs = [type("Job", (), {"job_id": "abc", "episode_index": index})() for index in (0, 1)]
    clips.begin(jobs, "bowl_in_bin", "bin_mug_mustard_marker_bowl", "nominal")
    running = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    running[0] = 1
    running[1] = 2
    held = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    held[0] = 9
    held[1] = 3
    clips.add("front", running, np.array([True, True]))
    clips.add("front", held, np.array([False, True]))
    clips.end()

    assert len(frames) == 2
    assert np.all(frames[1][0:2, 0:2] == 9)
    assert np.all(frames[1][0:2, 2:4] == 3)


def test_authored_scene_uses_its_instruction_and_success(tmp_path: Path):
    scene_dir = tmp_path / "scenes"
    scene_dir.mkdir()
    payload = {
        "instruction": "Pick up the banana and place it in the bowl",
        "episode_length_s": 50,
        "side": "left",
        "success": {"op": "in_container", "object": "banana", "container": "bowl"},
        "placements": [
            {
                "slot": "item_a",
                "name": "banana",
                "usd": "assets/objects/ycb/banana.usd",
                "size_m": [0.1, 0.18, 0.04],
                "position_m": [0.51, -0.12, 0.02],
            },
            {
                "slot": "container",
                "name": "bowl",
                "usd": "assets/objects/ycb/bowl.usd",
                "size_m": [0.16, 0.16, 0.05],
                "position_m": [0.56, 0.15, 0.03],
            },
        ],
    }
    (scene_dir / "banana_in_bowl-banana_bowl.json").write_text(json.dumps(payload), encoding="utf-8")
    scene = load_scene(tmp_path, "banana_in_bowl", "banana_bowl")
    assert scene is not None
    assert scene.instruction == "Pick up the banana and place it in the bowl"
    assert scene.spec is not None
    assert scene.spec.success == "in_container"
    assert scene.spec.roles == {"item_a": "item_a"}
    assert (scene.spec.check.goals[0].slot, scene.spec.check.goals[0].target) == ("item_a", "container")
    stage_surface = {"pos": [0.0, 0.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0]}
    pose, _rot = surface_pose(stage_surface, {"surface_pose": {"pos": [0.55, 0.0, -0.35]}})
    assert pose == (0.55, 0.0, -0.35)


def _load_submission_adapt():
    path = ROOT.parent / "task-franka-tabletop-sample-submission" / "brain" / "adapt.py"
    if not path.is_file():
        pytest.skip("sample submission is not next to this package")
    spec = importlib.util.spec_from_file_location("submission_adapt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_joint_action_chunk_keeps_decoded_absolute_joints():
    """GR00T's decoded ``joint_position`` is absolute. The brain must not add the state again."""
    adapt = _load_submission_adapt()
    ready = np.array([[0.0, -0.628, 0.0, -2.513, 0.0, 1.885, 0.0]], dtype=np.float32)
    decoded = (ready + 0.02)[:, None, :].repeat(40, axis=1)
    gripper = np.full((1, 40, 1), 0.75, dtype=np.float32)
    chunk = adapt.joint_action_chunk(({"joint_position": decoded, "gripper_position": gripper}, {}), horizon=8)
    assert chunk.shape == (1, 8, 8)
    np.testing.assert_allclose(chunk[..., :7], decoded[:, :8], atol=1e-6)
    np.testing.assert_allclose(chunk[..., 7], 0.75)
    # The Franka elbow and wrist must stay inside their limits (panda_joint4, panda_joint6).
    assert (chunk[..., 3] > -3.0718).all() and (chunk[..., 5] < 3.7525).all()


def test_tool_offset_is_the_droid_eef_correction():
    """The observation's tool offset is the rotation RoboLab applies before building eef_9d."""
    scipy_rotation = pytest.importorskip("scipy.spatial.transform").Rotation
    source = (ROOT / "source" / "franka_tabletop" / "franka_tabletop" / "core" / "observations.py").read_text(
        encoding="utf-8"
    )
    value = next(line for line in source.splitlines() if line.startswith("_TOOL_OFFSET_XYZW"))
    offset = ast.literal_eval(value.split("=", 1)[1].strip())
    droid = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64)
    np.testing.assert_allclose(scipy_rotation.from_quat(offset).as_matrix(), droid, atol=1e-9)


def test_placed_objects_rest_on_the_composed_tabletop():
    """A scene root pose is added to the table file's ``table_01`` offset.

    RoboLab's rubiks and mug-bin scenes zero that child and store the root at the tabletop height.
    Copying the root alone lifts the collider 0.35 m through the arm and the objects.
    """
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom

    root = Path(__file__).resolve().parents[2] / "envhub" / "environments"
    if not root.is_dir():
        pytest.skip("envhub bundle is not next to this package")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    compensated = {
        "rubiks_cube_banana_bowl",
        "rubiks_cube_bowl",
        "rubiks_cube_banana_bowl_mug_bin",
        "mugs2_bananas2_ketchup_rubiks3_bin",
        "mug_banana_ketchup_bowl_rubiks3_bin",
    }
    for env_id in ("franka-tabletop-v2", "franka-tabletop-v10", "franka-tabletop-v1", "franka-tabletop-v0"):
        bundle = root / env_id
        if not bundle.is_dir():
            continue
        stage = json.loads((bundle / "assets" / "stage.json").read_text(encoding="utf-8"))
        surfaces = {record["name"]: record for record in stage["surfaces"]}
        for scene_path in sorted((bundle / "scenes").glob("*.json")):
            payload = json.loads(scene_path.read_text(encoding="utf-8"))
            if env_id not in ("franka-tabletop-v2", "franka-tabletop-v10") and payload.get("scene_id") not in compensated:
                continue
            surface = surfaces[payload["surface"]]
            table = Usd.Stage.Open(str(bundle / surface["usd"]))
            child = table.GetPrimAtPath("/Root/table_01")
            offset = UsdGeom.Xformable(child).GetLocalTransformation().ExtractTranslation()
            pose = payload["surface_pose"]["pos"]
            top_z = pose[2] + offset[2]
            near_x = pose[0] + offset[0]
            assert near_x > 0.0, scene_path
            for item in payload["placements"]:
                if item.get("kind") == "anchor":
                    continue
                mesh = Usd.Stage.Open(str(bundle / item["usd"]))
                bounds = cache.ComputeWorldBound(mesh.GetDefaultPrim()).ComputeAlignedRange()
                minimum, maximum = bounds.GetMin(), bounds.GetMax()
                scale = tuple(float(value) for value in item.get("scale") or (1.0, 1.0, 1.0))
                minimum = [minimum[axis] * scale[axis] for axis in range(3)]
                maximum = [maximum[axis] * scale[axis] for axis in range(3)]
                bottom = item["position_m"][2] + minimum[2]
                # A tilted mesh's local z extent is not its height.
                if bottom <= top_z - 0.01:
                    quat = item["quat_wxyz"]
                    rotation = Gf.Rotation(Gf.Quatd(quat[0], Gf.Vec3d(quat[1], quat[2], quat[3])))
                    heights = [
                        item["position_m"][2] + rotation.TransformDir(Gf.Vec3d(x, y, z))[2]
                        for x in (minimum[0], maximum[0])
                        for y in (minimum[1], maximum[1])
                        for z in (minimum[2], maximum[2])
                    ]
                    if max(heights) > top_z:
                        continue
                    bottom = min(heights)
                assert bottom > top_z - 0.01, (scene_path.name, item["name"], bottom, top_z)


def test_sauce_bottles_rest_on_the_oak_top():
    """The bottles_crate poses were saved after the bottles had sunk into the table.

    Each bottle and the mug sit 3 mm above the oak top. The crate sits 10 mm above it. The old
    heights put the BBQ bottle and the mug inside the top and left the ranch bottle floating.
    """
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom

    bundle = Path(__file__).resolve().parents[2] / "envhub" / "environments" / "franka-tabletop-v13"
    scene_path = bundle / "scenes" / "sauce_bottles_crate-bottles_crate.json"
    if not scene_path.is_file():
        pytest.skip("franka-tabletop-v13 is not next to this package")
    payload = json.loads(scene_path.read_text(encoding="utf-8"))
    table = Usd.Stage.Open(str(bundle / "assets" / "fixtures" / "table_oak.usd"))
    offset = UsdGeom.Xformable(table.GetPrimAtPath("/Root/table_01")).GetLocalTransformation().ExtractTranslation()
    top_z = payload["surface_pose"]["pos"][2] + offset[2]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    for item in payload["placements"]:
        if item.get("kind") == "anchor":
            continue
        mesh = Usd.Stage.Open(str(bundle / item["usd"]))
        bounds = cache.ComputeWorldBound(mesh.GetDefaultPrim()).ComputeAlignedRange()
        minimum, maximum = bounds.GetMin(), bounds.GetMax()
        quat = item["quat_wxyz"]
        rotation = Gf.Rotation(Gf.Quatd(quat[0], Gf.Vec3d(quat[1], quat[2], quat[3])))
        heights = [
            item["position_m"][2] + rotation.TransformDir(Gf.Vec3d(x, y, z))[2]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ]
        # The crate is raised clear of the top. The bottles and the mug keep the 3 mm rest gap.
        clearance = 0.01 if item.get("kind") == "container" else 0.003
        assert min(heights) == pytest.approx(top_z + clearance, abs=1e-4), item["name"]


def test_cleaning_shelf_uses_the_source_mesh_pose():
    """The rack mesh is shifted off its root, and the small bottle sits on the tabletop.

    The source scene overrides ``SM_BulkStorageRack_A01_01`` by (-0.165, 0, 0) in the
    rack frame. After the root scale and yaw, that is 0.0825 m toward -Y. The smallest
    bottle was authored 3 mm below the oak top, so its body started inside the collider.
    """
    root = Path(__file__).resolve().parents[2] / "envhub" / "environments" / "franka-tabletop-v10" / "scenes"
    if not root.is_dir():
        pytest.skip("envhub bundle is not next to this package")
    names = (
        "bottles_on_shelf-shelf_with_cleaning_products.json",
        "jugs_on_shelf-shelf_with_cleaning_products.json",
        "one_bottle_in_square_pail-shelf_with_cleaning_products.json",
        "one_bottle_on_shelf-shelf_with_cleaning_products.json",
        "reorient_jug-shelf_with_cleaning_products.json",
    )
    for name in names:
        payload = json.loads((root / name).read_text(encoding="utf-8"))
        by_slot = {item["slot"]: item for item in payload["placements"]}
        rack = by_slot["large_storage_rack"]
        bottle = by_slot["whitepackerbottle_a03"]
        assert rack["kinematic"] is True
        assert rack["position_m"][1] == pytest.approx(-0.7378082587938234)
        assert bottle["position_m"][2] == pytest.approx(0.003)


def test_load_scene_keeps_the_authored_object_pose():
    """A scene file's pose is the pose that is simulated. The task does not move an object by name."""
    root = Path(__file__).resolve().parents[2] / "envhub" / "environments" / "franka-tabletop-v10" / "scenes"
    if not root.is_dir():
        pytest.skip("envhub bundle is not next to this package")
    names = (
        "butter_above_raisin-butter_raisin_box.json",
        "bigger_object-butter_raisin_box_grey_bin.json",
        "smaller_object-butter_raisin_box_grey_bin.json",
    )
    for name in names:
        payload = json.loads((root / name).read_text(encoding="utf-8"))
        scene = load_scene(root.parent, payload["task_id"], payload["scene_id"])
        assert scene is not None
        for item in payload["placements"]:
            if item.get("kind") == "anchor":
                continue
            assert scene.by_slot()[item["slot"]].position_m == pytest.approx(item["position_m"])


def test_finger_pads_keep_their_authored_ccd_and_contact_skin():
    """The pads author enableCCD and no contact offset. The task leaves the skin to PhysX."""
    pytest.importorskip("pxr")
    from franka_tabletop.paths import packaged_asset

    from pxr import Usd

    usd = Path(packaged_asset("robots", "franka_robotiq_2f85.usd"))
    if not usd.is_file():
        pytest.skip("packaged Franka USD is not next to this package")
    stage = Usd.Stage.Open(str(usd))
    gripper = f"{stage.GetDefaultPrim().GetPath()}/Gripper/Robotiq_2F_85"
    for side in ("left", "right"):
        pad = stage.GetPrimAtPath(f"{gripper}/{side}_inner_finger")
        assert pad.GetAttribute("physxRigidBody:enableCCD").Get() is True
        for collider in pad.GetChildren():
            assert not collider.GetAttribute("physxCollision:contactOffset").HasAuthoredValue()

    package = ROOT / "source" / "franka_tabletop" / "franka_tabletop"
    env_source = (package / "env_cfg.py").read_text(encoding="utf-8")
    for source in (env_source, (package / "core" / "robot.py").read_text(encoding="utf-8")):
        assert "contact_offset" not in source and "rest_offset" not in source


def test_finger_drive_saturates_short_of_closed_instead_of_locking():
    """The finger drive reaches its torque cap a few degrees before the closed target.

    A stiffer drive saturates within a fraction of a degree. The solver then treats it as a position
    lock that closes the pads through the object.
    """
    pytest.importorskip("pxr")
    from franka_tabletop.paths import packaged_asset

    from pxr import Usd

    usd = Path(packaged_asset("robots", "franka_robotiq_2f85.usd"))
    if not usd.is_file():
        pytest.skip("packaged Franka USD is not next to this package")
    stage = Usd.Stage.Open(str(usd))
    joint = stage.GetPrimAtPath(f"{stage.GetDefaultPrim().GetPath()}/Gripper/Robotiq_2F_85/Joints/finger_joint")
    # USD angular drive stiffness is per degree.
    stiffness_per_rad = joint.GetAttribute("drive:angular:physics:stiffness").Get() * 180.0 / np.pi
    saturation_deg = np.degrees(joint.GetAttribute("drive:angular:physics:maxForce").Get() / stiffness_per_rad)
    assert 3.0 <= saturation_deg <= 10.0


def test_inner_finger_mimics_are_overdamped_and_clamped_to_stroke():
    """The inner finger and knuckle mimics are compliant so they yield to contact, overdamped so they do not
    oscillate, and bounded to the 45 deg stroke of ``finger_joint`` so they cannot fold past closed or open.
    """
    pytest.importorskip("pxr")
    from franka_tabletop.paths import packaged_asset

    from pxr import Usd

    usd = Path(packaged_asset("robots", "franka_robotiq_2f85.usd"))
    if not usd.is_file():
        pytest.skip("packaged Franka USD is not next to this package")
    stage = Usd.Stage.Open(str(usd))
    joints = f"{stage.GetDefaultPrim().GetPath()}/Gripper/Robotiq_2F_85/Joints"
    for side in ("left", "right"):
        for name in (f"{side}_inner_finger_joint", f"{side}_inner_finger_knuckle_joint"):
            joint = stage.GetPrimAtPath(f"{joints}/{name}")
            freq = joint.GetAttribute("physxMimicJoint:rotX:naturalFrequency").Get()
            damping = joint.GetAttribute("physxMimicJoint:rotX:dampingRatio").Get()
            gearing = joint.GetAttribute("physxMimicJoint:rotX:gearing").Get()
            limits = (
                joint.GetAttribute("physics:lowerLimit").Get(),
                joint.GetAttribute("physics:upperLimit").Get(),
            )
            assert freq > 0.0 and damping >= 1.0, name
            expected_limits = (0.0, 45.0) if gearing < 0.0 else (-45.0, 0.0)
            assert limits == expected_limits, name


def test_a_sixteen_env_shard_keeps_every_contact_pair():
    """Closing the grippers in a full shard overflows the default PhysX patch buffer."""
    source = (ROOT / "source" / "franka_tabletop" / "franka_tabletop" / "env_cfg.py").read_text(encoding="utf-8")
    assert "gpu_max_rigid_patch_count = 2**21" in source
    assert "gpu_found_lost_pairs_capacity = 2**23" in source


def test_work_surface_collider_is_kinematic():
    """The oak and maple payloads are rigid bodies. They stay with the visual top."""
    pytest.importorskip("pxr")
    from pxr import Usd, UsdPhysics

    root = Path(__file__).resolve().parents[2] / "envhub" / "environments"
    if not root.is_dir():
        pytest.skip("envhub bundle is not next to this package")
    for env_id in ("franka-tabletop-v2", "franka-tabletop-v0", "franka-tabletop-v1", "franka-tabletop-v10"):
        for name in ("table_oak.usd", "table_maple.usd", "table_bamboo.usd"):
            path = root / env_id / "assets" / "fixtures" / name
            if not path.is_file():
                continue
            stage = Usd.Stage.Open(str(path))
            body = UsdPhysics.RigidBodyAPI(stage.GetDefaultPrim())
            assert body.GetRigidBodyEnabledAttr().Get() is True
            assert body.GetKinematicEnabledAttr().Get() is True, path


def test_right_exterior_camera_mirrors_the_left():
    """The right exterior is the left camera reflected across the robot center, and stays upright."""
    source = (ROOT / "source" / "franka_tabletop" / "franka_tabletop" / "core" / "sensors.py").read_text(
        encoding="utf-8"
    )
    assert "pos=(0.05, 0.57, 0.66)" in source
    assert "pos=(0.05, -0.57, 0.66)" in source
    left = _camera_axes((-0.195, 0.399, 0.805, -0.393))
    right = _camera_axes((0.399, -0.195, -0.393, 0.805))
    assert right["up"][2] > 0.5
    # Both quaternions are stored to three decimals, so the mirror holds to about 0.001.
    mirror = np.array([1.0, -1.0, 1.0])
    assert np.allclose(right["forward"], left["forward"] * mirror, atol=1e-3)
    assert np.allclose(right["up"], left["up"] * mirror, atol=1e-3)


def _camera_axes(xyzw: tuple[float, float, float, float]) -> dict[str, np.ndarray]:
    """OpenGL camera axes in the parent frame. The camera looks along its local -Z."""
    x, y, z, w = np.asarray(xyzw, dtype=np.float64)
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return {"forward": -rotation[:, 2], "up": rotation[:, 1]}


def test_benchmark_wxyz_rotation_converts_to_xyzw():
    assert xyzw_from_wxyz((1.0, 0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0, 1.0)
    assert xyzw_from_wxyz((0.0, 0.0, 0.0, 1.0)) == (0.0, 0.0, 1.0, 0.0)
    assert xyzw_from_wxyz((-0.393, -0.195, 0.399, 0.805)) == (-0.195, 0.399, 0.805, -0.393)
    assert xyzw_from_wxyz((-0.420, 0.570, 0.576, -0.409)) == (0.570, 0.576, -0.409, -0.420)


def test_scene_selects_one_background_and_one_surface(tmp_path: Path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "stage.json").write_text(
        json.dumps(
            {
                "backgrounds": [
                    {"name": "home_office", "texture": "assets/backgrounds/home_office.exr"},
                    {"name": "warehouse", "texture": "assets/backgrounds/warehouse.hdr", "intensity": 800.0},
                ],
                "background": "home_office",
                "fixture": {"usd": "assets/fixtures/franka_table.usd"},
                "surfaces": [
                    {
                        "name": "oak",
                        "usd": "assets/fixtures/table_oak.usd",
                        "table_m": [0.7, 1.0, 0.03],
                        "center_xy": [0.55, 0.0],
                    },
                    {
                        "name": "black",
                        "usd": "assets/fixtures/table_black.usd",
                        "table_m": [0.7, 1.0, 0.03],
                        "center_xy": [0.55, 0.0],
                    },
                ],
                "surface": "oak",
            }
        ),
        encoding="utf-8",
    )
    stage = read_stage(tmp_path)
    assert resolve_background(stage)["texture"] == "assets/backgrounds/home_office.exr"
    assert resolve_background(stage, "warehouse")["intensity"] == 800.0
    assert resolve_surface(stage, "black")["usd"] == "assets/fixtures/table_black.usd"
    with pytest.raises(KeyError):
        resolve_background(stage, "missing")


def test_stage_requires_a_background_and_a_fixture(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_stage(tmp_path)
    stage = tmp_path / "assets"
    stage.mkdir()
    (stage / "stage.json").write_text(json.dumps({"fixture": {"usd": "assets/fixtures/table.usd"}}), encoding="utf-8")
    with pytest.raises(ValueError):
        read_stage(tmp_path)


def test_stage_physics_block_changes_substeps_but_not_the_control_rate():
    pytest.importorskip("isaaclab_physx")
    from franka_tabletop.env_cfg import TabletopBenchmarkEnvCfg, _apply_benchmark_physics

    cfg = TabletopBenchmarkEnvCfg()
    physx = cfg.sim.physics.isaacsim_physx
    assert (cfg.decimation, cfg.sim.dt) == (8, 1.0 / 120.0)
    assert (physx.enable_ccd, physx.solve_articulation_contact_last) == (False, False)
    _apply_benchmark_physics(cfg, {"rate_hz": 60, "solver_position_iterations": 16, "enhanced_determinism": False})
    assert (cfg.decimation, cfg.sim.dt, cfg.sim.render_interval) == (4, 1.0 / 60.0, 4)
    assert cfg.decimation * cfg.sim.dt == pytest.approx(1.0 / 15.0)
    props = [p for p in cfg.scene.robot.spawn.articulation_props if hasattr(p, "solver_position_iteration_count")]
    assert [p.solver_position_iteration_count for p in props] == [16]
    assert physx.enable_enhanced_determinism is False
    with pytest.raises(ValueError):
        _apply_benchmark_physics(cfg, {"rate_hz": 50})


def test_brain_cfg_keeps_cameras_out_of_the_per_step_observation_group(monkeypatch):
    pytest.importorskip("isaaclab_physx")
    from franka_tabletop.core.observation_spec import OBSERVATION_CAMERAS
    from franka_tabletop.env_cfg import build_cfg

    monkeypatch.delenv("ISAAC_EVAL_BUNDLE_DIR", raising=False)
    brain = build_cfg(observation={})
    assert all(getattr(brain.observations.policy, name) is None for name in OBSERVATION_CAMERAS)
    assert all(getattr(brain.scene, name) is not None for name in OBSERVATION_CAMERAS)
    in_process = build_cfg()
    assert all(getattr(in_process.observations.policy, name) is not None for name in OBSERVATION_CAMERAS)


def test_a_recording_run_renders_the_front_camera_at_one_grid_cell(monkeypatch):
    pytest.importorskip("isaaclab_physx")
    pytest.importorskip("warp")
    from franka_tabletop.core.clips import TILE_HW
    from franka_tabletop.env_cfg import build_cfg

    monkeypatch.delenv("ISAAC_EVAL_BUNDLE_DIR", raising=False)
    monkeypatch.setattr("franka_tabletop.env_cfg.video_directory", lambda: "/output/videos")
    cfg = build_cfg(observation={"cameras": ["cam_wrist"]})
    assert (cfg.scene.cam_front.height, cfg.scene.cam_front.width) == TILE_HW
    assert (cfg.scene.cam_wrist.height, cfg.scene.cam_wrist.width) == (720, 1280)
    assert (cfg.scene.cam_exterior_left.height, cfg.scene.cam_exterior_left.width) == (720, 1280)
    assert cfg.scene.cam_exterior_right is None

    monkeypatch.setattr("franka_tabletop.env_cfg.video_directory", lambda: None)
    plain = build_cfg(observation={"cameras": ["cam_wrist"]})
    assert plain.scene.cam_front is None
    assert plain.scene.cam_exterior_left is None


def _names() -> dict[str, str]:
    return {name: name for name in ("cube", "block", "can", "bin", "tray", "lid")}


def _pose(position: list[float], half: list[float] | None = None, quat=(1.0, 0.0, 0.0, 0.0)) -> dict:
    return _body(position, tuple(2.0 * value for value in (half or [0.2, 0.2, 0.2])), quat)


def test_custom_slot_names_are_not_a_fixed_list(tmp_path: Path):
    scene_dir = tmp_path / "scenes"
    scene_dir.mkdir()
    payload = {
        "instruction": "Put the cube in the bin",
        "success": {"op": "in_container", "object": "cube", "container": "bin", "require_gripper_detached": True},
        "placements": [
            {
                "slot": "cube",
                "name": "cube",
                "usd": "a.usd",
                "size_m": [0.05, 0.05, 0.05],
                "position_m": [0.4, 0, 0.02],
            },
            {"slot": "bin", "name": "bin", "usd": "b.usd", "size_m": [0.2, 0.2, 0.1], "position_m": [0.5, 0.2, 0.05]},
        ],
    }
    (scene_dir / "custom-desk.json").write_text(json.dumps(payload), encoding="utf-8")
    scene = load_scene(tmp_path, "custom", "desk")
    assert scene is not None
    assert [placement.slot for placement in scene.placements] == ["cube", "bin"]
    assert scene.spec is not None
    assert scene.spec.check is not None
    goal = scene.spec.check.goals[0]
    assert (goal.slot, goal.target, goal.detach) == ("cube", "bin", True)


def test_all_averages_object_scores_and_any_takes_the_max():
    names = _names()
    poses = {"cube": _pose([0.0, 0.0, 0.0]), "block": _pose([1.0, 0.0, 0.0]), "bin": _pose([0.0, 0.0, 0.0])}
    shared = {"op": "in_container", "object": ["cube", "block"], "container": "bin"}
    both = spec_from_success("pack", {**shared, "logical": "all"}, 40.0, names)
    either = spec_from_success("pack", {**shared, "logical": "any"}, 40.0, names)
    score, latched, _ = task_progress(both, poses, Contacts(), {}, [])
    assert abs(score - 0.5) < 1e-9
    assert not task_satisfied(both, poses, Contacts(), {})
    nudged = {"cube": _pose([1.0, 0.0, 0.0]), "block": _pose([1.0, 0.0, 0.0]), "bin": _pose([0.0, 0.0, 0.0])}
    kept, _, _ = task_progress(both, nudged, Contacts(), latched, [])
    assert abs(kept - 0.5) < 1e-9
    either_score, _, _ = task_progress(either, poses, Contacts(), {}, [])
    assert either_score == 1.0
    assert task_satisfied(either, poses, Contacts(), {})


def test_three_objects_share_one_container():
    names = _names()
    spec = spec_from_success(
        "pack",
        {"op": "in_container", "object": ["cube", "block", "can"], "container": "bin"},
        40.0,
        names,
    )
    poses = {
        "cube": _pose([0.0, 0.0, 0.0]),
        "block": _pose([1.0, 0.0, 0.0]),
        "can": _pose([1.0, 0.0, 0.0]),
        "bin": _pose([0.0, 0.0, 0.0]),
    }
    score, _, _ = task_progress(spec, poses, Contacts(), {}, [])
    assert abs(score - 1.0 / 3.0) < 1e-9
    poses["block"] = _pose([0.0, 0.0, 0.0])
    poses["can"] = _pose([0.0, 0.0, 0.0])
    done, _, _ = task_progress(spec, poses, Contacts(), {}, [])
    assert done == 1.0
    assert task_satisfied(spec, poses, Contacts(), {})


def test_choose_needs_exactly_k_objects_in_success_and_at_least_k_in_progress():
    names = _names()
    spec = spec_from_success(
        "pack",
        {"op": "in_container", "object": ["cube", "block", "can"], "container": "bin", "logical": "choose", "k": 2},
        40.0,
        names,
    )

    def poses(*placed):
        found = {name: _pose([0.0 if name in placed else 1.0, 0.0, 0.0]) for name in ("cube", "block", "can")}
        return {**found, "bin": _pose([0.0, 0.0, 0.0])}

    assert task_satisfied(spec, poses("cube", "can"), Contacts(), {})
    assert not task_satisfied(spec, poses("cube"), Contacts(), {})
    assert not task_satisfied(spec, poses("cube", "block", "can"), Contacts(), {})
    assert task_progress(spec, poses("cube", "block", "can"), Contacts(), {}, [])[0] == 1.0
    assert task_progress(spec, poses("cube"), Contacts(), {}, [])[0] == 0.5


def test_in_order_follows_the_first_placement_sequence():
    names = _names()
    spec = spec_from_success(
        "pack",
        {"op": "in_order", "object": ["cube", "block"], "container": "bin"},
        40.0,
        names,
    )
    bin_pose = _pose([0.0, 0.0, 0.0])
    outside = _pose([1.0, 0.0, 0.0])
    inside = _pose([0.0, 0.0, 0.0])
    # The block goes in first. The order asks for the cube first, so the block earns no credit.
    block_first = {"cube": outside, "block": inside, "bin": bin_pose}
    both = {"cube": inside, "block": inside, "bin": bin_pose}
    score, latched, entered = task_progress(spec, block_first, Contacts(), {}, [])
    assert entered == ["block"]
    assert score == 0.0
    score, _, entered = task_progress(spec, both, Contacts(), latched, entered)
    assert entered == ["block", "cube"]
    assert score < 1.0
    cube_first = {"cube": inside, "block": outside, "bin": bin_pose}
    score, latched, entered = task_progress(spec, cube_first, Contacts(), {}, [])
    assert entered == ["cube"]
    assert abs(score - 0.5) < 1e-9
    score, latched, entered = task_progress(spec, both, Contacts(), latched, entered)
    assert entered == ["cube", "block"]
    assert score == 1.0
    left = {"cube": outside, "block": outside, "bin": bin_pose}
    kept, _, _ = task_progress(spec, left, Contacts(), latched, entered)
    assert kept == 1.0

    placed: dict[str, int] = {}
    for step, poses in enumerate([cube_first, both]):
        note_placements(spec, poses, Contacts(), placed, step)
    assert placed == {"cube": 0, "block": 1}
    assert task_satisfied(spec, both, Contacts(), placed)
    assert not task_satisfied(spec, both, Contacts(), {"cube": 1, "block": 0})


def test_outside_needs_the_hull_center_clear_and_does_not_stick():
    names = _names()
    spec = spec_from_success(
        "take",
        {"op": "outside", "object": ["cube"], "container": "bin", "require_gripper_detached": True},
        40.0,
        names,
    )
    bowl = _pose([0.0, 0.0, 0.0], [0.05, 0.05, 0.03])
    held = Contacts(touching={"cube": True})
    free = Contacts(touching={"cube": False})
    # The center is past the wall, so the spoon is outside, but the gripper still holds it.
    clear = {"cube": _pose([0.06, 0.0, 0.08], [0.05, 0.01, 0.03]), "bin": bowl}
    score, latched, _ = task_progress(spec, clear, held, {}, [])
    assert score == 0.0
    assert latched == {}
    score, latched, _ = task_progress(spec, clear, free, {}, [])
    assert score == 1.0
    assert task_progress(spec, clear, free, latched, [])[0] == 1.0
    # Back in the bowl, the credit is taken back.
    back = {"cube": _pose([0.0, 0.0, 0.0], [0.01, 0.01, 0.01]), "bin": bowl}
    assert task_progress(spec, back, free, latched, [])[0] == 0.0

    # Authored measuring spoon: tilted in the red bowl, gripper already released. The open top has no lid,
    # so its center, above the rim, is still in the bowl.
    spoon_pose = _pose(
        [0.332945, -0.137862, 0.158783],
        [0.0837055, 0.0413945, 0.0189455],
        (0.631510, -0.629371, 0.306461, 0.333421),
    )
    bowl_pose = _pose(
        [0.357784, -0.157000, 0.077346],
        [0.080722, 0.0805565, 0.027485],
        (0.9999995827674866, -0.000510019832290709, -0.0007578271324746311, 0.0),
    )
    authored = {"cube": spoon_pose, "bin": bowl_pose}
    assert task_progress(spec, authored, free, {}, [])[0] == 0.0


def test_on_top_reads_the_support_force_and_the_support_footprint():
    spec = spec_from_success(
        "stack", {"op": "on_top", "object": "lid", "support": "tray", "require_gripper_detached": True}, 40.0, _names()
    )
    poses = {"lid": _pose([0.0, 0.0, 0.05], [0.04, 0.04, 0.01]), "tray": _pose([0.0, 0.0, 0.0], [0.1, 0.1, 0.02])}
    force = {("lid", "tray"): np.array([0.0, 0.0, 1.0])}
    assert task_satisfied(spec, poses, Contacts(touching={"lid": False}, force=force), {})
    assert not task_satisfied(spec, poses, Contacts(touching={"lid": True}, force=force), {})
    assert not task_satisfied(spec, poses, Contacts(touching={"lid": False}), {})
    # The box center has to be over the tray, to within 0.01 m.
    off = {**poses, "lid": _pose([0.2, 0.0, 0.05], [0.04, 0.04, 0.01])}
    assert not task_satisfied(spec, off, Contacts(touching={"lid": False}, force=force), {})


def test_front_is_toward_the_robot():
    bowl = _body((0.5, 0.0, 0.0625), (0.125,) * 3)
    block = {"op": "on_side", "object": "cube", "reference": "bowl", "side": "front"}
    nearer = {"cube": _body((0.25, 0.0625, 0.0625), CUBE), "bowl": bowl}
    farther = {"cube": _body((0.75, 0.0, 0.0625), CUBE), "bowl": bowl}
    assert _met(block, nearer)
    assert not _met(block, farther)
    # The level check is skipped when the stage asks for the release.
    high = {"cube": _body((0.25, 0.0, 0.5), CUBE), "bowl": bowl}
    assert not _met(block, high)
    assert _met({**block, "require_gripper_detached": True}, high, Contacts(touching={"cube": False}))


def test_stacked_needs_contact_and_a_box_above_the_support():
    red = _body((0.5, 0.0, 0.0625), (0.125,) * 3)
    block = {"op": "stacked", "object": "yellow", "support": "red"}
    touch = Contacts(touch={("yellow", "red"): True}, touching={"yellow": True})

    def at(x, z):
        return {"yellow": _body((x, 0.0, z), (0.125,) * 3), "red": red}

    assert _met(block, at(0.5, 0.1875), touch)
    assert not _met({**block, "require_gripper_detached": True}, at(0.5, 0.1875), touch)
    assert not _met(block, at(0.5, 0.1875), Contacts(touching={"yellow": True}))
    assert not _met(block, at(0.75, 0.0625), touch)


def test_containment_tests_the_hull_center_not_the_root():
    corner = np.array([[x, y, z] for x in (0.0, 0.25) for y in (0.0, 0.25) for z in (0.0, 0.25)])
    item = {
        "pos": np.array([[0.375, -0.0625, 0.0625]]),
        "quat": np.array([[1.0, 0.0, 0.0, 0.0]]),
        "geometry": from_points(corner),
    }
    # The root sits on the wall, but the hull center of the mesh is 0.125 m inside it.
    assert _met(IN_BIN, {"cube": item, "bin": BIN})
    item = {**item, "pos": np.array([[0.125, -0.0625, 0.0625]])}
    assert not _met(IN_BIN, {"cube": item, "bin": BIN})
    # A containment tolerance does nothing, as in the reference benchmark.
    assert not _met({**IN_BIN, "tolerance": 0.5}, {"cube": item, "bin": BIN})


def test_the_success_termination_scores_the_step_before_it_reads_the_flags():
    torch = pytest.importorskip("torch")
    from franka_tabletop.core.terminations import task_success

    calls = []

    class _Env:
        def update_task_state(self):
            calls.append("score")

        def success_flags(self):
            calls.append("read")
            return torch.tensor([True, False])

    assert task_success(_Env()).tolist() == [True, False]
    assert calls == ["score", "read"]


def test_retired_ops_and_flags_are_rejected():
    for block in (
        {"op": "picked", "object": "cube"},
        {"op": "resting", "object": "cube"},
        {"op": "off", "object": "cube", "support": "bin"},
        {**IN_BIN, "require_grasp": ["cube"]},
        {**IN_BIN, "hold": ["cube"]},
    ):
        with pytest.raises(ValueError):
            spec_from_success("t", block, 40.0, _names())


def test_observation_spec_defaults_to_every_camera():
    from franka_tabletop.core.observation_spec import (
        OBSERVATION_CAMERAS,
        requested_cameras,
        requested_image_hw,
        unused_cameras,
    )

    for spec in (None, {}, {"cameras": []}):
        assert requested_cameras(spec) == OBSERVATION_CAMERAS
        assert unused_cameras(spec) == ()
    assert requested_image_hw(None) is None


def test_observation_spec_keeps_the_declared_cameras_in_observation_order():
    from franka_tabletop.core.observation_spec import requested_cameras, requested_image_hw, unused_cameras

    spec = {"cameras": ["cam_wrist", "cam_exterior_left"], "image_hw": [180, 320]}
    assert requested_cameras(spec) == ("cam_exterior_left", "cam_wrist")
    assert unused_cameras(spec) == ("cam_exterior_right",)
    assert requested_image_hw(spec) == (180, 320)
    with pytest.raises(ValueError):
        requested_cameras({"cameras": ["cam_front"]})
    with pytest.raises(ValueError):
        requested_image_hw({"image_hw": [0, 320]})


def test_tile_grid_places_env_zero_at_the_top_left():
    images = np.zeros((6, 2, 2, 3), dtype=np.uint8)
    images[0] = (1, 0, 0)
    images[1] = (0, 2, 0)
    images[4] = (0, 0, 3)
    images[5] = (4, 0, 0)
    grid = tile_grid(images, grid_hw=(8, 8))
    assert grid.shape == (8, 8, 3)
    assert np.all(grid[0:2, 0:2] == (1, 0, 0))
    assert np.all(grid[0:2, 2:4] == (0, 2, 0))
    assert np.all(grid[0:2, 4:8] == 0)
    assert np.all(grid[2:4, 0:2] == (0, 0, 3))
    assert np.all(grid[2:4, 2:4] == (4, 0, 0))
    assert np.all(grid[2:4, 4:8] == 0)
    assert np.all(grid[4:] == 0)


def test_tile_grid_rejects_a_seventeenth_environment_or_the_wrong_cell_size():
    with pytest.raises(ValueError):
        tile_grid(np.zeros((17, 2, 2, 3), dtype=np.uint8), grid_hw=(8, 8))
    with pytest.raises(ValueError):
        tile_grid(np.zeros((1, 3, 2, 3), dtype=np.uint8), grid_hw=(8, 8))


def test_a_mixed_shard_records_one_grid_named_for_its_first_job(tmp_path):
    from types import SimpleNamespace

    written: list[str] = []
    clips = EpisodeClips(str(tmp_path), lambda directory, name, frame, fps: written.append(name), grid_hw=(8, 8))
    jobs = [
        SimpleNamespace(job_id="aaa", task_id="a", scene_id="s1", variant="nominal", episode_index=0),
        SimpleNamespace(job_id="bbb", task_id="b", scene_id="s2", variant="nominal", episode_index=0),
    ]
    clips.begin(jobs, "a", "s1", "nominal")
    clips.add("front", np.zeros((2, 2, 2, 3), dtype=np.uint8))
    clips.add("exterior_left", np.zeros((2, 2, 2, 3), dtype=np.uint8))
    clips.add("wrist", np.zeros((2, 2, 2, 3), dtype=np.uint8))
    clips.end()
    assert written == ["front_aaa.mp4", "exterior_left_aaa.mp4", "wrist_aaa.mp4"]


def _mixed_stage() -> dict:
    return {
        "backgrounds": [{"name": "home_office", "texture": "a.exr"}, {"name": "billiard_hall", "texture": "b.hdr"}],
        "background": "home_office",
        "fixture": {"usd": "f.usd"},
        "surfaces": [
            {"name": "oak", "usd": "oak.usd", "pos": [0.5, 0.0, -0.35]},
            {"name": "maple", "usd": "maple.usd", "pos": [0.5, 0.0, -0.35]},
        ],
        "surface": "oak",
    }


def _mixed_scene(task: str, objects: list[str], **overrides) -> dict:
    placements = [
        {"slot": name, "kind": "item", "usd": f"{name}.usd", "position_m": [0.1 * index, 0.0, 0.0]}
        for index, name in enumerate(objects)
    ]
    placements.append({"slot": "mark", "kind": "anchor", "position_m": [0.0, 0.0, 0.0]})
    return {"task_id": task, "background": "home_office", "surface": "oak", "placements": placements, **overrides}


def test_mixed_plan_gives_each_env_its_own_mesh_per_position():
    from franka_tabletop.scenes.mixed import PARK_POSITION_M, plan_mixed

    scenes = [
        _mixed_scene("a", ["banana", "bowl"]),
        _mixed_scene(
            "b", ["mug", "banana", "bin"], surface="maple", surface_pose={"pos": [0.6, 0.0, -0.347]}
        ),
    ]
    scenes[1]["placements"][2]["kinematic"] = True
    plan = plan_mixed(scenes, _mixed_stage())
    assert plan.entities == ("obj0", "obj1", "obj2")
    assert [[v.usd if v else None for v in column] for column in plan.variants] == [
        ["banana.usd", "mug.usd"],
        ["bowl.usd", "banana.usd"],
        ["bin.usd", None],
    ]
    assert plan.choices == ((0, 1), (0, 1), (1, 0))
    assert plan.slot_entities[0] == {"banana": "obj0", "bowl": "obj1"}
    assert plan.slot_entities[1] == {"mug": "obj0", "banana": "obj1", "bin": "obj2"}
    assert plan.parked(0) == ["obj2"] and plan.parked(1) == []
    assert plan.variants[2][0].kinematic and plan.variants[2][1] is None
    assert plan.surfaces == ("oak", "maple") and plan.surface_choices == (0, 1)
    assert plan.surface_poses[1][0] == (0.6, 0.0, -0.347)
    assert plan.surface_poses[0][0] == (0.5, 0.0, -0.35)
    assert plan.start_poses[2][0] == pytest.approx((0.2, 0.0, 0.0))
    assert plan.choice_by_name()["surface"] == (0, 1)
    assert PARK_POSITION_M[2] > 1.0


def test_mixed_plan_rejects_scenes_with_different_backgrounds_or_duplicate_slots():
    from franka_tabletop.scenes.mixed import plan_mixed

    other = _mixed_scene("b", ["cup"], background="billiard_hall")
    with pytest.raises(ValueError, match="background"):
        plan_mixed([_mixed_scene("a", ["cup"]), other], _mixed_stage())
    with pytest.raises(ValueError, match="more than once"):
        plan_mixed([_mixed_scene("a", ["cup", "cup"])], _mixed_stage())
    with pytest.raises(ValueError, match="no objects"):
        plan_mixed([_mixed_scene("a", [])], _mixed_stage())


def test_explicit_worlds_pick_one_variant_per_env_without_the_product():
    from franka_tabletop.scenes.mixed import explicit_worlds

    names = ["robot", "obj0", "surface"]
    worlds, weights = explicit_worlds(names, [1, 2, 2], {"obj0": (1, 0), "surface": (0, 1)}, 2)
    assert worlds == ((0, 2, 3), (0, 1, 4))
    assert weights.tolist() == [1.0, 1.0]
    big = [f"obj{index}" for index in range(19)]
    worlds, _ = explicit_worlds(big, [16] * 19, {name: tuple(range(16)) for name in big}, 16)
    assert len(worlds) == 16
    with pytest.raises(ValueError):
        explicit_worlds(names, [1, 2, 2], {"obj0": (2, 0)}, 2)
    with pytest.raises(ValueError):
        explicit_worlds(names, [1, 2, 2], {"obj0": (0,)}, 2)


def test_planned_worlds_replace_the_cloner_only_while_the_env_is_built(monkeypatch):
    import types

    from franka_tabletop.core.clone import planned_worlds
    from franka_tabletop.scenes.mixed import plan_mixed

    cloner = types.ModuleType("isaaclab.cloner")
    cloner.make_valid_clone_combinations = original = lambda *args, **kwargs: "default"
    package = types.ModuleType("isaaclab")
    package.cloner = cloner
    monkeypatch.setitem(sys.modules, "isaaclab", package)
    monkeypatch.setitem(sys.modules, "isaaclab.cloner", cloner)
    plan = plan_mixed([_mixed_scene("a", ["cup"]), _mixed_scene("b", ["mug"])], _mixed_stage())
    with pytest.raises(RuntimeError):
        with planned_worlds(plan):
            worlds, _ = cloner.make_valid_clone_combinations(["obj0", "surface"], [2, 1], [object()])
            assert worlds == ((0, 2), (1, 2))
            raise RuntimeError("scene construction failed")
    assert cloner.make_valid_clone_combinations is original


def test_v11_backgrounds_plan_into_stages_of_at_most_nineteen_positions():
    bundle = ROOT.parent / "envhub" / "environments" / "franka-tabletop-v11"
    if not (bundle / "assets" / "stage.json").is_file():
        pytest.skip("franka-tabletop-v11 is not checked out")
    from franka_tabletop.scenes.mixed import plan_mixed

    stage = read_stage(bundle)
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((bundle / "scenes").glob("*.json"))]
    by_background: dict[str, list[dict]] = {}
    for payload in payloads:
        by_background.setdefault(payload["background"], []).append(payload)
    assert sum(len(group) for group in by_background.values()) == 120
    for group in by_background.values():
        plan = plan_mixed(group, stage)
        assert len(plan.entities) <= 19
        for env, payload in enumerate(group):
            placed = [item["slot"] for item in payload["placements"] if item.get("kind") != "anchor"]
            assert list(plan.slot_entities[env]) == placed
            assert len(set(plan.slot_entities[env].values())) == len(placed)


def _turn(axis, angle):
    quat = [np.cos(angle / 2.0), 0.0, 0.0, 0.0]
    quat[1 + axis] = np.sin(angle / 2.0)
    return tuple(quat)


def _moved(pose, pos=None, quat=None):
    """The pose now at ``pos`` and ``quat``, with the reset pose left where it was."""
    moved = dict(pose)
    if pos is not None:
        moved["pos"] = np.array([pos], dtype=np.float64)
    if quat is not None:
        moved["quat"] = np.array([quat], dtype=np.float64)
    return moved


def _rebased(pose):
    """The pose with its reset pose set to where it is now."""
    return {**pose, "initial_pos": pose["pos"], "initial_quat": pose["quat"]}


def test_a_reference_has_to_stay_near_its_reset_body():
    shelf = _body((0.5, 0.0, 0.25), (0.25, 0.25, 0.5))
    block = {"op": "on_top", "object": "cube", "reference": "shelf"}
    carried = Contacts(force={("cube", "shelf"): np.array([0.0, 0.0, 1.0])})
    cube = _body((0.5, 0.0, 0.53125), CUBE)

    def met(reference, rebase=False):
        return _met(block, {"cube": cube, "shelf": _rebased(reference) if rebase else reference}, carried)

    assert met(shelf)
    # Within 0.05 m and 15 degrees, a settled shelf still counts.
    assert met(_moved(shelf, pos=(0.546875, 0.0, 0.25), quat=_turn(0, np.radians(14.0))))
    # Shoved past 0.05 m, or turned past 15 degrees, it does not. The geometry alone would pass.
    shoved = _moved(shelf, pos=(0.5625, 0.0, 0.25))
    assert not met(shoved) and met(shoved, rebase=True)
    turned = _moved(shelf, quat=_turn(0, np.radians(16.0)))
    assert not met(turned) and met(turned, rebase=True)
    # A tipped shelf is the case in point: the mug rests on the fallen side.
    tipped = _moved(shelf, quat=_turn(1, np.pi / 2.0))
    assert not met(tipped) and met(tipped, rebase=True)


def test_a_reference_without_a_reset_pose_fails_its_stage():
    poses = {"cube": _body((0.5, 0.0, 0.09375), CUBE), "bin": BIN}
    assert _met(IN_BIN, poses)
    bare = {key: value for key, value in BIN.items() if not key.startswith("initial")}
    assert not _met(IN_BIN, {**poses, "bin": bare})


def test_a_shoved_container_does_not_count_as_taking_an_object_out_of_it():
    block = {"op": "outside_and_on_surface", "object": "cube", "container": "bin"}
    carried = Contacts(force={("cube", "surface"): np.array([0.0, 0.0, 1.0])})
    cube = _body((0.5, 0.0, 0.03125), CUBE)
    # The container slides away from the object, which sits on the table.
    away = _moved(BIN, pos=(0.5, 0.5, 0.0625))
    assert not _met(block, {"cube": cube, "bin": away}, carried)
    assert _met(block, {"cube": cube, "bin": _rebased(away)}, carried)


def test_objects_the_task_places_may_leave_their_reset_pose_but_the_base_may_not():
    block = {
        "op": "all",
        "checks": [
            {"op": "stacked", "object": "blue", "support": "red"},
            {"op": "stacked", "object": "green", "support": "blue"},
        ],
    }
    edge = (0.125,) * 3
    touch = Contacts(touch={("blue", "red"): True, ("green", "blue"): True})
    red = _body((0.5, 0.0, 0.0625), edge)
    # Blue and green came from elsewhere. Blue is the support of green and is placed by its own stage.
    blue = _moved(_body((0.9, 0.5, 0.0625), edge), pos=(0.5, 0.0, 0.1875))
    green = _moved(_body((0.9, -0.5, 0.0625), edge), pos=(0.5, 0.0, 0.3125))
    assert _met(block, {"red": red, "blue": blue, "green": green}, touch)
    # The base was shoved 0.0625 m, so the tower no longer counts.
    shoved = _moved(red, pos=(0.5, 0.0625, 0.0625))
    assert not _met(block, {"red": shoved, "blue": blue, "green": green}, touch)
    assert _met(block, {"red": _rebased(shoved), "blue": blue, "green": green}, touch)


def test_only_the_passing_branch_of_an_any_keeps_its_support_in_place():
    block = {
        "op": "any",
        "checks": [
            {"op": "stacked", "object": "upper", "support": "lower"},
            {"op": "stacked", "object": "lower", "support": "upper"},
        ],
    }
    edge = (0.125,) * 3
    touch = Contacts(touch={("upper", "lower"): True})
    upper = _moved(_body((0.9, 0.5, 0.0625), edge), pos=(0.5, 0.0, 0.1875))
    lower = _body((0.5, 0.0, 0.0625), edge)
    assert _met(block, {"upper": upper, "lower": lower}, touch)
    # The mug that carries the other is not exempt just because the other branch places it.
    assert not _met(block, {"upper": upper, "lower": _moved(lower, pos=(0.5, 0.0625, 0.0625))}, touch)


def test_a_shoved_container_earns_no_progress():
    spec = spec_from_success("t", IN_BIN, 40.0, {"cube": "cube", "bin": "bin"})
    cube = _body((0.5, 0.5, 0.09375), CUBE)
    shoved = _moved(BIN, pos=(0.5, 0.5, 0.0625))
    score, latched, _ = task_progress(spec, {"cube": cube, "bin": shoved}, Contacts(), {}, [])
    assert score == 0.0 and latched == {}
    assert task_progress(spec, {"cube": cube, "bin": _rebased(shoved)}, Contacts(), {}, [])[0] == 1.0
