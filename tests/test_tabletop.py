# Copyright (c) 2026, Nepher Robotics
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
from franka_tabletop.expert import fk, ik
from franka_tabletop.instructions import phrase
from franka_tabletop.predicates import evaluate, inside, on_footprint, task_progress, terminal_success
from franka_tabletop.scenes.bundle import (
    read_catalog,
    read_stage,
    resolve_background,
    resolve_surface,
    surface_pose,
    xyzw_from_wxyz,
)
from franka_tabletop.scenes.catalog import PUBLIC_SEEDS, ObjectAsset
from franka_tabletop.core.clips import EpisodeClips, tile_grid
from franka_tabletop.scenes.composer import compose, load_scene, world_position
from franka_tabletop.spec import FAMILIES, get_family, spec_from_success


def _sample_catalog() -> dict[str, ObjectAsset]:
    return {
        "block": ObjectAsset("block", (0.05, 0.05, 0.05), "item", "assets/objects/block.usd"),
        "can": ObjectAsset("can", (0.06, 0.06, 0.08), "item", "assets/objects/can.usd"),
        "bin": ObjectAsset("bin", (0.16, 0.16, 0.06), "container", "assets/objects/bin.usd"),
        "post": ObjectAsset("post", (0.04, 0.04, 0.12), "reference", "assets/objects/post.usd"),
    }


def _compose(seed: int, task_id: str, pose_jitter_m: float = 0.0):
    return compose(
        seed,
        task_id,
        _sample_catalog(),
        table_m=(1.1, 0.7, 0.04),
        center_xy=(0.35, 0.0),
        pose_jitter_m=pose_jitter_m,
        archetype="kitchen_counter",
    )


def test_inside_box():
    item = np.array([[0.0, 0.0, 0.0]])
    box = np.array([[0.0, 0.0, 0.0]])
    assert bool(inside(item, box, np.array([0.1, 0.1, 0.1]))[0])
    assert not bool(inside(item + np.array([[0.2, 0.0, 0.0]]), box, np.array([0.1, 0.1, 0.1]))[0])
    # A center just above the rim of a shallow bin is still in the bin. Far above it is not.
    assert bool(inside(item + np.array([[0.0, 0.0, 0.2]]), box, np.array([0.1, 0.1, 0.1]))[0])
    assert not bool(inside(item + np.array([[0.0, 0.0, 0.4]]), box, np.array([0.1, 0.1, 0.1]))[0])


def test_inside_follows_the_container_yaw():
    """A quarter turn swaps the box axes. The long local side is then the scene y axis."""
    origin = np.zeros((1, 3))
    half = np.array([0.21, 0.14, 0.05])
    yaw = np.array([[np.cos(np.pi / 4.0), 0.0, 0.0, np.sin(np.pi / 4.0)]])
    along_scene_y = np.array([[0.0, 0.18, 0.0]])
    assert bool(inside(along_scene_y, origin, half, yaw)[0])
    assert not bool(inside(along_scene_y, origin, half)[0])


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
    held = {}
    released = {}
    for task_id, scene_id in (
        ("bowl_in_bin", "bin_mug_mustard_marker_bowl"),
        ("bigger_object", "butter_raisin_box_grey_bin"),
    ):
        scene = load_scene(bundle, task_id, scene_id)
        if scene is None:
            continue
        poses = {}
        for placement in scene.placements:
            poses[placement.slot] = {
                "pos": np.asarray([placement.position_m], dtype=np.float64),
                "quat": np.asarray([placement.quat_wxyz], dtype=np.float64),
                "half": np.asarray(placement.asset.size_m, dtype=np.float64) * 0.5,
            }
            held[placement.slot] = True
            released[placement.slot] = False
        assert not terminal_success(scene.spec, poses, held, released, side=scene.side)


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
            poses = {}
            held = {}
            released = {}
            for placement in scene.placements:
                poses[placement.slot] = {
                    "pos": np.asarray([placement.position_m], dtype=np.float64),
                    "quat": np.asarray([placement.quat_wxyz], dtype=np.float64),
                    "half": np.asarray(placement.asset.size_m, dtype=np.float64) * 0.5,
                }
                held[placement.slot] = True
                released[placement.slot] = False
            assert not terminal_success(scene.spec, poses, held, released, side=scene.side), scene_path.name
            if published:
                touching = {slot: True for slot in held}
                assert not terminal_success(scene.spec, poses, held, touching, side=scene.side), scene_path.name


def test_a_pickup_counts_only_while_the_gripper_holds_it():
    """Picking an object up is not placing it. The score drops when the gripper lets go."""
    spec = spec_from_success(
        "pick",
        {"op": "picked", "object": "mug", "margin": 0.1, "hold": ["mug"], "require_grasp": []},
        40.0,
        {"mug": "item_a"},
    )
    poses = {
        "item_a": {
            "pos": np.asarray([[0.4, 0.0, 0.2]]),
            "quat": np.asarray([[1.0, 0.0, 0.0, 0.0]]),
            "half": np.asarray([0.04, 0.04, 0.04]),
        }
    }
    assert not terminal_success(spec, poses, {"item_a": True}, {"item_a": False})
    assert terminal_success(spec, poses, {"item_a": True}, {"item_a": True})
    low = {
        "item_a": {
            "pos": np.asarray([[0.4, 0.0, 0.05]]),
            "quat": np.asarray([[1.0, 0.0, 0.0, 0.0]]),
            "half": np.asarray([0.04, 0.04, 0.04]),
        }
    }
    assert not terminal_success(spec, low, {"item_a": True}, {"item_a": True})


def _scene_poses(scene) -> dict:
    poses = {}
    for placement in scene.placements:
        poses[placement.slot] = {
            "pos": np.asarray([placement.position_m], dtype=np.float64),
            "quat": np.asarray([placement.quat_wxyz], dtype=np.float64),
            "half": np.asarray(placement.asset.size_m, dtype=np.float64) * 0.5,
        }
    return poses


def test_v0_is_sixteen_home_office_oak_in_container_tasks():
    """v0 is sixteen v13 tasks on the home-office background and the oak table.

    Every success check is ``in_container``. The scenes and the meshes they name
    live in this bundle, not in v13.
    """
    bundle = Path(__file__).resolve().parents[2] / "envhub" / "environments" / "franka-tabletop-v0"
    if not bundle.is_dir():
        pytest.skip("franka-tabletop-v0 is not next to this package")
    text = (bundle / "benchmark.yaml").read_text(encoding="utf-8")
    ids = [line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("- task_id:")]
    expected = (
        ("animals_in_bin", "toys_cleanup"),
        ("apple_and_yogurt_in_bowl", "workdesk_snacks"),
        ("bigger_object", "butter_raisin_box_grey_bin"),
        ("blocks_in_bin", "toys_cleanup"),
        ("clamp_in_right_bin", "tools_container"),
        ("cubes_and_blocks_in_bin", "toys_cleanup"),
        ("foodpacking_1bin_1can", "foodpacking_1bin_1box_1can"),
        ("foodpacking_1bin_3can", "foodpacking_1bin_3box_3can"),
        ("red_dishes_in_bin", "rubiks_cube_banana_bowl_mug_bin"),
        ("red_items_in_bin", "rubiks_cube_banana_bowl_mug_bin"),
        ("rubiks_cubes_in_bin", "toys_cleanup"),
        ("smaller_object", "butter_raisin_box_grey_bin"),
        ("throw_away_apple", "workdesk_snacks"),
        ("throw_away_snacks", "workdesk_snacks"),
        ("tool_organization", "tools_container"),
        ("wood_spatula_to_bowl", "fruits_in_basket"),
    )
    assert ids == [task_id for task_id, _scene in expected]
    assert [line.strip() for line in text.splitlines() if line.strip().startswith("op:")] == ["op: in_container"] * 16
    scenes = sorted(path.name for path in (bundle / "scenes").glob("*.json"))
    assert scenes == sorted(f"{task_id}-{scene_id}.json" for task_id, scene_id in expected)
    assets = bundle / "assets"
    v13_assets = bundle.parent / "franka-tabletop-v13" / "assets"
    assert assets.is_dir()
    assert not assets.is_symlink()
    assert (assets / "stage.json").stat().st_ino != (v13_assets / "stage.json").stat().st_ino
    stage = json.loads((assets / "stage.json").read_text(encoding="utf-8"))
    home_office = resolve_background(stage, "home_office")
    assert home_office["texture"] == "assets/backgrounds/home_office.exr"
    assert resolve_surface(stage, "oak")["usd"] == "assets/fixtures/table_oak.usd"
    referenced = [stage["fixture"]["usd"], home_office["texture"], resolve_surface(stage, "oak")["usd"]]
    for task_id, scene_id in expected:
        scene = load_scene(bundle, task_id, scene_id)
        assert scene is not None
        assert scene.spec is not None
        assert scene.spec.success == "in_container"
        payload = json.loads((bundle / "scenes" / f"{task_id}-{scene_id}.json").read_text(encoding="utf-8"))
        assert payload["background"] == "home_office"
        assert payload["surface"] == "oak"
        assert payload["success"]["op"] == "in_container"
        for placement in scene.placements:
            referenced.append(placement.asset.bundle_usd())
    for relative in referenced:
        assert (bundle / relative).is_file(), relative


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
    """The four scored clips left the object where it started. Those poses score 0."""
    root = Path(__file__).resolve().parents[2] / "envhub" / "environments" / "franka-tabletop-v10"
    if not root.is_dir():
        pytest.skip("franka-tabletop-v10 is not next to this package")
    released = {}
    held = {}

    keyboard = load_scene(root, "keyboard_out_of_bin", "workdesk_bin")
    poses = _scene_poses(keyboard)
    for slot in poses:
        held[slot] = True
        released[slot] = False
    assert not terminal_success(keyboard.spec, poses, held, released, side=keyboard.side)
    # The center can leave the bin while the keyboard is still up on the rim.
    poses["keyboard"]["pos"] = np.asarray([[0.85, 0.35, 0.117]])
    assert not terminal_success(keyboard.spec, poses, held, released, side=keyboard.side)
    poses["keyboard"]["pos"] = np.asarray([[0.85, 0.35, 0.02]])
    poses["keyboard"]["quat"] = np.asarray([[1.0, 0.0, 0.0, 0.0]])
    assert terminal_success(keyboard.spec, poses, held, released, side=keyboard.side)
    assert not terminal_success(keyboard.spec, poses, held, released, side=keyboard.side, gripper_open=False)

    tower = load_scene(root, "unstack_rubiks_cube", "mug_banana_ketchup_bowl_rubiks3_bin")
    poses = _scene_poses(tower)
    for slot in poses:
        held[slot] = True
        released[slot] = False
    for name in ("rubiks_cube_top", "rubiks_cube_middle", "rubiks_cube_bottom"):
        poses[name]["quat"] = np.asarray([[0.2, 0.0, 0.0, 0.98]])
    assert not terminal_success(tower.spec, poses, held, released, side=tower.side)

    hammer = load_scene(root, "tools_picking_hammer", "tools_picking")
    poses = _scene_poses(hammer)
    for slot in poses:
        held[slot] = True
        released[slot] = False
    assert not terminal_success(hammer.spec, poses, held, released, side=hammer.side)


def test_a_sideways_pickup_uses_the_world_bottom():
    """Raising the center of a rotated cup is not a lift while a corner is still down."""
    spec = spec_from_success(
        "pick",
        {"op": "picked", "object": "cup", "margin": 0.053, "hold": ["cup"], "require_grasp": []},
        40.0,
        {"cup": "cup"},
    )
    tipped = {
        "cup": {
            "pos": np.asarray([[0.36, -0.26, 0.09]]),
            "quat": np.asarray([[0.7071, 0.0, 0.7071, 0.0]]),
            "half": np.asarray([0.0867, 0.0476, 0.0260]),
        }
    }
    held = {"cup": True}
    touching = {"cup": True}
    assert not terminal_success(spec, tipped, held, touching)
    lifted = {
        "cup": {
            "pos": np.asarray([[0.36, -0.26, 0.16]]),
            "quat": np.asarray([[0.7071, 0.0, 0.7071, 0.0]]),
            "half": np.asarray([0.0867, 0.0476, 0.0260]),
        }
    }
    assert terminal_success(spec, lifted, held, touching)


def test_each_family_places_without_overlap():
    for task_id in FAMILIES:
        for seed in PUBLIC_SEEDS:
            scene = _compose(seed, task_id, pose_jitter_m=0.02)
            points = [placement.position_m for placement in scene.placements]
            for left in range(len(points)):
                for right in range(left + 1, len(points)):
                    distance = np.linalg.norm(np.array(points[left][:2]) - np.array(points[right][:2]))
                    assert distance > 0.05


def test_compose_does_not_start_already_solved():
    for task_id in ("place_in_container", "place_relative"):
        spec = get_family(task_id)
        for seed in range(20):
            scene = _compose(seed, task_id)
            poses = {
                placement.slot: {
                    "pos": np.asarray([placement.position_m], dtype=np.float64),
                    "quat": np.array([[1.0, 0.0, 0.0, 0.0]]),
                    "half": np.asarray(placement.asset.size_m, dtype=np.float64) * 0.5,
                }
                for placement in scene.placements
            }
            assert not bool(evaluate(spec, poses, side=scene.side).reshape(-1)[0])


def test_terminal_success_requires_a_grasp_and_a_release():
    spec = get_family("place_in_container")
    origin = np.array([[0.0, 0.0, 0.0]])
    quat = np.array([[1.0, 0.0, 0.0, 0.0]])
    poses = {
        "item_a": {"pos": origin, "quat": quat, "half": np.ones(3)},
        "container": {"pos": origin, "quat": quat, "half": np.array([0.2, 0.2, 0.2])},
    }
    assert bool(evaluate(spec, poses)[0])
    assert not terminal_success(spec, poses, {"item_a": False}, {"item_a": False})
    assert not terminal_success(spec, poses, {"item_a": True}, {"item_a": True})
    assert terminal_success(spec, poses, {"item_a": True}, {"item_a": False})


def test_place_in_container_predicate_matches_a_centered_item():
    spec = get_family("place_in_container")
    origin = np.array([[0.0, 0.0, 0.0]])
    quat = np.array([[1.0, 0.0, 0.0, 0.0]])
    poses = {
        "item_a": {"pos": origin, "quat": quat, "half": np.ones(3)},
        "container": {"pos": origin, "quat": quat, "half": np.array([0.2, 0.2, 0.2])},
    }
    assert bool(evaluate(spec, poses)[0])


def test_hidden_salt_changes_the_verb():
    names = {"item": "red cup", "container": "wood bin"}
    assert phrase("place_in_container", 0, names, salt="public") != phrase(
        "place_in_container", 0, names, salt="phase-hidden"
    )


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
    assert scene.names()["item"] == "custom widget"
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
        "success": {"kind": "item_in_container", "roles": {"item": "banana", "container": "bowl"}},
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
    assert scene.spec.success == "item_in_container"
    assert scene.spec.roles["item"] == "item_a"
    assert scene.spec.roles["container"] == "container"
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

    Each mesh bottom sits 3 mm above the oak top. The old heights put the BBQ bottle and the mug
    inside the top and left the ranch bottle floating.
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
        assert min(heights) == pytest.approx(top_z + 0.003, abs=1e-4), item["name"]


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


def test_compose_places_only_catalog_objects(tmp_path: Path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "catalog.json").write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "name": "custom_widget",
                        "kind": "item",
                        "size_m": [0.05, 0.05, 0.04],
                        "usd": "assets/objects/w.usd",
                    },
                    {
                        "name": "custom_bin",
                        "kind": "container",
                        "size_m": [0.2, 0.2, 0.08],
                        "usd": "assets/objects/b.usd",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    scene = compose(
        3,
        "place_in_container",
        read_catalog(tmp_path),
        table_m=(1.1, 0.7, 0.04),
        center_xy=(0.35, 0.0),
    )
    assert {placement.asset.name for placement in scene.placements} == {"custom_widget", "custom_bin"}


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


def test_brain_cfg_keeps_cameras_out_of_the_per_step_observation_group():
    pytest.importorskip("isaaclab_physx")
    from franka_tabletop.core.observation_spec import OBSERVATION_CAMERAS
    from franka_tabletop.env_cfg import build_cfg

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


def _pose(position: list[float], half: list[float] | None = None) -> dict:
    return {
        "pos": np.asarray([position], dtype=np.float64),
        "quat": np.array([[1.0, 0.0, 0.0, 0.0]]),
        "half": np.asarray(half or [0.2, 0.2, 0.2], dtype=np.float64),
    }


def test_custom_slot_names_are_not_a_fixed_list(tmp_path: Path):
    scene_dir = tmp_path / "scenes"
    scene_dir.mkdir()
    payload = {
        "instruction": "Put the cube in the bin",
        "success": {"kind": "item_in_container", "roles": {"item": "cube", "container": "bin"}},
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
    assert scene.spec.roles["item"] == "cube"
    assert scene.spec.check is not None
    assert scene.spec.check.grasp == ("cube",)


def test_all_averages_object_scores_and_any_takes_the_max():
    names = _names()
    poses = {"cube": _pose([0.0, 0.0, 0.0]), "block": _pose([1.0, 0.0, 0.0]), "bin": _pose([0.0, 0.0, 0.0])}
    held = {"cube": True, "block": False}
    released = {"cube": False, "block": True}
    shared = {"op": "in_container", "object": ["cube", "block"], "container": "bin"}
    both = spec_from_success("pack", {**shared, "logical": "all"}, 40.0, names)
    either = spec_from_success("pack", {**shared, "logical": "any"}, 40.0, names)
    score, latched, _ = task_progress(both, poses, held, released, {}, [])
    assert abs(score - 0.5) < 1e-9
    assert not terminal_success(both, poses, held, released)
    nudged = {"cube": _pose([1.0, 0.0, 0.0]), "block": _pose([1.0, 0.0, 0.0]), "bin": _pose([0.0, 0.0, 0.0])}
    kept, _, _ = task_progress(both, nudged, held, released, latched, [])
    assert abs(kept - 0.5) < 1e-9
    either_score, _, _ = task_progress(either, poses, held, released, {}, [])
    assert either_score == 1.0


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
    held = {"cube": True, "block": True, "can": True}
    free = {"cube": False, "block": False, "can": False}
    score, _, _ = task_progress(spec, poses, held, free, {}, [])
    assert abs(score - 1.0 / 3.0) < 1e-9
    poses["block"] = _pose([0.0, 0.0, 0.0])
    poses["can"] = _pose([0.0, 0.0, 0.0])
    done, _, _ = task_progress(spec, poses, held, free, {}, [])
    assert done == 1.0


def test_in_order_follows_the_first_entry_sequence():
    names = _names()
    spec = spec_from_success(
        "pack",
        {"op": "in_order", "object": ["cube", "block"], "container": "bin"},
        40.0,
        names,
    )
    held = {"cube": True, "block": True}
    free = {"cube": False, "block": False}
    bin_pose = _pose([0.0, 0.0, 0.0])
    outside = _pose([1.0, 0.0, 0.0])
    inside = _pose([0.0, 0.0, 0.0])
    first = {"cube": outside, "block": inside, "bin": bin_pose}
    score, latched, entered = task_progress(spec, first, held, free, {}, [])
    assert entered == ["block"]
    assert score == 0.0
    second = {"cube": inside, "block": inside, "bin": bin_pose}
    score, _, entered = task_progress(spec, second, held, free, latched, entered)
    assert entered == ["block", "cube"]
    assert abs(score - 0.5) < 1e-9
    ordered = spec_from_success(
        "pack",
        {"op": "in_order", "object": ["cube", "block"], "container": "bin"},
        40.0,
        names,
    )
    cube_first = {"cube": inside, "block": outside, "bin": bin_pose}
    score, latched, entered = task_progress(ordered, cube_first, held, free, {}, [])
    assert entered == ["cube"]
    assert abs(score - 0.5) < 1e-9
    score, latched, entered = task_progress(ordered, second, held, free, latched, entered)
    assert entered == ["cube", "block"]
    assert score == 1.0
    left = {"cube": outside, "block": outside, "bin": bin_pose}
    kept, _, _ = task_progress(ordered, left, held, free, latched, entered)
    assert kept == 1.0


def test_outside_needs_every_corner_clear_and_does_not_stick():
    names = _names()
    spec = spec_from_success(
        "take",
        {"op": "outside", "object": ["cube"], "container": "bin", "require_grasp": ["cube"]},
        40.0,
        names,
    )
    bowl = _pose([0.0, 0.0, 0.0], [0.05, 0.05, 0.03])
    # Two corners stay in the bowl. The old half-out rule counted this as outside.
    spoon = _pose([0.06, 0.0, 0.08], [0.05, 0.01, 0.03])
    held = {"cube": True}
    free = {"cube": False}
    poses = {"cube": spoon, "bin": bowl}
    score, latched, _ = task_progress(spec, poses, held, free, {}, [], gripper_open=True)
    assert score == 0.0
    assert latched == {}

    clear = {"cube": _pose([1.0, 0.0, 0.0], [0.05, 0.01, 0.03]), "bin": bowl}
    score, latched, _ = task_progress(spec, clear, held, free, {}, [], gripper_open=True)
    assert score == 1.0
    assert task_progress(spec, clear, held, free, latched, [], gripper_open=True)[0] == 1.0
    touching = {"cube": True}
    assert task_progress(spec, clear, held, touching, latched, [], gripper_open=True)[0] == 0.0
    back = {"cube": _pose([0.0, 0.0, 0.0], [0.01, 0.01, 0.01]), "bin": bowl}
    assert task_progress(spec, back, held, free, {}, [], gripper_open=True)[0] == 0.0

    # Authored measuring spoon: tilted in the red bowl, gripper already released.
    spoon_pose = _pose(
        [0.332945, -0.137862, 0.158783],
        [0.0837055, 0.0413945, 0.0189455],
    )
    spoon_pose["quat"] = np.array([[0.631510, -0.629371, 0.306461, 0.333421]])
    bowl_pose = _pose(
        [0.357784, -0.157000, 0.077346],
        [0.080722, 0.0805565, 0.027485],
    )
    bowl_pose["quat"] = np.array([[0.9999995827674866, -0.000510019832290709, -0.0007578271324746311, 0.0]])
    authored = {"cube": spoon_pose, "bin": bowl_pose}
    assert task_progress(spec, authored, held, free, {}, [], gripper_open=True)[0] == 0.0


def test_on_top_uses_the_support_footprint():
    support = np.array([[0.0, 0.0, 0.0]])
    half = np.array([0.10, 0.08, 0.02])
    assert bool(on_footprint(np.array([[0.05, 0.0, 0.04]]), support, half)[0])
    assert not bool(on_footprint(np.array([[0.20, 0.0, 0.04]]), support, half)[0])
    assert not bool(on_footprint(np.array([[0.0, 0.0, -0.01]]), support, half)[0])
    spec = spec_from_success("stack", {"op": "on_top", "object": "lid", "support": "tray"}, 40.0, _names())
    poses = {"lid": _pose([0.0, 0.0, 0.05], [0.04, 0.04, 0.01]), "tray": _pose([0.0, 0.0, 0.0], [0.1, 0.1, 0.02])}
    assert bool(evaluate(spec, poses)[0])
    assert terminal_success(spec, poses, {"lid": True}, {"lid": False})
    assert not terminal_success(spec, poses, {"lid": True}, {"lid": True})


def test_inverse_kinematics_reduces_position_error():
    target = fk(np.zeros(7)) + np.array([0.02, 0.0, 0.0])
    before = np.linalg.norm(fk(np.zeros(7)) - target)
    after = np.linalg.norm(fk(ik(target, seed=np.zeros(7))) - target)
    assert after < before


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
