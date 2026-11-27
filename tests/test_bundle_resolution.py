# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for local/cached bundle resolution in franka_tabletop.scenes.bundle."""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "franka_tabletop"))

from franka_tabletop.scenes.bundle import resolve_bundle_path


def _write_manifest(bundle_dir: Path, env_id: str = "franka-tabletop-v11") -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "manifest.yaml").write_text(f"id: {env_id}\nversion: 1.0.0\n", encoding="utf-8")
    return bundle_dir


def test_resolve_bundle_path_prefers_explicit_then_env_then_unidroid_cache(tmp_path: Path, monkeypatch):
    # Ensure nepher is never imported when local/cached bundles exist.
    monkeypatch.setitem(sys.modules, "nepher", None)
    monkeypatch.delenv("ISAAC_EVAL_BUNDLE_DIR", raising=False)

    cache_root = tmp_path / "bench-cache"
    monkeypatch.setenv("UNIDROID_CACHE_DIR", str(cache_root))

    # 1. Unidroid bench-cache with wrapped single top-level folder
    cached_inner = _write_manifest(cache_root / "franka-s1-pub" / "bundle" / "franka-tabletop-v11")
    (cache_root / "franka-s1-pub" / "bundle" / "__MACOSX").mkdir(parents=True, exist_ok=True)
    assert Path(resolve_bundle_path("franka-s1-pub")) == cached_inner.resolve()

    # 2. ISAAC_EVAL_BUNDLE_DIR overrides cache
    env_bundle = _write_manifest(tmp_path / "env-bundle")
    monkeypatch.setenv("ISAAC_EVAL_BUNDLE_DIR", str(env_bundle))
    assert Path(resolve_bundle_path("franka-s1-pub")) == env_bundle.resolve()

    # 3. Explicit bundle_path overrides ISAAC_EVAL_BUNDLE_DIR
    explicit_bundle = _write_manifest(tmp_path / "explicit" / "wrapped")
    assert Path(resolve_bundle_path("franka-s1-pub", tmp_path / "explicit")) == explicit_bundle.resolve()


def test_resolve_bundle_path_falls_back_to_nepher_when_uncached(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("ISAAC_EVAL_BUNDLE_DIR", raising=False)
    monkeypatch.setenv("UNIDROID_CACHE_DIR", str(tmp_path / "empty-cache"))

    fallback_dir = tmp_path / "nepher-cache" / "franka-tabletop-v11"
    fallback_dir.mkdir(parents=True)
    fake_nepher = types.ModuleType("nepher")
    calls: list[tuple[str, str]] = []

    def _fake_load_env(env_id: str, category: str = "manipulation"):
        calls.append((env_id, category))
        return types.SimpleNamespace(cache_path=fallback_dir)

    fake_nepher.load_env = _fake_load_env
    monkeypatch.setitem(sys.modules, "nepher", fake_nepher)

    assert Path(resolve_bundle_path("franka-tabletop-v11")) == fallback_dir
    assert calls == [("franka-tabletop-v11", "manipulation")]
    assert resolve_bundle_path(None) == ""
