# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fail when a published file contains a blocked token.

The blocked tokens are stored as sha256 digests so this file does not reveal them.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

DENYLIST = {
    "641b26b6c6f8a09ce6ef3dac54768fe25517296ad6a5af7660b2726132d5bea4",
    "fab16132ae80815d3eec988aab6eae703c71db0fd0cae9b2b8e02dae2383c671",
    "2014e7b6016a99da47d622d416fe305d2651f3a7730a6b0cb640344a486808b9",
    "40a3edd1943cfb35de99107ac535dc0b98f716b862da72b3d8ca0ec4da3610b6",
    "af9aadb68121b1c2766c181a808b5c55294854e0065129e2ea639bf5984c5a28",
    "524b35430349cfa253f5ad8fc6622a74f0c19fef62a7d4335452466078f38656",
    "1352fc379f25acc10cd92617e4506564c07dacc7b56e69ca1790635a5c8433e4",
    "e9925cf52a3eaadff9d26585177dd9845fabc278e0536538ca8d18fa08c012b1",
}

_TOKEN = __import__("re").compile(r"[A-Za-z0-9_]+")
_TEXT = {".py", ".yaml", ".yml", ".md", ".json", ".sh", ".toml"}

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "franka_tabletop"))


def scan(paths: list[Path]) -> list[str]:
    hits = []
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in _TEXT:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in _TOKEN.findall(text):
            if hashlib.sha256(token.lower().encode("utf-8")).hexdigest() in DENYLIST:
                hits.append(f"{path}: {token}")
                break
    return hits


def test_published_files_and_bundle(tmp_path: Path):
    yaml = pytest.importorskip("yaml")
    files = [path for path in ROOT.rglob("*") if path.is_file() and "tests" not in path.parts]
    eval_config = ROOT.parent / "eval-nav" / "configs" / "task-franka-tabletop.yaml"
    if eval_config.is_file():
        files.append(eval_config)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "sample.json").write_text(
        json.dumps({"names": {"item": "block", "container": "bin"}}),
        encoding="utf-8",
    )
    (bundle / "benchmark.yaml").write_text(
        yaml.safe_dump({"version": "franka-tabletop-v1", "task_id": "place_in_container"}),
        encoding="utf-8",
    )
    files.extend(bundle.rglob("*"))
    assert scan(files) == []
