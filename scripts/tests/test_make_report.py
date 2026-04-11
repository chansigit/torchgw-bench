"""Unit tests for scripts/make_report.py helpers."""
from __future__ import annotations

import json
from pathlib import Path

import make_report  # noqa: E402  — sys.path set by conftest.py


# ---- load_results -------------------------------------------------------

def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def test_load_results_empty_dir_returns_empty_list(tmp_path: Path):
    assert make_report.load_results(tmp_path) == []


def test_load_results_reads_all_json(tmp_path: Path):
    _write(tmp_path / "core_01_foundation__torchgw__seed0.json",
           {"track": "core/01_foundation", "solver": "torchgw", "seed": 0, "status": "ok"})
    _write(tmp_path / "core_01_foundation__pot__seed0.json",
           {"track": "core/01_foundation", "solver": "pot", "seed": 0, "status": "ok"})
    recs = make_report.load_results(tmp_path)
    assert len(recs) == 2
    solvers = sorted(r["solver"] for r in recs)
    assert solvers == ["pot", "torchgw"]


def test_load_results_ignores_non_json(tmp_path: Path):
    (tmp_path / "README.md").write_text("not json")
    _write(tmp_path / "core_01_foundation__torchgw__seed0.json",
           {"track": "core/01_foundation", "solver": "torchgw", "status": "ok"})
    assert len(make_report.load_results(tmp_path)) == 1


def test_load_results_skips_broken_json(tmp_path: Path):
    (tmp_path / "broken.json").write_text("{not valid json")
    _write(tmp_path / "core_01_foundation__torchgw__seed0.json",
           {"track": "core/01_foundation", "status": "ok"})
    recs = make_report.load_results(tmp_path)
    assert len(recs) == 1
