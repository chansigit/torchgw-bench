"""Unit tests for scripts/make_report.py helpers."""
from __future__ import annotations

import json
from pathlib import Path

import make_report  # type: ignore[import-not-found]  # noqa: E402  — sys.path set by conftest.py


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


# ---- group_by_track -----------------------------------------------------

def test_group_by_track_bucketizes_records():
    records = [
        {"track": "core/01_foundation", "solver": "torchgw-landmark"},
        {"track": "core/01_foundation", "solver": "pot-entropic"},
        {"track": "core/05_tu_graph", "solver": "torchgw-precomputed"},
    ]
    groups = make_report.group_by_track(records)
    assert set(groups.keys()) == {"core/01_foundation", "core/05_tu_graph"}
    assert len(groups["core/01_foundation"]) == 2
    assert len(groups["core/05_tu_graph"]) == 1


def test_group_by_track_sorts_keys_deterministically():
    records = [
        {"track": "core/05_tu_graph"},
        {"track": "core/01_foundation"},
        {"track": "core/01_foundation"},
    ]
    groups = make_report.group_by_track(records)
    assert list(groups.keys()) == ["core/01_foundation", "core/05_tu_graph"]


# ---- render_track_section -----------------------------------------------

def test_render_track_section_header_and_table_columns():
    records = [
        {
            "track": "core/01_foundation",
            "solver": "torchgw-landmark",
            "seed": 0,
            "status": "ok",
            "dataset": {"name": "spiral_400_swissroll_500", "n_source": 400, "n_target": 500},
            "host": {"gpu": "NVIDIA H100 80GB"},
            "metrics": {
                "correctness": {"gw_cost": 0.0234, "marginal_error": 1.2e-6},
                "task": {"spearman_arclen": 0.9993},
                "efficiency": {"wall_s": 1.04, "gpu_peak_gb": 0.7, "iterations": 218},
            },
        },
        {
            "track": "core/01_foundation",
            "solver": "pot-entropic",
            "seed": 0,
            "status": "ok",
            "dataset": {"name": "spiral_400_swissroll_500", "n_source": 400, "n_target": 500},
            "host": {"gpu": "cpu"},
            "metrics": {
                "correctness": {"gw_cost": 0.0251, "marginal_error": 3.4e-7},
                "task": {"spearman_arclen": 0.9984},
                "efficiency": {"wall_s": 8.9, "gpu_peak_gb": None, "iterations": 500},
            },
        },
    ]
    md = make_report.render_track_section("core/01_foundation", records)
    assert "## Core Track: `core/01_foundation`" in md
    assert "spiral_400_swissroll_500" in md
    assert "Solver" in md
    assert "Spearman" in md
    assert "Wall (s)" in md
    assert "torchgw-landmark" in md
    assert "pot-entropic" in md


def test_render_track_section_handles_missing_metrics():
    records = [
        {
            "track": "core/01_foundation",
            "solver": "broken-baseline",
            "status": "fail",
            "error": "oh no",
            "dataset": {"name": "spiral_400_swissroll_500"},
            "metrics": {},
        },
    ]
    md = make_report.render_track_section("core/01_foundation", records)
    assert "broken-baseline" in md
    assert "fail" in md.lower() or "✗" in md or "FAIL" in md


# ---- render_docs_markdown -----------------------------------------------

def test_render_docs_markdown_core_tier_only_core_records():
    records = [
        {"track": "core/01_foundation", "solver": "torchgw", "status": "ok",
         "dataset": {"name": "spiral_400_swissroll_500", "n_source": 400, "n_target": 500},
         "host": {"gpu": "NVIDIA H100 80GB"},
         "metrics": {"correctness": {"gw_cost": 0.02}, "task": {"spearman_arclen": 0.99},
                     "efficiency": {"wall_s": 1.0, "gpu_peak_gb": 0.7, "iterations": 200}}},
        {"track": "extended/01_spatial_omics", "solver": "torchgw", "status": "ok",
         "metrics": {}},
    ]
    md = make_report.render_docs_markdown(records, tier="core")
    assert "# Core tier benchmark" in md or "# Core Tier" in md.lower()
    assert "core/01_foundation" in md
    assert "extended/01_spatial_omics" not in md


def test_render_docs_markdown_empty_records_still_renders_header():
    md = make_report.render_docs_markdown([], tier="core")
    assert "Core" in md
    assert isinstance(md, str)


# ---- multi-scale table --------------------------------------------------

def test_render_track_section_shows_scale_column_when_multiple_scales():
    records = [
        {
            "track": "core/01_foundation",
            "solver": "torchgw-landmark",
            "status": "ok",
            "dataset": {"name": "spiral_400_swissroll_500", "n_source": 400, "n_target": 500},
            "metrics": {"correctness": {"gw_cost": 0.001}, "task": {"spearman_arclen": 0.999},
                        "efficiency": {"wall_s": 7.1, "gpu_peak_gb": 0.04, "iterations": 300}},
        },
        {
            "track": "core/01_foundation",
            "solver": "torchgw-landmark",
            "status": "ok",
            "dataset": {"name": "spiral_4000_swissroll_5000", "n_source": 4000, "n_target": 5000},
            "metrics": {"correctness": {"gw_cost": 0.002}, "task": {"spearman_arclen": 0.998},
                        "efficiency": {"wall_s": 12.3, "gpu_peak_gb": 0.5, "iterations": 280}},
        },
    ]
    md = make_report.render_track_section("core/01_foundation", records)
    # Scale column header must appear
    assert "N×K" in md or "Scale" in md
    # Both scale values must appear as data
    assert "400×500" in md or "400" in md
    assert "4000×5000" in md or "4000" in md


def test_render_track_section_skip_record_shows_skip_status():
    records = [
        {
            "track": "core/01_foundation",
            "solver": "pot-entropic",
            "status": "skip",
            "error": "skipped: POT O(N²) memory guard (max(N,K)=12000 > 5000)",
            "dataset": {"name": "spiral_10000_swissroll_12000", "n_source": 10000, "n_target": 12000},
            "metrics": {},
        },
    ]
    md = make_report.render_track_section("core/01_foundation", records)
    assert "pot-entropic" in md
    # skip status marker must appear
    assert "skip" in md.lower() or "⊘" in md
