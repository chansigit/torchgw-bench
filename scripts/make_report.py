#!/usr/bin/env python
"""torchgw-bench reporter v1: aggregate results/*.json into docs markdown.

This script is stand-alone. It does NOT import from tracks/ and is NOT
imported by any track. It reads JSON records, groups them by track, and
renders per-tier markdown pages.
"""
from __future__ import annotations

__all__ = [
    "load_results",
    "group_by_track",
    "render_track_section",
    "render_docs_markdown",
    "main",
]

import json
from pathlib import Path


def load_results(results_dir: Path) -> list[dict]:
    """Scan a results directory for JSON records and return them as a list.

    Lenient: files that are not .json or that fail to parse are silently
    skipped. Missing schema fields are preserved as-is (the renderer handles
    absences with dict.get()).
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return []
    records: list[dict] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            records.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return records


from collections import defaultdict


def group_by_track(records: list[dict]) -> dict[str, list[dict]]:
    """Bucket records by their 'track' key, with deterministic sorted order."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r.get("track", "(unknown)")].append(r)
    return {k: groups[k] for k in sorted(groups.keys())}


def _fmt(value, spec: str = "") -> str:
    """Formatting helper that returns '—' for None/missing values."""
    if value is None:
        return "—"
    try:
        return format(value, spec) if spec else str(value)
    except (TypeError, ValueError):
        return str(value)


def render_track_section(track: str, records: list[dict]) -> str:
    """Render one markdown section for a single track."""
    lines: list[str] = []

    tier = track.split("/", 1)[0].capitalize() if "/" in track else "Unknown"
    lines.append(f"## {tier} Track: `{track}`")
    lines.append("")

    ds_name = next(
        (r.get("dataset", {}).get("name") for r in records if r.get("dataset", {}).get("name")),
        None,
    )
    if ds_name:
        lines.append(f"**Dataset:** `{ds_name}`")
    n_source = next(
        (r.get("dataset", {}).get("n_source") for r in records if r.get("dataset", {}).get("n_source")),
        None,
    )
    n_target = next(
        (r.get("dataset", {}).get("n_target") for r in records if r.get("dataset", {}).get("n_target")),
        None,
    )
    if n_source and n_target:
        lines.append(f"**Scale:** N={n_source}, K={n_target}")
    lines.append("")

    host_gpu = next(
        (r.get("host", {}).get("gpu") for r in records
         if r.get("status") == "ok" and r.get("host", {}).get("gpu")),
        None,
    )
    if host_gpu:
        lines.append(f"**Host:** `{host_gpu}`")
        lines.append("")

    lines.append("| Solver | Status | GW cost | Spearman | Wall (s) | GPU peak (GB) | Iterations |")
    lines.append("|---|:---:|---:|---:|---:|---:|---:|")
    for r in records:
        solver = r.get("solver", "?")
        status = r.get("status", "?")
        metrics = r.get("metrics", {}) or {}
        correctness = metrics.get("correctness", {}) or {}
        task = metrics.get("task", {}) or {}
        efficiency = metrics.get("efficiency", {}) or {}

        status_cell = {"ok": "✓", "fail": "✗ FAIL", "skip": "⊘ skip"}.get(status, status)

        row = (
            f"| `{solver}` | {status_cell} | "
            f"{_fmt(correctness.get('gw_cost'), '.4f')} | "
            f"{_fmt(task.get('spearman_arclen'), '.4f')} | "
            f"{_fmt(efficiency.get('wall_s'), '.2f')} | "
            f"{_fmt(efficiency.get('gpu_peak_gb'), '.2f')} | "
            f"{_fmt(efficiency.get('iterations'))} |"
        )
        lines.append(row)

        if status == "fail" and r.get("error"):
            lines.append(f"|     | error: `{r['error']}` |||||||")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    raise NotImplementedError("main() is implemented in Task 14")


if __name__ == "__main__":
    main()
