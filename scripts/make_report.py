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


def main() -> None:
    raise NotImplementedError("main() is implemented in Task 14")


if __name__ == "__main__":
    main()
