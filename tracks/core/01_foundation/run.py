#!/usr/bin/env python
"""Track: core/01_foundation — spiral → Swiss roll GW alignment.

Phase 1 scope: N=400, K=500 only; solvers torchgw-landmark and pot-entropic.

This file is self-contained. It does NOT import from any sibling track or
from scripts/. Helper functions defined here are unit-tested by
tests/test_run.py via a sys.path hook in tests/conftest.py.
"""
from __future__ import annotations

__all__ = [
    "sample_spiral",
    "sample_swiss_roll",
    "arclen_spearman",
    "get_host_info",
    "build_record",
    "run_torchgw_landmark",
    "run_pot_entropic",
    "main",
]


def main() -> None:
    raise NotImplementedError("main() is implemented in a later task")


if __name__ == "__main__":
    main()
