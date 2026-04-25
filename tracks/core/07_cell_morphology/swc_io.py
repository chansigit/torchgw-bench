"""Thin SWC reader. Returns (n, 7) float64 array of [id, type, x, y, z, r, parent]."""
from __future__ import annotations
import pathlib
import numpy as np


def read_swc(path: str | pathlib.Path) -> np.ndarray:
    rows = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 7:
                continue
            rows.append([float(p) for p in parts[:7]])
    if not rows:
        raise ValueError(f"no node rows in {path}")
    return np.asarray(rows, dtype=np.float64)
