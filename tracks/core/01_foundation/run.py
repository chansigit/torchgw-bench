#!/usr/bin/env python
"""Track: core/01_foundation — spiral → Swiss roll GW alignment.

Phase 1 scope: N=400, K=500 only; solvers torchgw-landmark and pot-entropic.

This file is self-contained. It does NOT import from any sibling track or
from scripts/. Helper functions defined here are unit-tested by
tests/test_run.py via a sys.path hook in tests/conftest.py.

Note: this stub will be filled in by Tasks 5–10 of the Phase 1 plan. The
``__all__`` export list is deliberately omitted in the stub and will be
added in Task 10 once all helpers exist.
"""
from __future__ import annotations

import numpy as np


def sample_spiral(n: int, noise: float = 0.05, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """2D Archimedean spiral with Gaussian noise.

    Returns:
        points: (n, 2) float32 array
        angles: (n,) float64 array, the parameter used to generate each point
    """
    rng = np.random.default_rng(seed)
    radius = np.linspace(0.3, 1.0, n)
    angles = np.linspace(0, 9, n)
    eps = rng.normal(size=(2, n)) * noise
    x = (radius + eps[0]) * np.cos(angles)
    y = (radius + eps[1]) * np.sin(angles)
    points = np.stack((x, y), axis=1).astype(np.float32)
    return points, angles


def sample_swiss_roll(n: int, noise: float = 0.05, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """3D Swiss roll parameterised by the same angular schedule as the spiral.

    Returns:
        points: (n, 3) float32 array
        angles: (n,) float64 array
    """
    rng = np.random.default_rng(seed)
    radius = np.linspace(0.3, 1.0, n)
    angles = np.linspace(0, 9, n)
    eps = rng.normal(size=(2, n)) * noise
    x = (radius + eps[0]) * np.cos(angles)
    y = (radius + eps[1]) * np.sin(angles)
    z = rng.uniform(size=n)
    points = np.stack((x, z, y), axis=1).astype(np.float32)
    return points, angles


def main() -> None:
    raise NotImplementedError("main() is implemented in a later task")


if __name__ == "__main__":
    main()
