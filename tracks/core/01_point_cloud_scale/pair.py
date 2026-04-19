from __future__ import annotations

"""C1 point-cloud scale track — rotation-based pair generation."""

import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Load c1_io (io.py) from same directory without shadowing stdlib io
_spec = importlib.util.spec_from_file_location(
    "c1_io", Path(__file__).parent / "io.py"
)
_c1_io = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_c1_io)  # type: ignore[union-attr]


def make_pair(
    P: np.ndarray, n: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create (source, target, R_gt) where target = source @ R_gt.T.

    Pipeline
    --------
    1. FPS-downsample *P* to *n* points using the given *seed*.
    2. Generate a uniform random rotation via ``Rotation.random(random_state=seed+1)``.
    3. ``target = source @ R_gt.T``

    Identity index correspondence: ``source[i] <-> target[i]``.

    Parameters
    ----------
    P:
        Input point cloud, shape ``(N, 3)``.
    n:
        Number of points to keep after FPS downsampling.
    seed:
        Determinism seed used for both FPS and rotation sampling.

    Returns
    -------
    source : np.ndarray, shape (n, 3), float32
    target : np.ndarray, shape (n, 3), float32
    R_gt   : np.ndarray, shape (3, 3), float32
    """
    source: np.ndarray = _c1_io.fps_downsample(P, n, seed=seed)

    R_gt: np.ndarray = Rotation.random(
        rng=np.random.default_rng(seed + 1)
    ).as_matrix().astype(np.float32)

    target: np.ndarray = (source @ R_gt.T).astype(np.float32)

    return source, target, R_gt


def make_synthetic_spiral_pair(
    n: int, seed: int, noise_std: float = 0.0, n_turns: float = 4.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3D random Gaussian cloud (originally an asymmetric helix — replaced
    2026-04-19 because the helix has residual rotational symmetry along its
    axis, making identity correspondence non-unique under SO(3) targets).

    Source: N points ~ 3D Gaussian N(0, I) (random scatter, no structure).
    Target: R · source + optional Gaussian noise.
    GT correspondence: identity (source[i] <-> target[i]).

    Why Gaussian scatter not spiral: random 3D clouds have no exact symmetries
    almost surely → identity is the unique GW optimum. Spiral/torus/helix all
    have rotational or cyclic symmetries that make identity only one of many
    equivalent solutions, breaking the P@1 metric.

    Parameters
    ----------
    n:
        Point count.
    seed:
        Determinism seed for rotation and noise.
    noise_std:
        Per-coordinate Gaussian noise added to target (default 0.0 = exact
        rotation).  Noise > 0 stress-tests torchgw's robustness.
    n_turns:
        Number of spiral turns (default 4).

    Returns
    -------
    source : np.ndarray (n, 3) float32
    target : np.ndarray (n, 3) float32
    R_gt   : np.ndarray (3, 3) float32
    """
    t = np.linspace(0.0, 2.0 * np.pi * n_turns, n, dtype=np.float64)
    s = t / (2.0 * np.pi * n_turns)            # 0 → 1 along spiral
    r = 1.0 + 0.5 * s                           # radius grows linearly
    source = np.stack(
        [r * np.cos(t), r * np.sin(t), s], axis=1
    ).astype(np.float32)

    R_gt = Rotation.random(
        rng=np.random.default_rng(seed + 1)
    ).as_matrix().astype(np.float32)
    target = source @ R_gt.T

    if noise_std > 0:
        rng_noise = np.random.default_rng(seed + 7)
        target = target + rng_noise.normal(
            0.0, noise_std, target.shape
        ).astype(np.float32)

    return source, target.astype(np.float32), R_gt
