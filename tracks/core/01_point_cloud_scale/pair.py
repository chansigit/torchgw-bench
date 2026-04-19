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

    R_gt: np.ndarray = Rotation.random(random_state=seed + 1).as_matrix().astype(np.float32)

    target: np.ndarray = (source @ R_gt.T).astype(np.float32)

    return source, target, R_gt
