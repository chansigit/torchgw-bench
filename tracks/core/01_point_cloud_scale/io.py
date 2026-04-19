from __future__ import annotations

"""C1 point-cloud scale track — OFF mesh parser and farthest-point sampling."""

import numpy as np


def read_off(path: str) -> np.ndarray:
    """Parse a ModelNet .off mesh file; return (M, 3) vertex array (float32).

    OFF format
    ----------
    Standard::

        OFF
        <nv> <nf> 0
        x0 y0 z0
        ...

    Quirky ModelNet variant (header and counts concatenated on line 1)::

        OFF<nv> <nf> 0
        x0 y0 z0
        ...

    Face lines are skipped entirely.

    Parameters
    ----------
    path:
        Path to an ``.off`` file.

    Returns
    -------
    vertices:
        Float32 array of shape ``(nv, 3)``.
    """
    with open(path, "r") as fh:
        lines = fh.readlines()

    idx = 0
    first = lines[idx].strip()

    # Handle the "OFF<nv> <nf> 0" quirk
    if first.upper().startswith("OFF") and len(first) > 3:
        counts_str = first[3:].strip()
        idx += 1
    else:
        # Standard: line 0 is "OFF", line 1 is "<nv> <nf> 0"
        idx += 1
        counts_str = lines[idx].strip()
        idx += 1

    parts = counts_str.split()
    nv = int(parts[0])

    vertices: list[list[float]] = []
    for _ in range(nv):
        row = lines[idx].split()
        vertices.append([float(row[0]), float(row[1]), float(row[2])])
        idx += 1

    return np.array(vertices, dtype=np.float32)


def fps_downsample(points: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Farthest-point sampling (FPS).

    Selects *n* points from *points* such that consecutive selections
    maximise the minimum distance to the already-selected set.  O(n * N)
    time; fine for N <= 100 k, n <= 100 k.

    Parameters
    ----------
    points:
        Float array of shape ``(N, 3)``.
    n:
        Number of points to select.
    seed:
        RNG seed used to pick the initial point.

    Returns
    -------
    selected:
        Float32 array of shape ``(min(n, N), 3)``.  If ``n >= N`` all
        points are returned unchanged (no upsampling).
    """
    points = np.asarray(points, dtype=np.float32)
    N = len(points)
    if n >= N:
        return points.copy()

    rng = np.random.default_rng(seed)
    selected = np.empty((n, 3), dtype=np.float32)

    # Index of the initial (random) point
    start = int(rng.integers(0, N))
    selected[0] = points[start]

    # min-distance-to-selected-set for every point
    min_dists = np.sum((points - selected[0]) ** 2, axis=1)  # squared L2

    for i in range(1, n):
        # Pick the point furthest from the current selected set
        farthest = int(np.argmax(min_dists))
        selected[i] = points[farthest]
        # Update min-distance array
        new_dists = np.sum((points - selected[i]) ** 2, axis=1)
        np.minimum(min_dists, new_dists, out=min_dists)

    return selected
