#!/usr/bin/env python
"""C8 brain-alignment benchmark — one (resolution, solver, pair, seed) cell."""
from __future__ import annotations
import argparse
import datetime as _dt
import json
import os
import pathlib
import socket
import sys
import time
import numpy as np
import nibabel as nb

TRACK = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TRACK))

import io_brain
import precompute
import eval_brain
import solvers


def _read_manifest() -> list[tuple[str, list[int], list[int]]]:
    out = []
    with open(TRACK / "manifest.txt") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("subject_id"):
                continue
            parts = line.split("\t")
            sid, train, test = parts[0], parts[1], parts[2]
            train_idx = [int(x) for x in train.split(",")]
            test_idx = [int(x) for x in test.split(",")]
            out.append((sid, train_idx, test_idx))
    return out


def _peak_rss_gb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 2**30
    except ImportError:
        return float("nan")


def _load_subject(sid: str, train_idx: list[int], test_idx: list[int],
                  resolution: str, hemi: str,
                  fsaverage_path: str, cache_dir: pathlib.Path):
    """Project all 32 contrasts to fsaverage surface, split train/test.

    cmaps layout from nilearn: subject-major ordering.
    For n_subjects=12 and n_contrasts contrasts requested,
    cmaps[sub_row_idx * n_contrasts + contrast_j] = NIfTI for
    (subject sub_row_idx, contrast j).
    ext_vars has one row per subject (12 rows), NOT one per (subject, contrast).
    """
    from nilearn import datasets, surface
    contrasts = io_brain.list_localizer_contrasts()
    n_contrasts = len(contrasts)

    loc = datasets.fetch_localizer_contrasts(
        contrasts, n_subjects=12,
        data_dir=str(io_brain.DATA_ROOT / "localizer"),
        get_anats=False, get_masks=False, verbose=0,
    )
    df = loc.ext_vars
    # ext_vars has 12 rows (one per subject); find this subject's row index
    sub_row_idxs = df.index[df["participant_id"] == sid].tolist()
    if not sub_row_idxs:
        raise ValueError(
            f"subject {sid!r} not in localizer cache "
            f"(available: {list(df['participant_id'])})"
        )
    sub_row_idx = sub_row_idxs[0]

    # Project each contrast NIfTI to surface
    # cmaps ordering: subject-major — cmaps[sub_row_idx * n_contrasts + j]
    surface_vals: list[np.ndarray | None] = []
    for j in range(n_contrasts):
        cmap_idx = sub_row_idx * n_contrasts + j
        if cmap_idx >= len(loc.cmaps) or loc.cmaps[cmap_idx] is None:
            surface_vals.append(None)
            continue
        try:
            img = nb.load(loc.cmaps[cmap_idx])
            v = np.asarray(surface.vol_to_surf(img, fsaverage_path),
                           dtype=np.float32)
            surface_vals.append(v)
        except Exception:
            surface_vals.append(None)

    # Determine n_vertices from the first valid projection
    valid = next((v for v in surface_vals if v is not None), None)
    if valid is None:
        raise RuntimeError(f"No valid contrasts projected for subject {sid!r}")
    n_v = valid.shape[0]

    F = np.zeros((n_v, n_contrasts), dtype=np.float32)
    for i, v in enumerate(surface_vals):
        if v is not None:
            # vol_to_surf produces NaN at cortical vertices outside the MNI152
            # volume ("Mean of empty slice"). Replace with 0 so NaNs don't
            # propagate into C_lin or downstream eval.
            F[:, i] = np.nan_to_num(v, nan=0.0)

    F_train = F[:, train_idx]
    F_test = F[:, test_idx]

    # Geometry: geodesic distance on the fsaverage surface
    verts, faces = io_brain.load_fsaverage_mesh(resolution, hemi)
    sparse = (verts.shape[0] > 30000)
    C_geo = precompute.geodesic_matrix(verts, faces, sparse=sparse,
                                       cache_dir=cache_dir)
    return {"C_geo": C_geo, "F_train": F_train, "F_test": F_test, "n_v": n_v}


def _normalize_cost(C: np.ndarray) -> np.ndarray:
    """Normalize a cost matrix to [0, 1] by dividing by its max.

    Required for numerical stability of entropic solvers: without this,
    geodesic matrices with max ~500 (fsaverage5 units) cause Sinkhorn
    weights exp(-C/epsilon) to underflow to 0 at epsilon=5e-3.
    """
    from scipy import sparse as _sp
    if _sp.issparse(C):
        mx = C.data.max() if C.nnz > 0 else 1.0
        return C / mx
    mx = C.max()
    return C / mx if mx > 0 else C


def _gw_full_call(solver, sub_A, sub_B, *, epsilon, fgw_alpha, seed,
                  rho_a, rho_b):
    """Build C_lin from train features and call solver.

    All cost matrices are normalized to [0, 1] for numerical stability of
    entropic (Sinkhorn-based) solvers. C_lin is already in [0, 2] (cosine
    dissimilarity), so we normalize it by 2.
    """
    C_lin = precompute.feature_cost_matrix(sub_A["F_train"], sub_B["F_train"])
    C_geo_a = _normalize_cost(sub_A["C_geo"])
    C_geo_b = _normalize_cost(sub_B["C_geo"])
    C_lin_n = C_lin / 2.0  # cosine dissim ∈ [0, 2] → [0, 1]
    return solvers.fgw_pair(solver, C_geo_a, C_geo_b, C_lin_n,
                            epsilon=epsilon, fgw_alpha=fgw_alpha, seed=seed,
                            rho_a=rho_a, rho_b=rho_b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", required=True,
                    choices=["fsaverage5", "fsaverage6", "fsaverage7"])
    ap.add_argument("--solver", required=True, choices=[
        "fugw-native", "pot-entropic-fgw", "torchgw-balanced", "torchgw-unbalanced",
    ])
    ap.add_argument("--pair", required=True,
                    help="<sub_a>__<sub_b>, e.g. S01__S04")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epsilon", type=float, default=5e-3)
    ap.add_argument("--fgw-alpha", type=float, default=0.5)
    ap.add_argument("--rho", type=float, default=1.0,
                    help="(unbalanced solvers only) symmetric rho_a=rho_b")
    ap.add_argument("--hemi", default="left", choices=["left", "right"])
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    # Resolve fsaverage mesh path once
    from nilearn import datasets as nd
    fs = nd.fetch_surf_fsaverage(mesh=args.resolution,
        data_dir=str(io_brain.DATA_ROOT / "fsaverage"))
    fs_path = fs["pial_left" if args.hemi == "left" else "pial_right"]

    sub_a, sub_b = args.pair.split("__")
    manifest = {row[0]: (row[1], row[2]) for row in _read_manifest()}
    train_a, test_a = manifest[sub_a]
    train_b, test_b = manifest[sub_b]
    cache_dir = (TRACK.parents[2] / "results" / "c8_brain_alignment"
                 / "_precompute_cache" / args.resolution)

    rec = {
        "track": "core/08_brain_alignment",
        "resolution": args.resolution, "solver": args.solver, "pair": args.pair,
        "seed": args.seed, "epsilon": args.epsilon, "fgw_alpha": args.fgw_alpha,
        "rho": args.rho, "hemi": args.hemi,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "status": "ok", "error": None,
        "metrics": {}, "efficiency": {},
    }
    out_file = args.out / (
        f"core_08_brain__{args.solver}__{args.resolution}"
        f"__{args.pair}__seed{args.seed}.json"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()

        t_load = time.perf_counter()
        sub_A = _load_subject(sub_a, train_a, test_a, args.resolution,
                              args.hemi, fs_path, cache_dir)
        sub_B = _load_subject(sub_b, train_b, test_b, args.resolution,
                              args.hemi, fs_path, cache_dir)
        rec["n_vertices"] = sub_A["n_v"]
        wall_load = time.perf_counter() - t_load

        # Sparse geodesic guard: only fugw-native consumes sparse natively
        from scipy import sparse as _sp
        if _sp.issparse(sub_A["C_geo"]) and args.solver != "fugw-native":
            raise RuntimeError(
                f"sparse geodesic at {args.resolution} only supported by "
                f"fugw-native; {args.solver} requires dense (memory OOM expected)")

        t_solve = time.perf_counter()
        out = _gw_full_call(args.solver, sub_A, sub_B,
                            epsilon=args.epsilon, fgw_alpha=args.fgw_alpha,
                            seed=args.seed, rho_a=args.rho, rho_b=args.rho)
        wall_solve = time.perf_counter() - t_solve
        T = out["T"]

        ev = eval_brain.eval_alignment(T, sub_A["F_test"], sub_B["F_test"])
        rec["metrics"] = {**ev, "fgw_objective": out["fgw_objective"]}
        rec["efficiency"] = {
            "wall_s_load":  float(wall_load),
            "wall_s_solve": float(wall_solve),
            "wall_s_total": float(wall_load + wall_solve),
            "gpu_peak_gb":  float(torch.cuda.max_memory_allocated() / 2**30)
                            if torch.cuda.is_available() else None,
            "cpu_peak_gb":  _peak_rss_gb(),
        }
    except Exception as e:
        rec["status"] = "fail"; rec["error"] = f"{type(e).__name__}: {e}"

    with open(out_file, "w") as fh:
        json.dump(rec, fh, indent=2, default=str)
    print(f"[c8] wrote {out_file} (status={rec['status']})")


if __name__ == "__main__":
    main()
