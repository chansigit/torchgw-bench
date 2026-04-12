# Track: core/03_branched

**Task:** Align a **branched 2D spiral** to a **branched 3D Swiss roll**.
Both manifolds carry the same logical structure — a main arc and a
perpendicular side-branch emerging at `θ_branch ≈ 6` — so the geometry
itself is non-symmetric under orientation reversal. Pure GW should land
on the forward correspondence deterministically.

## Why

Pure GW on spiral↔swiss-roll (track 01) has two equivalent optima
(forward + reverse). Track 02 breaks the tie via FGW features. This track
takes the opposite approach: break the tie at the **data** level by making
the manifolds intrinsically non-symmetric.

## Dataset

- `sample_branched_spiral(n, branch_frac=0.3, theta_branch=6.0, branch_len=0.4)`
  - `(1 - branch_frac) * n` points on the main spiral (θ ∈ [0, 9])
  - `branch_frac * n` points on a perpendicular branch at `theta_branch`
  - Returns `(points (n,2), angles (n,), labels (n,) ∈ {0=main, 1=branch})`
- `sample_branched_swiss_roll(n, ...)` — analogous in 3D

## Solvers supported

| `--solver` | Library | Notes |
|---|---|---|
| `torchgw-landmark` | torchgw | Same config as track 01 |

## Metrics

Phase 1c additions on top of the standard schema:

- `task.branch_accuracy` — fraction of source points whose matched target
  has the same branch label
- `task.main_arclen_spearman` — **signed** Spearman ρ on the main-branch
  points only (no abs)
