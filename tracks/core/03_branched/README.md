# Track: core/03_branched

**Task:** Align a **spiral with a Y-fork (2D)** to a **Swiss roll with a
Y-fork (3D)**. Two straight tails diverge symmetrically from the outer end
of each manifold, making the two endpoints of the curve geometrically
distinct. Pure GW converges to the forward correspondence deterministically.

## Why

Pure GW on spiral ↔ Swiss roll (track 01) has two equivalent optima
(forward + reverse) because both manifolds are reversal-symmetric. Track 02
breaks the tie via FGW features. This track takes the opposite approach:
break the tie at the **data** level by attaching a Y-fork (two diverging
tails) to one end only. The resulting manifold is a spiral with a
"swallow-tail" — the inner end is a single high-curvature terminus, the
outer end is a pair of straight branches.

## Dataset

- `sample_branched_spiral(n, branch_frac=0.3, theta_tail_start=9.0,
   tail_len=0.6, fork_angle=π/3)`
  - `(1 - branch_frac) * n` points on the main spiral (θ ∈ [0, 9])
  - `branch_frac * n / 2` points on tail A and another half on tail B.
    Both tails start at the spiral's outer endpoint (θ=9) and head outward
    — they are the outward radial direction rotated by ±fork_angle/2.
    Using the radial (not the tangent) as the symmetry axis guarantees both
    tails move strictly to r > 1, so neither tail curls back onto the
    spiral's inner body.
  - Returns `(points (n,2), angles (n,), labels (n,) ∈ {0=main, 1=tail})`.
    Both tails share label 1 (they are collectively the non-main region).
- `sample_branched_swiss_roll(n, ...)` — analogous in 3D with an
  independent z per point

The function names keep the `branched` prefix for historical reasons; the
current geometric construction is a symmetric Y-fork of two tails.

## Solvers supported

| `--solver` | Library | Notes |
|---|---|---|
| `torchgw-landmark` | torchgw | Same config as track 01 |

## Metrics

- `task.branch_accuracy` — fraction of source points whose argmax-matched
  target carries the same label (main vs. tail)
- `task.main_arclen_spearman` — **signed** Spearman ρ on main-arc points
  only (no `abs`); forward match gives +1, reverse would give −1
