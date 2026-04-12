# Track: core/03_branched

**Task:** Align a **spiral-with-tail (2D)** to a **Swiss-roll-with-tail (3D)**.
A short straight tail extends tangentially from the outer end of each
manifold, making the overall geometry intrinsically asymmetric under
orientation reversal. Pure GW converges to the forward correspondence
deterministically.

## Why

Pure GW on spiral ↔ Swiss roll (track 01) has two equivalent optima
(forward + reverse) because both manifolds are reversal-symmetric. Track 02
breaks the tie via FGW features. This track takes the opposite approach:
break the tie at the **data** level by appending a tangential tail to one
end only, so the two endpoints have different local geometry (tight
spiral-curvature vs. straight line).

## Dataset

- `sample_branched_spiral(n, branch_frac=0.2, theta_tail_start=9.0, tail_len=0.8)`
  - `(1 - branch_frac) * n` points on the main spiral (θ ∈ [0, 9])
  - `branch_frac * n` points on a straight tail attached at θ=9, extending
    along the local tangent for `tail_len` units
  - Returns `(points (n,2), angles (n,), labels (n,) ∈ {0=main, 1=tail})`
  - Tail-point angles are `9 + s`, preserving θ monotonicity across the
    whole curve
- `sample_branched_swiss_roll(n, ...)` — analogous in 3D, with an independent
  z-coordinate drawn uniformly for every point

The function names keep the `branched` prefix for legacy reasons; the
geometric construction is a single-sided tail, not a perpendicular T-branch.

## Solvers supported

| `--solver` | Library | Notes |
|---|---|---|
| `torchgw-landmark` | torchgw | Same config as track 01 |

## Metrics

- `task.branch_accuracy` — fraction of source points whose argmax-matched
  target carries the same label (main vs. tail)
- `task.main_arclen_spearman` — **signed** Spearman ρ on main-arc points
  only (no `abs`); forward match gives +1, reverse would give −1
