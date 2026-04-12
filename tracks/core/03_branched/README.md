# Track: core/03_branched

**Task:** Align a **2D spiral with an asymmetric Y-fork** to a **3D Swiss roll
with the same Y-fork**. The spiral's outer end splits into two tails of
different lengths and at different angles from the tangent — an
intrinsically asymmetric geometry. The `branch_frac` label partitions the
dataset into a "backbone" (main + tail 1) and an "off-axis branch" (tail 2).

## Why

Pure GW on spiral ↔ Swiss roll (track 01) has two equivalent optima
(forward + reverse) because both manifolds are reversal-symmetric. Track 02
breaks the tie via FGW features. This track takes the opposite approach:
break the tie at the **data** level by attaching a Y-fork with **unequal**
tails (one long + tangential, one short + off-axis). The curve has a
natural "main direction" (spiral continues through tail 1) plus an
auxiliary branch (tail 2).

## Dataset

- `sample_branched_spiral(n, branch_frac=0.3, theta_tail_start=9.0,
   tail1_len=1.2, tail2_len=0.6, tail2_angle=π/6)`
  - `(1 - branch_frac) * n` points on the main spiral (θ ∈ [0, 9])
  - `branch_frac * n / (1 + tail2_len/tail1_len)` points on **tail 1**
    — the *curve-continuation* branch, a straight segment along the
    spiral's local tangent at θ=9 of length `tail1_len`
  - remaining tail points on **tail 2** — a shorter branch at angle
    `tail2_angle` toward the outward radial, of length `tail2_len`
  - Point density is roughly uniform along both tails
- Labels:
  - **0 (backbone)**: main spiral + tail 1 — together they parameterise a
    single continuous curve with θ ∈ [0, 9 + tail1_len]
  - **1 (off-axis branch)**: tail 2
- `sample_branched_swiss_roll(n, ...)` — 3D analogue with independent z

## Solvers supported

| `--solver` | Library | Notes |
|---|---|---|
| `torchgw-landmark` | torchgw | Same config as track 01 |

## Metrics

- `task.branch_accuracy` — fraction of source points whose argmax-matched
  target carries the same label
- `task.main_arclen_spearman` — **signed** Spearman ρ on backbone points
  (label 0 = main + tail 1). The backbone is a monotone parametric curve,
  so a forward match gives +1.
- `task.tail_arclen_spearman` — **signed** Spearman ρ on the off-axis
  branch (label 1 = tail 2) alone. Measures whether the short branch is
  matched as a coherent region (+1) or scrambled (low / negative).

## Notes on asymmetric-fork difficulty

Pure GW aligns structural distances; it has no direct signal to tell
which of two tails is the "continuation" and which is the "branch". When
tail 1 and tail 2 are similar in scale (both around 0.6 units), GW can
swap them — that's the symmetric-V case where `branch_accuracy` stays
≥0.98. Once the tails become asymmetric (1.2 vs 0.6 here), the harder
problem — matching a long tangent extension vs. a short off-axis stub —
introduces ambiguity that pure GW cannot fully resolve from geometry
alone. Expect `tail_arclen_spearman` to be the sensitive metric here;
see `docs/experiments/2026-04-12-symmetry-breaking.md`.
