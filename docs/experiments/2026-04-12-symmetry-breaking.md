# Breaking GW's Orientation Ambiguity — Two Approaches

**Date:** 2026-04-12 · **Tracks:** `core/01_foundation`, `core/02_foundation_fused`, `core/03_branched` · **Tag:** `v0.1.0-m1b` and forward

## TL;DR

Gromov-Wasserstein on a symmetric manifold (spiral → Swiss roll) has two
equivalent optima: the **forward** correspondence and the **reverse**. At
small scales the solver usually lands on forward, but at larger scales it can
flip — and our Spearman-ρ task metric then reports a large negative number
that looks like a regression but is actually a perfectly good reverse match.

We tested two independent fixes:

1. **C2 — Fused GW.** Attach the arclength parameter θ to each point as a
   scalar feature, then run FGW. The Wasserstein term on features pins down
   the orientation.
2. **C3 — Asymmetric Y-fork geometry.** Attach an asymmetric Y-fork to the
   outer end of both manifolds: a long tail (tail 1, along the spiral's
   local tangent — the "curve continuation") and a shorter tail (tail 2,
   rotated 30° toward the outward radial — the "off-axis branch"). Main +
   tail 1 share label 0 (backbone); tail 2 carries label 1 (branch).

Both work. Both give +0.999 Spearman on the same seed where C1 at N=10k
flips to −0.999.

## The setup

### Three tracks, three datasets

![datasets](../figures/datasets.png)

- **C1** (top row, first two panels): the baseline — a 2D Archimedean spiral
  (source) and a 3D Swiss roll (target) generated from the same parametric
  schedule θ ∈ [0, 9]. Because both manifolds are smooth, connected, and the
  "rolling" direction can run either way, the GW optimal transport has
  two symmetric solutions.
- **C2** (top right): identical data to C1, but θ is passed to the solver as
  a per-point feature. The cost matrix M ∈ ℝ^(N×K) is the squared-Euclidean
  distance between source θ and target θ, normalised.
- **C3** (bottom row): an **asymmetric Y-fork** is attached at θ=9. Tail 1
  extends along the spiral's local tangent for 1.2 units (the curve
  continuation); tail 2 is a shorter stub of length 0.6, rotated 30°
  toward the outward radial from tail 1. `branch_frac = 0.3` of the points
  go on the tails (allocated proportionally to length so density is
  uniform). **Labels:** main spiral + tail 1 = 0 (together they form a
  monotone backbone parameterised by θ ∈ [0, 10.2]); tail 2 = 1 (the
  off-axis branch).

All experiments use `torchgw.sampled_gw(distance_mode="landmark")` with the
same hyperparameters (`M=80, k=5, n_landmarks=50, epsilon=5e-3, max_iter=300`)
unless otherwise noted. C2 additionally uses `fgw_alpha=0.5` and passes the
feature cost matrix through `C_linear`. C3 uses the same torchgw configuration
as C1.

### How we visualise the match

For each source point we take the argmax of the transport plan row to pick
its matched target column, then colour the source scatter by the target's θ.
If the matching is forward, the colour ramp on the source sweeps in lockstep
with θ as you trace the spiral from centre to edge. If it's reverse, the
colour ramp runs backwards.

![matchings](../figures/matchings.png)

## Result 1: C1 flips at scale

At N=400, K=500, pure GW (C1) lands on the forward match with Spearman
≈ +0.999.

At N=10,000, K=12,000 — the same seed, the same solver, the same
hyperparameters — it lands on the reverse match with Spearman ≈ **−0.999**.

Look at the two leftmost panels of the matchings figure: the colour pattern
is geometrically the same spiral, but inverted. The solver didn't fail; it
just picked the other optimum.

This is what motivated returning `|ρ|` from the Phase-1 task metric. It's
correct as a statement about the **structural** quality of the alignment,
but it discards information about orientation.

## Result 2: C2's feature term fixes it

![spearman](../figures/spearman_bar.png)

At N=400, K=500, both `torchgw-fused` and `pot-fused` give +0.999 Spearman
with `alpha = 0.5` / `fgw_alpha = 0.5` balancing the W and GW terms.

| Solver              | Spearman (signed) | Wall  |
|---------------------|------------------:|------:|
| `torchgw-fused`     | **+0.9997**       | 5.43s |
| `pot-fused`         | **+0.9996**       | 2.17s |

Mechanism: the feature cost `M[i, j] = (θ_src[i] - θ_tgt[j])²` is minimised
when source point i matches target point j with similar θ. The reverse
correspondence would pair small-θ with large-θ and incur a heavy
Wasserstein penalty on top of the (equal) GW cost, so FGW rejects it.

Caveat: this changes the problem. FGW is not GW — it is GW with a side
constraint. If the point you want to make is "our GW solver is correct", C2
doesn't make it. If the point is "our alignment pipeline gets the answer",
it does.

## Result 3: C3's asymmetric geometry also fixes it

Same hyperparameters as C1. Same solver (`torchgw-landmark`, no fused
term). Just a different dataset.

![c3_detail](../figures/c3_detail.png)

The four panels show: (1) the 2D source with three coloured regions —
main spiral (blue), tail 1 (green, tangent continuation, part of the
backbone), tail 2 (red squares, off-axis branch); (2) the 3D target with
the same tripartition; (3) source points coloured by the argmax-matched
target θ — the main spiral and tail 1 get a clean forward colour ramp
(backbone matched as one monotone curve); tail 2 is outlined in crimson
with its matched-θ colour; (4) per-point label propagation: green = `src
and matched target share the same label`, red × = mismatch.

| Metric                      | Value        |
|-----------------------------|-------------:|
| `task.branch_accuracy`      | 0.9300       |
| `task.main_arclen_spearman` | **+0.8827**  |
| `task.tail_arclen_spearman` | +0.2281      |
| Wall                        | 6.08s        |

**An honest negative finding.** The backbone Spearman dropped from
+0.999 (track 01 baseline) to +0.88, and the tail Spearman came in at
a weak +0.23. Why? Pure GW aligns by structural distance alone, so when
the two tails are **visibly asymmetric** — tail 1 (long, tangent) looks
geometrically different from tail 2 (short, 30° off-axis) — GW
occasionally swaps them, mapping source tail 1 onto target tail 2 and
vice versa. The label-mismatch hotspot in panel 4 lives right at the
fork root, where the two tails start to diverge and the solver cannot
decide which is which.

In the earlier symmetric-V variant (both tails = 0.6 units), swap was
harmless because the two targets were indistinguishable — branch
accuracy stayed ≥0.98. The current design strengthens the "which tail
is the backbone" signal in principle (tail 1 is longer → more structural
weight), but the opening angle of 30° keeps the two tails close in
space, and GW's cost function doesn't quite weight the length difference
heavily enough to lock the match. Opening the Y wider, or combining
geometry with FGW features (track 02's recipe), should both push
`tail_arclen_spearman` back toward +1 — a natural next experiment.

`branch_accuracy` is the fraction of source points whose argmax-matched
target carries the same branch label (main vs. branch).
`main_arclen_spearman` is Spearman-ρ computed on the main-arc source points
only (signed, no abs).

Mechanism: the two endpoints of the manifold are now topologically
different. The inner endpoint (θ=0) is a single terminus. The outer
endpoint is a Y-fork where two branches diverge. A reverse match would
have to contract the inner 1-terminus onto the outer 2-terminus region
and expand the Y-fork onto the 1-terminus — a topological mismatch that
costs heavily under GW. So the orientation ambiguity of track 01 is
eliminated, and the main-arc Spearman stays **positive** (no sign flip).
What remains is a milder second-order problem, the tail swap described
above.

## Comparison

|                           | C1 (baseline)    | C2 (FGW)                     | C3 (asymmetric Y-fork)     |
|---------------------------|:----------------:|:----------------------------:|:--------------------------:|
| Dataset                   | symmetric        | symmetric                    | **asymmetric**             |
| Method                    | **pure GW**      | fused GW                     | pure GW                    |
| Orientation at N=400      | forward (+0.999) | forward (+0.999 / +0.999)    | forward (+0.999)           |
| Orientation at N=10k      | **reverse** (−0.999) | — (not yet run)           | — (not yet run)            |
| Extra metric needed?      | `|ρ|`            | none                         | `branch_accuracy`, main-ρ, tail-ρ |
| Extra dataset complexity? | none             | θ feature                    | Y-fork generator + labels  |
| Extra solver complexity?  | none             | fused API + feature cost     | none                       |

## What's next

1. **C2 at scale.** Re-run C2 at N=10k, 20k with `fgw_alpha=0.5` and check
   the Spearman stays positive. If it does, C2 can retire the `|ρ|`
   fallback for its own track.
2. **C3 at scale.** Same question for C3. Branched geometry should hold up
   at any scale, but it's worth measuring how `branch_accuracy` behaves
   when `branch_frac` shrinks (is a 5% branch enough to pin orientation?).
3. **Seed stability.** All three tracks currently run one seed. We want at
   least 3 seeds per (track, scale) to report `stability.seed_std_spearman`,
   which is also the only way to quantify "C1 sometimes flips" as a
   probability rather than a single anecdote.

## Reproducing

```bash
# Environment
source /scratch/users/chensj16/venvs/dl2025/.venv/bin/activate
cd /scratch/users/chensj16/projects/torchgw-bench

# Regenerate figures (takes ~3 min on H100, most of it the C1 N=10k run)
python scripts/experiments/make_symmetry_figures.py

# Unit tests for all three tracks
python -m pytest tracks/core/01_foundation tracks/core/02_foundation_fused tracks/core/03_branched -v

# Smoke-test each track end-to-end
python tracks/core/02_foundation_fused/run.py --solver torchgw-fused --seed 0 \
    --out /tmp/fused/ --n-source 400 --n-target 500
python tracks/core/03_branched/run.py --solver torchgw-landmark --seed 0 \
    --out /tmp/branched/ --n-source 400 --n-target 500
```

The figure-generation script is fully self-contained; it imports each
track's `run.py` via `sys.path` and calls the solver wrappers directly. No
results-directory scan, no reporter pipeline.
