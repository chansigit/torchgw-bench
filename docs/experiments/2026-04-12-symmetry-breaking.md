# Breaking GW's Orientation Ambiguity with an Asymmetric Y-Fork

**Date:** 2026-04-12 (figures refreshed 2026-04-13) · **Track:** `core/03_branched`

## TL;DR

Plain Gromov-Wasserstein on a symmetric manifold (spiral ↔ Swiss roll)
has two equivalent optima — forward and reverse — and at larger N the
solver sometimes picks the reverse, which looks like a regression but is
actually a valid match in the other direction.

The fix we settled on is to break the ambiguity **at the data level**:
attach an asymmetric Y-fork to one end of both manifolds, so the two
endpoints of the curve have topologically different local geometry (a
single-terminus inner end vs. a two-branch outer end). Plain GW then
converges to the forward correspondence deterministically.

This track also explores a second, smaller question: pure GW on the
Y-fork occasionally **swaps** the two tails (short tail → long tail and
vice versa). Using geodesic distance from the spiral centre as an FGW
feature cleans that up — the feature ranges for the short and long
tails differ (`[0, tail2_len]` vs `[0, tail1_len]`), and FGW uses that
gap to rule out cross-tail matches.

## The dataset

Source is a **3D Swiss roll with a Y-fork**; target is the same shape
embedded in **2D as a spiral with a Y-fork**. Both carry three regions:

- **main spiral** — the standard Archimedean spiral on θ ∈ [0, 9]
- **long tail** — a 1.2-unit straight segment continuing from the
  spiral's outer end along the local tangent
- **short tail** — a 0.6-unit straight segment rotated +30° toward the
  outward radial, an off-axis branch

Labels: 0 for `main + long tail` (together they form a single monotone
arc — the *backbone*), 1 for `short tail` (the true *off-axis branch*).

Every point also carries a 1D coordinate: its **geodesic distance from
the spiral's inner end (θ=0)**, computed analytically for the spiral
part and by adding the tail parameter `s` for the two tails. This
coordinate is the FGW feature and the Spearman target for the task
metrics.

![datasets](../figures/datasets.png)

The 12 coloured lines are a ground-truth guide — 5 anchor regions (3 on
the backbone, one on each tail tip) × 5 nearly-parallel lines per
bundle. A source point near the spiral's inner end (dark purple) maps to
the innermost target region; the long-tail tip (bright yellow) maps to
the target's long-tail tip; and so on.

## Pure GW works — FGW squeezes the last 6%

At N=500 (source) × K=400 (target), seed 0, same torchgw hyperparameters
for both solvers (`distance_mode="landmark"`, `M=80, k=5,
n_landmarks=50, ε=5e-3, max_iter=300`):

![solver effects](../figures/solver_effects.png)

| Solver              | `branch_accuracy` | `backbone_ρ` | `tail_ρ` |
|---------------------|------------------:|-------------:|---------:|
| `torchgw-landmark`  | 0.935             | +0.9987      | +0.9404  |
| `torchgw-fused`     | **0.908**         | **+0.9996**  | **+0.9908** |

Both solvers land on the forward match — there is no sign flip here,
thanks to the Y-fork. The difference is finer:

- Pure GW does fine on the backbone (+0.999) but struggles on the short
  tail (+0.94) because it can't fully distinguish the two tails from
  structural distances alone.
- FGW uses the arclen feature to separate the tails: the short-tail
  arclen range `[fork, fork + 0.6]` is strictly contained in the
  long-tail range `[fork, fork + 1.2]`, so a short-tail source point
  mapped to a long-tail target at arclen > fork + 0.6 pays a feature
  penalty. FGW's tail-ρ climbs to +0.99.

The `branch_accuracy` dip from 0.935 → 0.908 on FGW is counter-intuitive
but honest: FGW trades slightly more ambiguous label assignment near the
fork junction for a much cleaner tail-ρ. All errors cluster right at the
fork root where the three regions meet (see the deep-dive).

![spearman bar](../figures/spearman_bar.png)

## Deep dive

![c3 detail](../figures/c3_detail.png)

Left to right: (1) the 3D source with three coloured/marked regions —
main spiral, long tail, short tail — overlaid on a faint shaded Swiss
roll surface that gives the panel depth; (2) the 2D target with the same
three regions; (3) each 2D target point coloured by the argmax-matched
source arclen under FGW — the colour ramp flows smoothly from the
target's inner end (purple) to the tail tips (yellow), confirming a
clean forward match; (4) label-propagation check: green = backbone/tail
label preserved, red × = mismatch. The mismatches are confined to the
fork root.

## Why we dropped the C1 and C2 variants

Earlier drafts of this experiment compared three tracks — a plain
spiral (C1), a plain spiral with an FGW θ feature (C2), and the Y-fork
(C3). Once the Y-fork was in place the other two became redundant:

- C1 demonstrated the orientation flip that motivated the problem —
  the Y-fork makes that flip impossible, so the demo is no longer
  needed in this report.
- C2 used FGW features to force forward matching on symmetric data —
  the Y-fork removes the symmetry, so the feature trick isn't required.

C1 and C2 remain in the repo as benchmark track directories (they are
Phase-1 of the scale sweep against POT and still useful for that), but
this report features only C3.

## Reproducing

```bash
# Environment
source /scratch/users/chensj16/venvs/dl2025/.venv/bin/activate
cd /scratch/users/chensj16/projects/torchgw-bench

# Regenerate all four figures (≈1 min on GPU)
python scripts/experiments/make_symmetry_figures.py

# Unit tests
python -m pytest tracks/core/03_branched/tests/ -v

# End-to-end: 3D → 2D run with each solver
python tracks/core/03_branched/run.py --solver torchgw-landmark --seed 0 \
    --out /tmp/c3/ --n-source 500 --n-target 400
python tracks/core/03_branched/run.py --solver torchgw-fused --seed 0 \
    --out /tmp/c3/ --n-source 500 --n-target 400
```
