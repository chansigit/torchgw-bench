# Track: core/02_foundation_fused

**Task:** Same spiral → Swiss roll alignment as `01_foundation`, but using
**Fused Gromov-Wasserstein** with the arclength parameter θ as a per-point
feature. The Wasserstein term on features breaks the forward/reverse
orientation symmetry of pure GW, so the solver lands on the positive
(forward) matching deterministically.

## Why

Pure GW on symmetric manifolds has two equivalent optima (forward and
reverse). At large scales the solver occasionally converges to the reverse
one, and the raw Spearman rank correlation flips sign. FGW with a weak
feature term (α ≈ 0.5) forces the forward solution.

## Solvers

| `--solver` | Library | Notes |
|---|---|---|
| `torchgw-fused` | torchgw | `sampled_gw(fgw_alpha=0.5, C_linear=M)` |
| `pot-fused` | POT | `ot.gromov.entropic_fused_gromov_wasserstein(M, ..., alpha=0.5)` |

Feature construction: `F_src = src_angles[:, None]`, `F_tgt = tgt_angles[:, None]`;
inter-domain cost `M = ot.dist(F_src, F_tgt, metric="sqeuclidean")`, normalised.

## Metrics

Same as `01_foundation`. `task.spearman_arclen` is reported **with sign**
(no absolute value) since the forward solution is enforced by the feature term.
