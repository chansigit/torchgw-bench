# C8 — fMRI brain alignment vs FUGW (2026-04-26)

> **Status: scaffold — bench runs pending. `<fill>` placeholders
> populated post-bench. Spec at
> `docs/superpowers/specs/2026-04-26-c8-brain-alignment-design.md`;
> plan at `docs/superpowers/plans/2026-04-26-c8-brain-alignment.md`.**

## Setup

Inter-subject cortical alignment on **Brainomics Localizer** (12 subjects
× 32 task contrasts in MNI152 volume space; we project to fsaverage via
`vol_to_surf`). Originally planned IBC but EBRAINS auth required; the
Localizer has the same task-contrast inter-subject alignment structure
and is fully open in nilearn.

Pipeline: per-subject MNI152 contrast → `vol_to_surf` to fsaverage{5,6,7}
→ C_geo (mesh geodesic, sparse at fsaverage7) and C_lin (1 - cosine on
train features) per pair → FGW solver → apply plan to held-out test
contrasts → vertex-wise Pearson r + retrieval accuracy.

**FUGW backend (probe at install)**: `<fill from
tracks/core/08_brain_alignment/fugw_probe.txt>`.

## Solvers (4)
| Solver | Algorithm |
|---|---|
| fugw-native       | FUGW package, full unbalanced FGW (rho=1.0, eps=5e-3, divergence='kl') |
| pot-entropic-fgw  | POT balanced entropic FGW |
| torchgw-balanced  | torchgw current main, balanced FGW |
| torchgw-unbalanced| torchgw + new (rho_a, rho_b) PR; symmetric rho=1.0 |

## Headline results

![quality](../figures/c8_quality.png)
![retrieval](../figures/c8_retrieval.png)
![wall](../figures/c8_wall.png)
![survival](../figures/c8_survival.png)

| solver | fsaverage5 r | fsaverage6 r | fsaverage7 r | wall (s) @ fs5 |
|---|---|---|---|---|
| fugw-native        | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| pot-entropic-fgw   | `<fill>` | OOM      | OOM      | `<fill>` |
| torchgw-balanced   | `<fill>` | `<fill>` | OOM (likely) | `<fill>` |
| torchgw-unbalanced | `<fill>` | `<fill>` | `<fill>` | `<fill>` |

## Take-home

1. **Algorithm gap closed?** `<fill from torchgw-unbalanced vs fugw-native
   func_corr difference at fsaverage5/6>`. If within ε_corr=0.02 → claim
   "the new Sinkhorn variant matches FUGW on real fMRI data."
2. **Scale ceiling lifted?** `<fill from fsaverage7 survival count>`.
3. **Speed:** `<fill from wall ratio per resolution>`.

## Caveats

- Brainomics Localizer used in lieu of IBC (EBRAINS auth required for
  IBC; Localizer is fully open and structurally similar). FUGW paper's
  IBC numbers are not directly comparable; we report relative ordering
  of solvers, not absolute numbers vs paper.
- Brainomics has 3 missing (subject, contrast) NIfTIs out of 384 (12×32);
  affected vertices are zero-filled in the feature matrix.
- torchgw-unbalanced is a NEW solver, not a FUGW reproduction. Inner
  Sinkhorn is correctly two-sided (Sejourne-style scale-from-zero), but
  outer GW iteration uses torchgw's existing sampled-MC Lambda_gw — no
  Sejourne outer correction. Fixed point differs from FUGW package's by
  design.
- Triton fast-path falls back to PyTorch when `rho_a == rho_b != 1.0`;
  torchgw-unbalanced wall numbers are lower-bound estimates.
- fsaverage7 dense `C_geo` is 213 GB; sparse path used and only
  fugw-native consumes sparse cost matrices natively.
- Smoke at pot-entropic-fgw / fsaverage5 / S01__S04 / seed=0:
  func_corr=0.695, retrieval_top1=0.80, top5=1.00 — pipeline known to
  produce non-trivial alignment.

## Reproducing

```bash
micromamba activate c8_brain
bash tracks/core/08_brain_alignment/fetch.sh
bash scripts/run_c8_bench.sh
python scripts/experiments/make_c8_plots.py
```
