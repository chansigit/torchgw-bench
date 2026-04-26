# C8 — fMRI Brain Alignment vs FUGW: Design Spec

**Date:** 2026-04-26
**Status:** approved (brainstorming → spec); awaiting user review before plan
**Track slot:** `tracks/core/08_brain_alignment`
**Supersedes:** C7 (cell morphology) was paused after smoke confirmed torchgw
mismatched for "many tiny GW" workloads. C8 moves to the opposite regime
(few large GW per subject pair) where torchgw was actually designed to win.

## 1. One-liner

On the IBC fMRI dataset, do inter-subject cortical alignment across three
freesurfer resolutions (fsaverage5/6/7) with a 4-solver shootout to
quantify two things: (a) does giving torchgw fully-unbalanced Sinkhorn
support let it match the FUGW package's alignment quality, and (b) does
torchgw's sampled-MC scale past the ~30 k vertex memory ceiling that C1
identified, on real cortical meshes (~163 k vertices at fsaverage7).

## 2. Why this track

- **Closes the C1 thread on real data.** C1 found torchgw's effective
  ceiling is N ≈ 30 k on synthetic spirals. fsaverage6 (40 k) and
  fsaverage7 (163 k) are the natural follow-up: does the ceiling hold,
  break, or change shape on actual cortical meshes?
- **Closes the "torchgw vs domain-tuned solver" question.** C7 showed
  torchgw is mismatched for many-tiny-GW workloads; C8 tests the opposite
  regime (one large fused-unbalanced GW per pair) where it should win.
- **Generates an upstreamable contribution.** torchgw's Sinkhorn currently
  supports only one-sided semi-relaxed (`rho`); extending it to two-sided
  unbalanced (`rho_a, rho_b`) is a clean ~150 LOC PR (see §10) that turns
  torchgw into a credible competitor for fMRI alignment.

## 3. Pre-work — torchgw upstream PR (blocking)

Before C8 bench can run, modify `/scratch/users/chensj16/projects/sgw/`
torchgw source:

- `_sinkhorn_loop_pytorch`: replace single `tau` with `(tau_a, tau_b)`,
  apply two-sided KL damping symmetrically.
- `_sinkhorn_loop` (compiled-iter path): same change.
- `_sinkhorn_torch`, `_sinkhorn_unrolled`, `_sinkhorn_differentiable`:
  thread `(rho_a, rho_b)` through their signatures.
- `_gw_loop`: pass `(rho_a, rho_b)` from caller to `sinkhorn_fn`.
- Public API `sampled_gw` and `sampled_lowrank_gw`: add `rho_a, rho_b`
  kwargs (default `None` → falls back to single `rho` semi-relaxed when
  one is set, or balanced when both are `None`).
- Triton fast path (`_triton_sinkhorn.py`): when `rho_a != rho_b` (i.e.
  fully unbalanced), fall back to PyTorch path. Don't extend the Triton
  kernel in this PR — that's follow-up.
- Tests: numerically match POT's `entropic_unbalanced_sinkhorn` on small
  synthetic problems, and FUGW package's `solve_unbalanced_sinkhorn`
  output to within ε. Convergence rate sanity (small-rho → balanced).

Estimated work: 1–2 days. Open PR upstream after C8 bench validates the
implementation.

## 4. Dataset

**Brainomics Localizer (Pinel et al. 2007)** — 94 subjects × 32 task contrasts in MNI152 volume space.

Originally we planned to use IBC, but its data fetcher (`ibc_api`) requires
EBRAINS account registration; Localizer has the same task-contrast inter-subject
alignment structure, lives in nilearn natively (no auth), and is fully reproducible.

- Acquired via `nilearn.datasets.fetch_localizer_contrasts`.
- 12 subjects (S01-S12) pinned in `manifest.txt`.
- 32 contrasts split 70/30 (22 train / 10 test) by sorted contrast name.
- Volume → fsaverage{5,6,7} surface projection happens in `precompute.py`
  via `nilearn.surface.vol_to_surf`.

Out of scope (deliberate): real IBC, HCP, NSD, cross-species, multimodal.

## 5. Pipeline and swap point

```
IBC subject (cortical surface + N task contrast maps)
   │
   ▼  per-subject preprocessing (cached):
   │    - load fsaverage{5,6,7} surface (vertices + faces)
   │    - load N task contrasts (1 vector per contrast per vertex)
   │    - split contrasts → train (used to compute cost) / test (held-out)
   │    - C_geo = mesh geodesic distance matrix (gdist; sparse-stored at
   │              fsaverage7 due to 213 GB dense size)
   │    - F_train = (n_vertices × n_train_contrasts) feature matrix
   │
   ▼  ── SWAP POINT ──  for each subject pair (A, B):
   │     C_lin_AB = 1 - corr(F_train_A vertex × F_train_B vertex)
   │     T_AB = solver.fgw(C_geo_A, C_geo_B, C_lin=C_lin_AB,
   │                       fgw_alpha=0.5, epsilon=5e-3,
   │                       rho_a=1.0, rho_b=1.0)   # unbalanced kwargs
   │
   ▼  evaluate:
   │     - apply T_AB to held-out F_test_A → F̂_test_B
   │     - vertex-wise Pearson corr against actual F_test_B (metric A)
   │     - retrieval: argmax over B's test contrasts (metric B)
   │     - record final FGW objective (metric C) and efficiency (metric E)
```

**Controlled-variable invariant:** all four solvers consume the same
`(C_geo_A, C_geo_B, C_lin_AB)` tuple. Solvers differ only in how they
solve FGW, not in how cost matrices are constructed.

## 6. Solvers (4)

| Solver | Algorithm | Role |
|---|---|---|
| `fugw-native` | FUGW package, full unbalanced FGW (rho_a=rho_b=1.0) | literature reference |
| `pot-entropic-fgw` | `ot.gromov.entropic_fused_gromov_wasserstein`, balanced | non-fancy GPU baseline |
| `torchgw-balanced` | torchgw current main, balanced FGW | "before-PR" baseline |
| `torchgw-unbalanced` | torchgw + new `(rho_a, rho_b)` Sinkhorn | **main contender** |

Fixed shared hyperparameters across solvers:
- `epsilon = 5e-3`
- `fgw_alpha = 0.5` (FUGW paper default)
- `rho_a = rho_b = 1.0` (unbalanced solvers only; balanced ones unaffected)
- `max_iter = 500`

## 7. Evaluation metrics (A + B + C + E)

For each (resolution, solver, pair, seed) cell, JSON record contains:

**A — held-out functional correlation**
- `func_corr_holdout_mean`, `_std` over held-out test contrasts (mean
  vertex-wise Pearson r between F̂_test_B and F_test_B)

**B — retrieval accuracy**
- `retrieval_top1`, `retrieval_top5` (hit rate of correct contrast in
  top-k by cosine sim)

**C — FGW objective**
- `fgw_objective_final`, `n_iter_to_converge`

**E — efficiency**
- `wall_s_total`, `wall_s_solve` (excludes preprocessing)
- `gpu_peak_gb`, `cpu_peak_gb`
- OOM / fail records: `status="fail"`, `error="<error string>"`

## 8. Experiment matrix

```
resolutions   = [fsaverage5, fsaverage6, fsaverage7]   # 3
solvers       = [fugw-native, pot-entropic-fgw,
                 torchgw-balanced, torchgw-unbalanced]  # 4
subject_pairs = 12 * 11 / 2                            # 66
seeds         = [0, 1, 2]                              # 3

Total = 3 × 4 × 66 × 3 = 2376 alignment runs
```

Hardware budget projection:
- fsaverage5 (~10k vertices): all solvers fit, ~5–30 s/pair → ~7 h total
- fsaverage6 (~40k): pot-entropic-fgw likely OOM; torchgw and FUGW survive,
  ~2 min/pair survivors → ~10 h
- fsaverage7 (~163k): pot-entropic-fgw 100% OOM; torchgw-balanced ~90% OOM
  per C1; torchgw-unbalanced and fugw-native are the survivors
  → ~10 h on survivors

Total budget: ~25 h on a single 80 GB H100 / L40S. Failed cells write
`status=fail` JSONs (no silent skips).

## 9. File layout (mirrors C2/C5/C7)

```
tracks/core/08_brain_alignment/
├── README.md
├── env.yaml                 # nilearn, nibabel, fugw, gdist, torchgw, pot
├── fetch.sh                 # IBC + fsaverage meshes via nilearn
├── manifest.txt             # 12 subject IDs + train/test contrast indices
├── io.py                    # surface + contrast loaders
├── precompute.py            # C_geo (sparse where needed) + F_train, cached
├── solvers.py               # 4-solver FGW dispatch w/ unified kwargs
├── eval.py                  # holdout corr + retrieval + objective
├── run.py                   # CLI: --resolution --solver --pair --seed
└── tests/
    ├── test_io.py
    ├── test_precompute.py
    ├── test_eval.py
    └── test_solvers.py

scripts/run_c8_bench.sh
scripts/experiments/make_c8_plots.py
docs/experiments/2026-04-26-c8-brain-alignment.md
docs/figures/c8_*.png
```

## 10. Engineering caveats (acknowledged, not engineered around)

1. **torchgw-unbalanced does not reproduce FUGW exactly.** The outer GW
   iteration uses torchgw's existing sampled-MC `Lambda_gw` formula
   (no Sejourne-style outer correction). Inner Sinkhorn is correctly
   unbalanced after the PR, but the fixed point will differ from FUGW
   package's. This is a deliberate algorithmic choice (a new solver,
   not a port), documented in writeup.
2. **fsaverage7 dense `C_geo` is 213 GB.** Must use sparse / on-the-fly
   geodesic at this resolution. `precompute.py` switches to sparse CSR
   above fsaverage6.
3. **torchgw OOM expected at fsaverage7.** C1 already showed the ceiling
   at ~30 k. If the unbalanced PR doesn't lift this, fsaverage7 cells
   for both torchgw variants fail. That is itself a finding — the PR
   adds *algorithmic* support but not *scale* support. Honestly reported.
4. **IBC release pinning.** nilearn fetch defaults to "latest" which can
   change. Fix nilearn version in `env.yaml`, freeze subject ID list in
   manifest.
5. **Triton fast-path** is not extended for unbalanced — falls back to
   PyTorch when `(rho_a, rho_b)` differ from balanced. The wall numbers
   for `torchgw-unbalanced` therefore include this overhead and are
   **lower-bound** estimates of the algorithm's performance. Documented.

## 11. Success criteria

The writeup will claim, with quantitative backing:

- **Algorithmic competitiveness**: at fsaverage5/6, `torchgw-unbalanced`
  achieves `func_corr_holdout` within `ε_corr ≤ 0.02` of `fugw-native`.
  If yes → "the new Sinkhorn variant matches FUGW on real fMRI data."
- **Algorithmic gap on quality**: `torchgw-balanced` vs `torchgw-unbalanced`
  shows the marginal value of unbalanced support (paired-difference test).
- **Scale finding**: at fsaverage7, exact survival counts per solver and
  whether `torchgw-unbalanced` lifts the C1 30k ceiling.
- **Speed comparison** (only on cells where both solvers succeed):
  `torchgw-*` vs `fugw-native` wall ratio per resolution.

Negative results are acceptable: if `torchgw-unbalanced` is consistently
worse than FUGW on quality, the writeup's conclusion shifts to "the
inner-Sinkhorn fix alone is insufficient — outer GW iteration needs the
Sejourne-style correction too" and recommends that as future work.

## 12. Explicit non-goals

- Not reproducing FUGW paper's exact numbers (no Sejourne outer iter).
- Not adding HCP / NSD / cross-species datasets.
- Not extending the Triton fast-path for unbalanced (PyTorch fallback OK).
- Not adding multimodal alignment (structural + functional).
- Not adding `vertex_displacement` metric (FUGW-strength axis; would
  unfairly penalize torchgw which doesn't optimize for smoothness).
- Not coarse-to-fine multiscale init (torchgw has multiscale; out of scope
  for this track).
