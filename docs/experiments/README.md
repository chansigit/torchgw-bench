# torchgw-bench experiments — index

Comparison of `torchgw` and POT GW / FGW solvers across four tracks:
**C3 Y-fork branched manifold** (FGW with feature anchor),
**C6 TACO shape correspondence** (pure GW on bilaterally-symmetric
meshes), **C2 single-cell multi-omics** (paired RNA+ATAC alignment),
and **C5 bilingual word-embedding alignment** (cosine cost on fastText
vectors, Alvarez-Melis 2018). All figures in [`../figures/`](../figures);
hardware is **NVIDIA H100 80GB HBM3** throughout.

## Take-home

> **On the cost axis torchgw wins when it wins** (1–2 orders of
> magnitude faster than POT, and the only option above N≈5k where POT's
> O(N²) memory wall kicks in). **But in C5 it catastrophically loses
> on accuracy** — dense cosine cost breaks `sampled_gw`'s MC estimator.
>
> **On the accuracy axis the winner depends on cost-matrix structure,
> not N**:
>
> - **Feature-anchored FGW** (C3): tie — both hit ρ ≥ 0.98 at
>   saturation. Torchgw wins the cost axis cleanly.
> - **Pure GW on symmetric feature-free mesh** (C6): POT-exact wins by
>   ~1.3× on supervised geodesic error; its sparse CG plan beats
>   torchgw's diffuse Sinkhorn plan when the task needs sharp 1-to-1
>   matching.
> - **Cross-modality with kNN-geodesic cost** (C2): **torchgw-
>   precomputed wins outright** — lower FOSCTTM and 2–9× faster than
>   pot-entropic, once the `M_samples` default is raised past the
>   scalability floor.
> - **Bilingual word embedding with dense cosine cost** (C5):
>   **pot-entropic wins by ~150×**. torchgw-precomputed gives P@1 ≈ 0
>   on the same cost matrices because MC sampling has insufficient
>   SNR on near-Gaussian-dense cost.
>
> ### Decision rule (revised with C5)
>
> 1. **N ≥ 10⁴** → torchgw (POT OOMs), but ONLY if cost is
>    structured (kNN, sparse, long-tailed). Dense cost → POT with
>    memory workarounds, or abandon GW.
> 2. **Feature-anchored FGW** → torchgw if speed matters, POT
>    otherwise; both reach the same quality.
> 3. **Pure GW on clean symmetric geometry** → POT-exact for best
>    argmax matching.
> 4. **Cost built from kNN / graph / geodesic** (C2-style) → torchgw-
>    precomputed with SCOT cost and `M_samples ≥ N/2`.
> 5. **Dense-cosine cost on high-dim embeddings** (C5-style) → POT,
>    full stop. torchgw-dijkstra is usable (it internally builds kNN
>    structure) but loses to POT by ~6×.
> 6. **Never leave `M_samples` at default 80** for N < 10k.
> 7. **Momentum α=0.9 (default) is aggressive for noisy gradients** —
>    α=0.1 implements an implicit 10-iter EMA that helps on marginal
>    SNR, but does not rescue truly dense cost at scale.

## C3 — Y-fork branched spiral / Swiss roll (`core/03_branched`)

3D Y-fork Swiss roll → 2D Y-fork spiral, asymmetric tails at 30° (short
tail half the length of long tail). Per-point feature is geodesic
arclen from spiral start; FGW uses this as its linear cost
(`fgw_alpha = 0.5`).

![dataset](../figures/datasets.png)

- **[Symmetry-breaking schematic (2026-04-12)](2026-04-12-symmetry-breaking.md)** —
  why the Y-fork matters: symmetric spirals have GW orientation
  ambiguity; the asymmetric tail + arclen feature breaks it.

- **[6-solver scale benchmark (2026-04-13)](2026-04-13-c3-benchmark.md)** —
  scale sweep `N ∈ {400, …, 20000}`, 6 GPU solvers. **torchgw is 1–2
  orders of magnitude faster than POT at N ≥ 2000 and the only option
  above N = 5000** (POT O(N²) memory wall). All solvers hit tail
  ρ ≥ 0.94. Figures: [`torchgw_vs_pot.png`](../figures/torchgw_vs_pot.png),
  [`e1_solver_shootout.png`](../figures/e1_solver_shootout.png),
  [`e2_scale_sweep.png`](../figures/e2_scale_sweep.png),
  [`rho_by_position.png`](../figures/rho_by_position.png).

- **[Anytime Pareto — quality vs compute (2026-04-14)](2026-04-14-c3-anytime.md)** —
  `max_iter ∈ {5, …, 500}` with `--force-full`. **Almost every solver
  saturates by iter = 5** because the arclen FGW feature locks the
  matching in one outer iteration. Only POT-BAPG (fp64) shows a true
  anytime curve. Figure:
  [`c3_anytime_pareto.png`](../figures/c3_anytime_pareto.png).

- **[Epsilon sensitivity (2026-04-16)](2026-04-16-c3-epsilon.md)** —
  `ε ∈ {5e-4, …, 5e-1}`. **torchgw is essentially ε-immune** (±0.04 ρ
  across four decades). **POT-entropic has a single usable ε** (too
  small → NaN under-flow, too large → collapse to ρ = 0.30). Figure:
  [`c3_eps_sweep.png`](../figures/c3_eps_sweep.png).

## C6 — TACO shape correspondence (`core/06_shape_correspondence`)

Pure GW matching between same-class TACO meshes in different poses,
9 classes × 2 pairs × 3 seeds = 54 (pair, seed) cells. All solvers
run with `fgw_alpha = 1.0` (no linear feature cost).

![TACO dataset](../figures/c6_taco_dataset.png)

- **[TACO benchmark (2026-04-16)](2026-04-16-c6-shape.md)** — principled
  evaluation with two metrics:
  - *Supervised* mean normalised geodesic error (task-aligned)
  - *Unsupervised* pair distortion (GW-native)

  **pot-exact wins supervised by 1.33×** (0.183 vs 0.243 for
  torchgw-dijkstra) and **unsupervised by only 1.12×** (0.063 vs
  0.068). The smaller unsupervised gap shows torchgw's plans optimise
  the GW objective nearly as well as POT's; the extra supervised-metric
  gap is mirror-flip selection (left paw ↔ right paw are geodesically
  equivalent but supervision prefers one). Figure:
  [`c6_principled_eval.png`](../figures/c6_principled_eval.png),
  qualitative
  [`c6_mapping_viz.png`](../figures/c6_mapping_viz.png).

  **Non-trivial ε finding**: the track default `ε = 5e-3` (calibrated
  on C3's FGW setup) is 10× too small for pure GW on symmetric shapes;
  torchgw wants `ε = 5e-2` to break the mirror-optima tie. Supplementary:
  [`c6_hyperparam_sweep.png`](../figures/c6_hyperparam_sweep.png).

## C2 — Single-cell multi-omics (`core/02_single_cell_omics`)

Cross-modality GW alignment on paired 10x PBMC 10k Multiome (11,898
cells × 36,601 genes + 143,887 peaks). Splits modalities, preprocesses
each independently (PCA RNA + cisTopic LDA ATAC), and recovers the
cross-modality correspondence using only within-modality similarity
structure. Ground truth is identity (paired data); metric is FOSCTTM
(Fraction Of Samples Closer Than True Match).

Matches SCOT / SCOT+ preprocessing exactly (cisTopic topic model,
L2-norm, kNN connectivity, hop-count Dijkstra, uniform marginals).

- **[Benchmark + M-samples study (2026-04-17)](2026-04-17-c2-single-cell.md)** —
  5 GPU solvers × N ∈ {1000, 2000, 5000} × 3 seeds.

  Headline table (FOSCTTM, lower = better; literature SCOT+ is 0.12):

  | Solver | N=1000 | N=2000 | N=5000 | wall@5k |
  |---|---|---|---|---|
  | **torchgw-precomputed (M=3N/4)** | **0.140** | **0.136** | **0.134** | 26 s |
  | pot-entropic-gpu                | 0.150 | 0.145 | 0.143 | 57 s |
  | torchgw-landmark (M=3N/4)       | 0.180 | 0.156 | 0.162 | **5 s** |
  | pot-exact-gpu                   | 0.261 | 0.179 | 0.152 | 70 s |
  | torchgw-dijkstra (M=3N/4)       | 0.335 | 0.223 | 0.154 | 414 s ⚠️ |

  **Three headline findings**:

  1. **torchgw-precomputed is the overall winner at every N**
     (lowest FOSCTTM + 2× faster than pot-entropic). Closes gap to
     SCOT+ published 0.12 down to 1.12×.
  2. **The whole `torchgw vs POT` gap was `M_samples` too small**.
     torchgw's default M=80 under-samples the N×N cost matrix on
     small-N tasks and gives catastrophic results (FOSCTTM 0.31 at
     N=5000). Raising to M = 3N/4 flips torchgw from "behind POT" to
     "ahead of POT". Figure: [`c2_msamples_sweep.png`](../figures/c2_msamples_sweep.png).
  3. **Preprocessing dominates solver choice**: LSI (0.25) → sklearn
     LDA (0.16) → **cisTopic (0.14)** — topic-modelling ATAC matters
     more than any solver tuning. We match SCOT+'s recipe exactly and
     compare only the solver layer.

  Additional: `ε = 5e-3` is the sweet spot on C2 (small ε for noisy
  data, opposite of C6). torchgw-dijkstra is Pareto-dominated under
  cisTopic — `precomputed` (best quality) or `landmark` (best speed)
  are the useful torchgw modes.

## C5 — Bilingual word-embedding alignment (`core/05_word_embedding`)

Pure GW alignment between English and Spanish/Finnish fastText vectors.
Intra-lingual cost is mean-normalized cosine distance; evaluation is
P@1/P@5 with CSLS against MUSE bilingual dictionaries. Reproduces
Alvarez-Melis & Jaakkola 2018.

- **[C5 benchmark + structure analysis (2026-04-18)](2026-04-18-c5-word-embedding.md)** —
  5 GPU solvers × 2 pairs {en-es, en-fi} × N ∈ {2000, 5000, 10000}.

  Headline (P@1-CSLS; seed=0, 3-seed rerun pending):

  | Solver | en-es N=5000 | en-fi N=5000 |
  |---|---|---|
  | **pot-entropic-gpu** | **0.495** | **0.155** |
  | pot-exact-gpu | 0.450 | 0.006 |
  | torchgw-dijkstra | 0.086 | 0.001 |
  | torchgw-precomputed | 0.003 | 0.000 |
  | torchgw-landmark | 0.001 | 0.001 |

  **The exploitable-structure thesis** (C5's main contribution):
  > torchgw's MC scalability implicitly assumes cost-matrix structure
  > — sparse, low-rank, or localized. For structured cost, MC sampling
  > naturally lands in informative regions. For **structurally dense**
  > cost (near-Gaussian entries with weak tail signal), the MC gradient
  > becomes an uninformative bootstrap. The algorithm cannot recover.

  **Smoking gun**: at en-es N=5000, torchgw-dijkstra (which builds its
  *own* kNN-sparse geodesic cost internally) reaches P@1=0.086 while
  torchgw-precomputed (fed our dense cosine cost) reaches 0.003 —
  same solver, same N, same ε, ~30× difference **due to cost structure
  alone**.

  **SNR argument**: per-entry Sinkhorn row argmin requires
  SNR ≳ √(2 ln N). At N=2000 with mean-normalized cosine (σ_C ≈ 0.08)
  and M_samples=N, SNR ≈ 2.4 vs threshold 3.9 — the math predicts
  failure quantitatively.

  Two C5-specific sub-findings:
  - **Preprocessing is a 0/1 switch**: `normalize_vecs='both'` +
    `normalize_dists='mean'` (paper's code defaults, not text defaults)
    vs unit+range — en-fi P@1 jumps 12× (0.007 → 0.091) at N=2000.
  - **ε finding**: paper text says ε=5e-5, paper code uses 5e-4. POT
    0.9.6's Sinkhorn does not converge at 5e-5. **5e-4 is the actual
    operating point.**

## Cross-track synthesis

| Axis | C3 (FGW anchor) | C6 (symmetric mesh) | C2 (cross-omics + kNN) | C5 (dense cosine) |
|---|---|---|---|---|
| Who wins accuracy | Tie | POT-exact 1.33× | torchgw-precomputed | **POT-entropic ~150×** |
| Who wins cost | torchgw | torchgw | torchgw | (POT wins accuracy decisively) |
| Cost structure | — (FGW) | dense geodesic mesh | **kNN hop-count** (sparse) | **dense cosine** (near-Gaussian) |
| Best ε | 5e-3 (immune) | 5e-2 | 5e-3 | **5e-4** |
| Best M_samples | 80 OK | 80 OK | 3N/4 required | 3N/4 insufficient |
| Dominant failure | POT OOM | torchgw mirror flip | M_samples floor | **MC SNR below Gumbel threshold** |
| Winning torchgw mode | any | precomputed (tuned) | precomputed | none — abandon sampled_gw |

### The cross-track lesson (revised)

Same architecture (sampled-GW + Sinkhorn) plays out differently against
**four task structures**, and the governing variable is **cost-matrix
structure**, not N:

- **C3 (feature anchor)**: FGW feature locks the answer; Sinkhorn
  diffuseness is harmless.
- **C6 (dense geodesic, symmetric)**: POT-exact's sparse CG commits to
  one mirror; torchgw's diffuse plan averages mirrors.
- **C2 (kNN-sparse geodesic)**: MC sampling naturally hits informative
  entries (few non-zero, long-tailed); torchgw wins.
- **C5 (dense cosine, weak tail)**: MC sampling is a uninformative
  bootstrap; torchgw catastrophically loses to POT.

The axis is not "N size" or "task hardness" but **cost matrix entry
distribution**:
- Long-tailed / sparse / bimodal → torchgw wins (informative sampling)
- Near-Gaussian dense → POT wins (exact bilinear gradient needed)

**This is the unified diagnostic from four tracks**: before deploying
torchgw, plot a histogram of your cost matrix entries. Long tail or
peak-at-zero → go torchgw. Dense Gaussian → stay with POT.

**`M_samples` is torchgw's hidden quality knob.** The default M=80 is
tuned for N >> 10⁴ (where N² is astronomical and sampling is the whole
point). At N < 10⁴ it under-samples and can produce either catastrophic
or randomly-unstable results. Rule of thumb for N ≤ 10⁴:
`M = max(1000, 3N/4)`, capped at N. Cost is essentially flat in M, so
larger is free quality up to saturation.

## Reproducing

```bash
source /scratch/users/chensj16/venvs/dl2025/.venv/bin/activate
cd /scratch/users/chensj16/projects/torchgw-bench

# --- C3 Y-fork FGW benchmark ---
bash scripts/run_c3_benchmark.sh && python scripts/experiments/make_c3_benchmark_plots.py
bash scripts/run_c3_anytime.sh   && python scripts/experiments/make_c3_anytime_plot.py
bash scripts/run_c3_eps_sweep.sh && python scripts/experiments/make_c3_eps_plot.py

# --- C6 TACO shape correspondence ---
bash tracks/core/06_shape_correspondence/fetch.sh  # ~120 MB
python scripts/experiments/run_c6_principled_eval.py
python scripts/experiments/make_c6_principled_plot.py
python scripts/experiments/make_c6_mapping_viz.py

# --- C2 single-cell multi-omics ---
bash tracks/core/02_single_cell_omics/fetch.sh     # ~184 MB
# one-time cisTopic env (R 4.4 + BioC):
micromamba create -n cistopic -c conda-forge -y \
    "r-base>=4.3,<4.5" r-matrix r-plyr r-data.table r-doparallel \
    r-dosnow r-feather r-fitdistrplus r-lda r-remotes r-biocmanager
micromamba run -n cistopic R -e '
  BiocManager::install(c("S4Vectors","GenomicRanges","rtracklayer",
                          "AUCell","RcisTarget"), update=FALSE, ask=FALSE);
  remotes::install_github("aertslab/cisTopic", upgrade="never",
                            dependencies=FALSE)'
# benchmark (first run fits cisTopic LDA ~60 min then caches)
bash scripts/run_c2_cistopic_bench.sh
python scripts/experiments/run_c2_msamples_sweep.py
python scripts/experiments/make_c2_msamples_plot.py
python scripts/experiments/make_c2_sc_plots.py

# --- C5 bilingual word-embedding alignment ---
bash tracks/core/05_word_embedding/fetch.sh    # ~3 GB (en/es/fi fastText + MUSE dicts)
bash scripts/run_c5_bench.sh                   # ~4 hours for 90 cells
python scripts/experiments/run_c5_msamples_sweep.py
python scripts/experiments/make_c5_plots.py

# --- Tests ---
python -m pytest tracks/core/03_branched/tests/ \
                  tracks/core/06_shape_correspondence/tests/ \
                  tracks/core/05_word_embedding/tests/ -v
```
