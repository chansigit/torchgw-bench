# torchgw-bench experiments — index

Comparison of `torchgw` and POT GW / FGW solvers across two tracks:
**C3 Y-fork branched manifold** (FGW with geodesic-arclen feature) and
**C6 TACO shape correspondence** (pure GW on bilaterally-symmetric
meshes). All figures in [`../figures/`](../figures); hardware is
**NVIDIA H100 80GB HBM3** throughout.

## Take-home

> **POT solves "accurate", torchgw solves "tractable".**
>
> On the cost axis, torchgw scales to N ≥ 10⁴ where POT's O(N²)
> memory wall and conditional-gradient per-iter cost put it out of
> reach. On the accuracy axis, POT-exact's sparse conditional-gradient
> plans beat torchgw's Sinkhorn-diffuse plans on tasks that require
> sharp 1-to-1 matching (shape correspondence).
>
> Decision rule:
>
> - **N large (≥ 10k)** → torchgw, no choice.
> - **Feature-anchored FGW** (any N) → torchgw if speed matters, POT
>   otherwise; they reach the same quality.
> - **Pure GW + symmetric data + need sharp matching** → POT-exact.
> - **Pure GW + soft / neighbourhood-level task** → torchgw is fine.

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

## Cross-track synthesis

| Axis | C3 (FGW with features) | C6 (pure GW on symmetric shapes) |
|---|---|---|
| Who wins accuracy | Tie (both ρ ≥ 0.98 at saturation) | POT-exact (1.33× supervised) |
| Who wins cost | torchgw (1–2 orders faster) | torchgw (2–7× faster) |
| torchgw sweet spot ε | 5e-3 (default) — ε-immune anyway | 5e-2 (10× default) |
| Scaling ceiling | torchgw: N=20k, POT: N<5k | both tested at N=2k |
| Dominant failure mode | POT runs out of memory | torchgw picks wrong mirror |

**The cross-track lesson**: the C3 "torchgw crushes POT" story and the
C6 "POT wins on accuracy" story both come from the same algorithmic
design — Sinkhorn-regularised sampled-GW — interacting with different
task structure. With a feature anchor the Sinkhorn diffuseness doesn't
hurt; without one, on symmetric data, it does.

## Reproducing

```bash
source /scratch/users/chensj16/venvs/dl2025/.venv/bin/activate
cd /scratch/users/chensj16/projects/torchgw-bench

# C3 (Y-fork FGW benchmark + anytime + epsilon)
bash scripts/run_c3_benchmark.sh && python scripts/experiments/make_c3_benchmark_plots.py
bash scripts/run_c3_anytime.sh   && python scripts/experiments/make_c3_anytime_plot.py
bash scripts/run_c3_eps_sweep.sh && python scripts/experiments/make_c3_eps_plot.py

# C6 (TACO shape correspondence)
bash tracks/core/06_shape_correspondence/fetch.sh  # ~120 MB
python scripts/experiments/run_c6_principled_eval.py
python scripts/experiments/make_c6_principled_plot.py
python scripts/experiments/make_c6_mapping_viz.py

# Tests
python -m pytest tracks/core/03_branched/tests/ tracks/core/06_shape_correspondence/tests/ -v
```
