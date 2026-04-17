# C3 Y-fork FGW Experiments — Summary

Four experiments on a shared dataset (Y-fork Swiss roll → Y-fork 2D spiral,
asymmetric tails at 30° with short tail half the length of long tail, per-point
feature = geodesic arclen from spiral start). All figures in
[`../figures/`](../figures).

## The dataset in one picture

![dataset](../figures/datasets.png)

3D Swiss roll (left cluster) is the source; 2D spiral (right clusters) is the
target. Each has an asymmetric Y-fork: the long tail extends along the tangent
(label 0), the short tail splays outward at 30° (label 1). Per-point feature is
geodesic arc length from the spiral start, used as the FGW linear cost.

## Experiment index

## C6 — TACO shape correspondence (`core/06_shape_correspondence`)

### [v1 benchmark (2026-04-16)](2026-04-16-c6-shape.md)

Pure GW matching between same-class TACO meshes in different poses. 5
GPU solvers across 18 pairs. **Opposite conclusion from the Y-fork
track**: POT-exact dominates torchgw on accuracy by ~2× because the
sampled-GW design produces diffuse transport plans that fail argmax-
based shape correspondence.

**Figure:** [`c6_shape_benchmark.png`](../figures/c6_shape_benchmark.png).

---

## C3 — Y-fork branched spiral / Swiss roll (`core/03_branched`)

### 1. [Symmetry-breaking schematic (2026-04-12)](2026-04-12-symmetry-breaking.md)

Why the Y-fork matters: symmetric spirals have GW orientation ambiguity (two
optimal plans, mirror images). The asymmetric tail + geodesic-arclen feature
breaks the symmetry and gives the matcher a unique correct answer.

### 2. [6-solver scale benchmark (2026-04-13)](2026-04-13-c3-benchmark.md)

**Question:** Which FGW solver scales best? Runs 3 torchgw variants + 3 POT
variants (× CPU/GPU for POT) across `N ∈ {400, …, 20000}`.

**Figures:** [`torchgw_vs_pot.png`](../figures/torchgw_vs_pot.png),
[`e1_solver_shootout.png`](../figures/e1_solver_shootout.png),
[`e2_scale_sweep.png`](../figures/e2_scale_sweep.png),
[`rho_by_position.png`](../figures/rho_by_position.png).

**Main finding:** torchgw is 1–2 orders of magnitude faster than POT at
N ≥ 2000 and the only option above N=5000 (POT's O(N²) memory blows up). All
6 GPU solvers produce correct matches (backbone-ρ = +1.0 everywhere).

### 3. [Anytime Pareto — quality vs compute (2026-04-14, H100 rerun 2026-04-16)](2026-04-14-c3-anytime.md)

**Question:** Given a compute budget, which solver gives the best ρ? Does
investing more iterations buy proportional quality? Sweeps
`max_iter ∈ {5, …, 500}` with `--force-full` (early stop disabled) at N=4000.

**Figure:** [`c3_anytime_pareto.png`](../figures/c3_anytime_pareto.png).

**Main finding:** Almost every solver saturates by iter=5 — the geodesic-arclen
FGW feature locks the matching within a handful of outer iterations. **Only
POT-BAPG (fp64)** shows a true anytime curve (ρ 0.95 → 0.98). torchgw variants
dominate the Pareto front; POT-exact is dominated (same ρ at ~60× wall time).

### 4. [Epsilon sensitivity (2026-04-16)](2026-04-16-c3-epsilon.md)

**Question:** How robust are ε-regularised FGW solvers to the choice of ε?
Sweeps `ε ∈ {5e-4, …, 5e-1}` (four orders of magnitude) across 5 solvers.

**Figure:** [`c3_eps_sweep.png`](../figures/c3_eps_sweep.png).

**Main finding:** torchgw is essentially ε-immune (±0.04 ρ across 4 decades).
POT-entropic has a **single** usable ε — too small → NaN under-flow, too large
→ ρ collapses to 0.30. POT-BAPG has a sweet spot at ε=5e-3 (ρ=0.983,
competitive with torchgw). The benchmark's `ε=5e-3` default is the only viable
choice for POT.

## Cross-experiment takeaways

1. **torchgw wins the cost axis across the board.** 1–2 orders faster at
   any N ≥ 2000, ε-insensitive, and the only family that scales past N=5000.

2. **POT-exact wins nothing on this dataset.** Comparable ρ to torchgw but
   linearly proportional wall time with no quality return — it's the most
   expensive, least useful choice.

3. **POT-BAPG is surprising.** When configured correctly (fp64 on GPU to
   avoid the BAPG underflow bug documented in the anytime writeup, ε=5e-3
   sweet spot), BAPG is competitive with torchgw on ρ and, on H100, fast
   enough to be practical. On L40S (fp64-throttled gaming-class GPU) it's
   35× slower — hardware matters.

4. **The data-side story is the real one.** Across every experiment, the
   thing that makes the matching work is the **asymmetric Y-fork + geodesic-
   arclen FGW feature**, not any particular solver's cleverness. All 6 GPU
   solvers give ρ ≥ 0.94 on the hard short-tail axis; most give ρ ≥ 0.98.
   The paper-worthy discriminator is **cost**, not correctness.

## Reproducing

```bash
source /scratch/users/chensj16/venvs/dl2025/.venv/bin/activate
cd /scratch/users/chensj16/projects/torchgw-bench

# Scale benchmark (6 solvers × 8 scales × 3–5 seeds, ~60 min on H100)
bash scripts/run_c3_benchmark.sh
python scripts/experiments/make_c3_benchmark_plots.py

# Anytime Pareto (6 solvers × 7 max_iter × 3 seeds, N=4000)
bash scripts/run_c3_anytime.sh
python scripts/experiments/make_c3_anytime_plot.py

# Epsilon sensitivity (5 solvers × 4 eps × 3 seeds, N=4000)
bash scripts/run_c3_eps_sweep.sh
python scripts/experiments/make_c3_eps_plot.py

# Tests
python -m pytest tracks/core/03_branched/tests/ -v
```

## Hardware

All experiments ran on **NVIDIA H100 80GB HBM3**. A partial anytime run on
L40S was archived to `results/c3_anytime_l40s/` for reference (shows the ~35×
H100-vs-L40S speedup on fp64-heavy workloads like POT-BAPG).
