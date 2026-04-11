# Track: core/01_foundation

**Task:** Cross-dimension manifold alignment — align a 2D Archimedean spiral
to a 3D Swiss roll parameterised by the same arc length. Because the
ground-truth correspondence is known (each spiral point `i` matches Swiss
roll point `i`), this track doubles as a correctness anchor for GW solvers
and a methodological backbone for the rest of the benchmark.

## Dataset

Fully synthetic, generated inside `run.py` — nothing to download. Phase 1
ships the smallest size only (N=400, K=500); larger scales (4k/20k/50k) are
added in a later phase.

## Solvers supported in Phase 1

| `--solver` identifier | Library | Mode |
|-----------------------|---------|------|
| `torchgw-landmark` | torchgw | `sampled_gw(distance_mode="landmark", mixed_precision=True)` |
| `pot-entropic` | POT | `ot.gromov.entropic_gromov_wasserstein(epsilon=5e-3)` |

More solvers (POT exact, CNT-GW, OTT-JAX, Triton on/off ablations) are added
in later phases.

## Metrics

Phase 1 records the subset of the JSON schema actually computed:

- `correctness.gw_cost` — final GW cost from the solver log
- `correctness.marginal_error` — `max|T·1 − p|`
- `task.spearman_arclen` — Spearman ρ between source and target angle ranks
  derived from the argmax of `T` per row
- `efficiency.wall_s` — end-to-end wall clock around the solver call
- `efficiency.gpu_peak_gb` — `torch.cuda.max_memory_allocated()` (if CUDA)
- `efficiency.iterations` — outer GW iterations reported by the solver log

## Usage

```bash
conda activate tgwbench-base
python tracks/core/01_foundation/run.py --solver torchgw-landmark --seed 0 --out ../../../results/
python tracks/core/01_foundation/run.py --solver pot-entropic     --seed 0 --out ../../../results/
```

## Citation

Spiral → Swiss roll is the canonical GW sanity benchmark; the generators
are borrowed from `torchgw/examples/benchmark_scale.py`.
