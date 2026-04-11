# torchgw-bench cross-track conventions

This file is the **sole cross-track contract** in `torchgw-bench`. There is no
shared Python library; the only thing tracks have in common is the rules
below. `scripts/make_report.py` reads these conventions when aggregating
results.

## 1. Directory layout per track

Every track lives in a directory of the form:

```
tracks/<tier>/<NN>_<name>/
```

where `<tier>` is `core`, `extended`, or `gallery`, `<NN>` is a two-digit
ordering prefix (semantically meaningless, only for visual sorting), and
`<name>` is a short snake_case identifier.

Required files inside each non-Gallery track directory:

| File | Purpose |
|------|---------|
| `README.md` | Task description, dataset, baselines, metrics, citations |
| `env.yaml` | Declares which `envs/*.yaml` to activate |
| `run.py` | CLI entry point (see §2) |

Optional files:

| File | Purpose |
|------|---------|
| `fetch.sh` | Downloads track-specific raw data into `data/<track>/` |
| `requirements.txt` | Extra pip packages on top of `env.yaml` |
| `tests/test_run.py` | Unit tests for helpers inside `run.py` |
| `tests/conftest.py` | `sys.path` hook so pytest can import `run.py` |
| `notebooks/` | Exploratory Jupyter notebooks |

Gallery tracks may omit `run.py` in favour of `notebook.ipynb`. Gallery outputs
are static figures embedded in docs, not JSON records.

## 2. `run.py` CLI contract

Every non-Gallery `run.py` **must** accept at least these flags:

```
--solver <solver-id>       # e.g., "torchgw-landmark", "pot-entropic"
--seed <int>               # deterministic seed for stochastic portions
--out <path>               # output directory (usually results/)
```

and **may** accept additional flags such as:

```
--subset small|full        # smoke vs full run
--device cuda|cpu          # device override
--n-source <int>           # per-track scale override
--n-target <int>
```

`run.py` is **self-contained**: it may import any Python package it likes, but
it must **not** import anything from `scripts/` or from sibling tracks.

## 3. Output JSON naming

Each successful (or failed) run writes **exactly one** JSON file:

```
<out>/<tier>_<NN>_<name>__<solver>__seed<N>.json
```

Examples:

```
results/core_01_foundation__torchgw-landmark__seed0.json
results/core_01_foundation__pot-entropic__seed0.json
results/extended_06_domain_adapt__torchgw-fgw__seed2.json
```

## 4. JSON schema (recommended, not enforced)

The reporter parses leniently (`dict.get()`); missing keys are left blank
rather than raising errors. The recommended shape is:

```json
{
  "track": "core/01_foundation",
  "solver": "torchgw-landmark",
  "solver_version": "torchgw==0.4.2+abc1234",
  "seed": 0,
  "subset": "full",
  "timestamp": "2026-04-10T14:32:00Z",
  "host": {
    "gpu": "NVIDIA H100 80GB HBM3",
    "cpu": "AMD EPYC 7763",
    "torch": "2.6.0",
    "cuda": "12.4"
  },
  "status": "ok",
  "error": null,
  "dataset": {
    "name": "spiral_400_swissroll_500",
    "n_source": 400,
    "n_target": 500,
    "source_dim": 2,
    "target_dim": 3
  },
  "hyperparams": {
    "M": 80,
    "epsilon": 0.005,
    "distance_mode": "landmark",
    "max_iter": 300
  },
  "metrics": {
    "correctness": {
      "gw_cost": 0.0234,
      "marginal_error": 1.2e-06
    },
    "task": {
      "spearman_arclen": 0.999
    },
    "efficiency": {
      "wall_s": 1.04,
      "gpu_peak_gb": 0.7,
      "cpu_rss_gb": 1.2,
      "iterations": 218
    },
    "stability": {
      "seed_std_wall_s": null
    }
  },
  "artifacts": {}
}
```

Rules:

- Every record **must** carry `track`, `solver`, `seed`, `status`, and `timestamp`.
- `status` is one of `"ok"`, `"fail"`, `"skip"`.
- On failure, the script should still write a JSON with `"status": "fail"`
  and an `"error"` string, so that the reporter can display the failure.
- On a baseline environment not being available, use `"status": "skip"`.
- `metrics` contains sub-dictionaries; a track fills in whichever subset it
  can compute. The reporter will render empty cells for missing metric keys.

## 5. `env.yaml` format

Each track's `env.yaml` points at a shared conda env by name:

```yaml
# tracks/core/01_foundation/env.yaml
env: base
```

The `run_tier.sh` script activates `tgwbench-<env>` before invoking `run.py`.
If the track also needs a second environment for an isolated baseline (e.g.,
JAX / OTT-JAX), declare `baseline_envs:`:

```yaml
env: base
baseline_envs:
  - jax     # used by run_baseline_ott.py, activated by run_tier.sh in turn
```

## 6. Regression tolerances (for `scripts/diff_report.py`, Phase 6)

These are the default tolerances used when comparing two snapshots:

| Metric family | Tolerance | Meaning |
|---------------|-----------|---------|
| `correctness.*` | ±1% relative | Considered "same" within 1% relative change |
| `task.*` (accuracy / f1 / etc.) | ±0.5% absolute | Considered stable within 0.5% |
| `efficiency.wall_s`, `efficiency.gpu_peak_gb` | +10% | Slowdown up to 10% is tolerated; beyond that is a regression |
| `efficiency.iterations` | +15% | Slight increase tolerated |
| `stability.seed_std_*` | +20% | Noisier results up to 20% are tolerated |

Tracks may override these by documenting the override in their own README.

## 7. Solver naming convention

`--solver` strings are plain identifiers such as:

- `torchgw-landmark`, `torchgw-dijkstra`, `torchgw-precomputed`
- `torchgw-lowrank`
- `torchgw-fgw-landmark`
- `pot-exact`, `pot-entropic`, `pot-fgw`, `pot-lr`
- `cntgw-kpca`, `cntgw-euc`
- `ott-jax-lr`, `ott-jax-sinkhorn`
- `scot`, `pamona`, `paste`, `srm`, `gwl`, ...

Tracks document which `--solver` values they support in their `README.md`.

## 8. Host info (recommended)

Every record should include a `host` block with at least:

- `gpu`: GPU model name (or `"cpu"`)
- `torch`: `torch.__version__`
- `cuda`: `torch.version.cuda` (or `null` on CPU)

Optionally also `cpu`, `hostname`, `python`. The reporter displays the GPU in
the per-track section header.
