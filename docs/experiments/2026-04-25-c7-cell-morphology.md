# C7 — Cell morphology vs CAJAL (2026-04-25)

> **Status: scaffold — bench runs deferred to SLURM. Numbers below marked
> `<fill>` are populated after Stage A/B sweeps complete. See
> `docs/superpowers/specs/2026-04-25-c7-cell-morphology-design.md` and
> `docs/superpowers/plans/2026-04-25-c7-cell-morphology.md`.**

## Setup

Reused [CAJAL](https://cajal.readthedocs.io/)'s preprocessing (SWC → sample
N points → intracell geodesic distance matrix per cell), then swapped only
the pairwise-GW step across four solvers:

- `cajal-native` — `cajal.run_gw.compute_gw_distance_matrix` defaults
  (CPU + multiprocessing). Internal backend: `<fill from probe>`.
- `pot-entropic-gpu` — POT entropic GW on GPU. Strips the CPU↔GPU axis
  away from the POT↔torchgw axis.
- `pot-exact-gpu` — POT exact CG GW on GPU. Quality reference; only
  viable at `N_per_cell ≤ 200` (CG memory).
- `torchgw-precomputed` — `torchgw.sampled_gw(distance_mode="precomputed")`
  with `M = max(min(N, 1000), ⌈3N/4⌉)`.

Two stages, sample-size sweep `N_per_cell ∈ {50, 200, 500, 1000}`, 3 seeds
each.

| Stage | Source | Cells | Classes |
|---|---|---|---|
| A | NeuroMorpho.org hand-picked | ~300 | 3 (pyramidal / basket / Purkinje) |
| B | Allen Brain Atlas Cell Types DB | ~1000 | dendrite_type {spiny, aspiny, sparsely spiny} |

## Stage A — sanity

![ARI](../figures/c7_stage_a_ari.png)
![wall](../figures/c7_stage_a_wall.png)

Sanity gate (cajal-native at the highest N must reach `ARI_ward > 0.8`):
**`<fill: pass / fail>`**.

## Stage B — benchmark

![ARI](../figures/c7_stage_b_ari.png)
![per-pair](../figures/c7_stage_b_per_pair.png)
![UMAP](../figures/c7_stage_b_umap.png)

Headline table at `N_per_cell = 1000` (mean ± std, 3 seeds):

| solver | ARI_ward | kNN acc (k=5) | wall (s, full N×N) | per-pair (ms) |
|---|---|---|---|---|
| cajal-native       | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| pot-entropic-gpu   | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| pot-exact-gpu      | — (skip > N=200) | — | — | — |
| torchgw-precomputed| `<fill>` | `<fill>` | `<fill>` | `<fill>` |

## The sample-size threshold

Smallest `N_per_cell` at which `torchgw-precomputed` beats `cajal-native`
on full-matrix wall **without losing more than 0.02 absolute ARI_ward**:
**`<fill>`**.

If no such N exists, the conclusion is "torchgw is not the right tool for
many-tiny-GW workloads on this regime; use POT or stay on CAJAL native."
The per-pair latency plot is the smoking gun — torchgw's GPU launch
overhead at small `N_per_cell` exceeds the per-pair compute cost CAJAL
amortizes via CPU multiprocessing.

## Take-home

1. **`<fill: threshold finding or negative-result statement>`**.
2. **CPU↔GPU vs POT↔torchgw decomposition**: `<fill from
   pot-entropic-gpu vs cajal-native vs torchgw-precomputed comparison>`.
3. **"Many tiny GW" deployment rule**: `<fill>`.

## Caveats

- `torchgw` runs pairs serially on GPU; CAJAL parallelizes pairs on CPU.
  The per-pair latency plot is the apples-to-apples comparison; the
  full-matrix plot rewards CAJAL's parallelism. Both shown for honesty
  per spec §10.
- `pot-exact-gpu` is skipped beyond `N_per_cell = 200` (CG memory).
- Manifests pin specific cell IDs (`stage_a_manifest.txt`,
  `stage_b_manifest.txt`); results are reproducible only against those
  exact IDs.
- `cajal-native`'s internal backend is recorded by
  `tracks/core/07_cell_morphology/probe_cajal_backend.py` and quoted
  verbatim above.

## Reproducing

```bash
micromamba activate c7_morph
bash tracks/core/07_cell_morphology/fetch.sh

# Stage A first; verify gate before Stage B
bash scripts/run_c7_stage_a.sh
python scripts/experiments/make_c7_plots.py --stage A

# Stage B (only if Stage A gate passed)
bash scripts/run_c7_stage_b.sh
python scripts/experiments/make_c7_plots.py --stage B
```
