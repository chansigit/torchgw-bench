# torchgw-bench

Multi-discipline evaluation system for [torchgw](https://github.com/chansigit/torchgw).

`torchgw-bench` is a companion repository to torchgw that hosts evaluation
tracks across many scientific disciplines — synthetic manifolds, single-cell
multi-omics, cryo-EM, medical imaging, 3D shapes, graphs, protein structure,
neuroimaging, and more. The tiered design (Core / Extended / Gallery) balances
rigorous paper-quality benchmarks with broad showcase coverage.

See the design spec at [`docs/specs/2026-04-10-torchgw-bench-eval-design.md`](docs/specs/2026-04-10-torchgw-bench-eval-design.md)
and the cross-track contract at [`CONVENTIONS.md`](CONVENTIONS.md).

## Architectural principle: zero shared code

Every track in `tracks/` is a self-contained directory with its own `run.py`.
Tracks never import each other, and there is no shared Python library. The
only cross-track coupling is the **documented JSON output convention** (see
`CONVENTIONS.md`). This keeps per-track maintenance at O(1) and makes it
trivial to add or delete a discipline.

## Install

Requirements: `mamba` (or `conda`) and a local clone of the torchgw source.

```bash
git clone https://github.com/chansigit/torchgw-bench.git
cd torchgw-bench

# Create the conda envs declared in envs/*.yaml
TORCHGW_SRC=/path/to/torchgw bash scripts/bootstrap_envs.sh
```

The bootstrap script creates one conda env per YAML under `envs/` (named
`tgwbench-<env>`) and installs `torchgw` as an editable install from
`$TORCHGW_SRC`.

## Run

```bash
# Activate the env required by the track, run the track, write a JSON record
conda activate tgwbench-base
python tracks/core/01_foundation/run.py \
    --solver torchgw-landmark \
    --seed 0 \
    --out results/

# Run every track in a tier (activates the right env per track)
bash scripts/run_tier.sh core
```

## Report

```bash
# Regenerate the per-tier docs markdown from results/*.json
python scripts/make_report.py --format docs --out docs/
```

The generated `docs/tier_core.md` is under version control (auto-regenerated)
and displays the per-track tables.

## Add a new track

See [`CONVENTIONS.md`](CONVENTIONS.md) for the full per-track contract. The
short version: make a new directory `tracks/<tier>/<NN>_<name>/`, write a
`run.py` following the inline skeleton in the spec §6.4, and make sure it
writes a JSON file matching the naming pattern
`<tier>_<NN>_<name>__<solver>__seed<N>.json`.

## License

Non-commercial source-available (same license as torchgw). See `LICENSE`.
