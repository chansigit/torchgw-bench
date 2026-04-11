---
title: torchgw-bench — Multi-Discipline Evaluation System Design
status: draft
date: 2026-04-10
owner: Sijie Chen
related_repos: [chansigit/torchgw, chansigit/torchgw-bench (to-be-created)]
---

# torchgw-bench — Multi-Discipline Evaluation System

## 1. Purpose

`torchgw-bench` is a new, **independent repository** that hosts a multi-tier
evaluation system for [torchgw](https://github.com/chansigit/torchgw). It
serves three audiences simultaneously:

1. **Paper-quality benchmark data** — rigorous multi-baseline comparisons on a
   hand-picked set of benchmarks that demonstrate torchgw's correctness, speed,
   and cross-domain reach. Data is produced at a quality sufficient for a paper
   hero table; the reporter does not auto-generate LaTeX in v1 (see §2), so the
   final table is hand-assembled from docs markdown when needed.
2. **Manual release gating** — a reproducible CLI suite that a maintainer runs
   before a release to detect quality/speed regressions against the previous
   snapshot. (No commit-level CI — see §2.)
3. **Public showcase / documentation** — a broad "discipline atlas" that
   backs up the claim *torchgw covers many scientific domains*, materialized as
   per-track README pages, plots, and notebooks.

The unifying goal is to **cover many scientific disciplines** — from synthetic
manifolds through single-cell omics, cryo-EM, medical imaging, 3D shapes, graph
classification, protein structure, neuroimaging, and more — without drowning
torchgw's main repository in benchmark infrastructure, baseline dependencies,
or dataset weights.

## 2. Non-Goals

- **No commit-level CI automation.** All runs are manual. Release gating is a
  human action: `bash scripts/run_tier.sh core && python scripts/make_report.py
  --format diff --from v0.X --to HEAD`.
- **No differentiability benchmarks (v1).** torchgw v0.4.1's exact implicit
  gradient mode is deferred to a future tier; the v1 system focuses on forward
  quality, speed, and stability.
- **No semantic/NLP core tracks.** NLP cross-lingual alignment (MUSE) is
  deliberately kept out of Core/Extended; domain adaptation (Office-31) is
  retained in Extended as an OT-classic benchmark.
- **No plugin framework.** There is no shared Python library across tracks. No
  registry, no base class, no protocol (see §4).
- **No single-environment install.** Baselines have irreconcilable dependency
  conflicts (JAX vs PyTorch-KeOps, R-only scOT, etc.); the system explicitly
  embraces multiple isolated conda environments (see §8).
- **No Paper LaTeX reporter in v1.** Reporter outputs docs markdown, CI JSON,
  and regression diffs only. Paper tables are handcrafted when needed.

## 3. Design Principles

### 3.1 Extreme per-track isolation

Every track is a **self-contained directory** that contains its own
dataset loaders, solver calls, baseline wrappers, and metric computation.
Tracks **do not import each other**, **do not import a shared Python
package**, and **do not share a base class**. The sole cross-track coupling is
a **documented JSON output convention** (see §6.3). This bears the cost of
some code duplication in exchange for:

- Trivial failure isolation: a broken track cannot break other tracks.
- Zero learning curve to add a new track (just a `run.py` script).
- Free pick-and-mix across dependencies (a track can use any library without
  fighting a shared lockfile).
- Easy deletion: removing a track is `rm -rf tracks/.../<name>`, nothing else.

### 3.2 Lenient convention over strict schema

The JSON output convention (§6.3) is specified in `CONVENTIONS.md` as prose.
The reporter reads whatever is present and renders what it finds; missing
fields are left blank rather than raising an error. There is no `jsonschema`
validation step and no `pydantic` model — the reporter is robust to tracks
that only partially fill out the recommended fields.

### 3.3 Tiered coverage with over-provisioning

The evaluation space is partitioned into three tiers of decreasing depth:

| Tier     | Count | Depth                                    | Purpose                       |
|----------|-------|------------------------------------------|-------------------------------|
| Core     | 6     | Multi-seed × multi-baseline × all metrics | Paper hero table, docs main   |
| Extended | 10    | Single seed × 1–2 baselines × main metrics | Breadth, docs secondary, buffer against failures |
| Gallery  | 10+   | One notebook + one figure + README        | Long-tail discipline showcase |

Extended is **deliberately over-provisioned** (10 tracks instead of a minimal
6–7) to absorb the likely failure of 2–3 risky tracks (e.g., E9 GW-FID is a
novel idea, E5 connectomics suffers GW symmetry issues, E7 SMLM data is
specialized). Losing 2–3 Extended tracks is fine; the remaining 7 still
carries the breadth claim.

### 3.4 Reproducibility-first, automation-second

All runs are invoked through `scripts/run_tier.sh`. Every run writes a JSON
file to `results/` with version metadata (torchgw commit hash, dataset hash,
host info). For releases, the `results/` snapshot is frozen into
`artifacts/v<X.Y.Z>/` (git LFS). Regression checks are a manual diff between
two snapshots.

## 4. Architecture Choice

Among three brainstormed options:

- **A — Monolith package with plugin registry** (core library + protocol +
  CLI). Rejected: too much framework overhead for a user who explicitly said
  "不要太复杂".
- **B — Snakemake workflow**. Rejected: DSL learning curve, less Python-idiomatic,
  paper-table generation awkward.
- **C (original) — Git submodule per domain**. Rejected: submodule hell,
  cross-domain reporting pain.
- **C′ — Flat per-domain with shared `shared/` module**. Rejected because the
  user wanted zero shared code.
- **C″ — Zero-shared flat per-domain**. **SELECTED**. Each track is a directory
  with a standalone `run.py`. No shared Python library. The only cross-track
  contract is the JSON output convention in `CONVENTIONS.md`.

## 5. Repository Layout

```
torchgw-bench/                      # New independent repo
├── CONVENTIONS.md                  # Documented cross-track contract (JSON schema + naming + CLI)
├── README.md                       # How to install, run, add tracks
├── LICENSE
├── pyproject.toml                  # Declares python version + ruff/pytest. NOT an installable package.
├── envs/                           # Shared conda env YAMLs (configuration, not code)
│   ├── base.yaml                   # torch + torchgw + POT + numpy/scipy/sklearn/matplotlib
│   ├── jax.yaml                    # JAX + OTT-JAX (isolated)
│   ├── bio.yaml                    # scanpy + anndata + scvi-tools + harmonypy + scOT
│   ├── graph.yaml                  # torch_geometric + dgl + networkx
│   ├── imaging.yaml                # SimpleITK + nibabel + cryosparc-tools + BrainIAK
│   ├── vision3d.yaml               # open3d + trimesh + pymeshlab
│   └── nlp.yaml                    # huggingface + gensim
├── tracks/
│   ├── core/                       # 6 tracks — see §7.1
│   │   ├── 01_foundation/
│   │   ├── 02_single_cell_omics/
│   │   ├── 03_cryoem/
│   │   ├── 04_medical_imaging/
│   │   ├── 05_tu_graph/
│   │   └── 06_shape3d/
│   ├── extended/                   # 10 tracks — see §7.2
│   │   ├── 01_spatial_omics/
│   │   ├── 02_histopath/
│   │   ├── 03_fmri_hyperalign/
│   │   ├── 04_protein_fgw/
│   │   ├── 05_connectome/
│   │   ├── 06_domain_adapt/
│   │   ├── 07_smlm/
│   │   ├── 08_kg_entity/
│   │   ├── 09_gen_model_gwfid/
│   │   └── 10_protein_afdb/
│   └── gallery/                    # 10+ tracks — see §7.3
│       ├── 01_color_transfer/
│       ├── 02_morphometrics/
│       ├── 03_mol_conformers/
│       ├── 04_rl_state_match/
│       ├── 05_weather_ens/
│       ├── 06_finance_net/
│       ├── 07_music_melody/
│       ├── 08_cosmo_halo/
│       ├── 09_hic_contact/
│       └── 10_phylo_tree/
├── scripts/                        # Top-level standalone tools (do NOT import any track)
│   ├── bootstrap_envs.sh           # mamba env create for all envs/*.yaml
│   ├── fetch_data.sh               # Download raw datasets for a tier
│   ├── run_tier.sh                 # Iterate tracks, activate env, run run.py
│   ├── make_report.py              # Scan results/*.json -> docs markdown + CI JSON
│   └── diff_report.py              # Compare two results/ snapshots for regressions
├── data/                           # .gitignore — downloaded raw datasets land here
├── results/                        # .gitignore — per-run JSON outputs land here
│   └── README.md                   # Explains file-naming convention (pure docs)
├── artifacts/                      # git LFS — frozen snapshots per release
│   ├── v0.1.0/
│   └── v0.2.0/
└── docs/                           # Sphinx, published to GitHub Pages
    ├── index.md
    ├── tier_core.md                # Auto-generated by make_report.py
    ├── tier_extended.md
    └── tier_gallery.md
```

**Key layout notes:**

- `pyproject.toml` exists only to let `ruff`/`pytest` target the repo — it is
  **not an installable package**, so there is no `from tgwbench.shared import
  ...` temptation.
- `envs/*.yaml` are configuration files; they are not "shared code".
- `scripts/make_report.py` is a **top-level standalone script**. It reads
  `results/*.json` by inspection, is not imported by any track, and needs only
  `json`, `pathlib`, `jinja2`, and `matplotlib` to run.
- Per-track numeric prefixes (`01_`, `02_`, …) are purely for visual ordering;
  they carry no semantic meaning.

## 6. Per-Track Contract

### 6.1 Directory contents

Every track directory contains:

| File              | Required  | Purpose                                                                     |
|-------------------|-----------|-----------------------------------------------------------------------------|
| `README.md`       | yes       | Task, dataset, baselines, metrics, citations                                |
| `env.yaml`        | yes       | Which `envs/*.yaml` to use (e.g., `env: bio`)                               |
| `run.py`          | yes       | CLI entry point                                                             |
| `fetch.sh`        | optional  | Download track-specific raw data into `../../../data/<track>/`              |
| `requirements.txt`| optional  | Extra pip packages on top of `env.yaml`                                     |
| `notebooks/`      | optional  | Exploratory Jupyter notebooks                                               |

Gallery tracks may **omit `run.py` in favor of `notebook.ipynb`**; their
outputs are static figures embedded in docs rather than JSON records.

### 6.2 `run.py` CLI contract

Every non-Gallery `run.py` accepts at least these flags:

```bash
python tracks/<tier>/<NN>_<name>/run.py \
    --solver <solver-id>       # e.g., "torchgw-landmark", "pot-entropic", "ott-jax-lr"
    --seed <int>               # seed for stochastic portions
    --out <path>               # output directory (typically ../../../results/)
    [--subset small|full]      # optional smoke-vs-full flag
    [--device cuda|cpu]        # optional, defaults to cuda if available
```

The script is **self-contained**: it may import any Python package it likes,
but it must **not** import anything from `torchgw-bench/scripts/` or from
sibling tracks.

### 6.3 JSON output schema (recommended)

Each run writes exactly one JSON file to:

```
results/<tier>_<NN>_<name>__<solver>__seed<N>.json
```

The recommended schema is documented in `CONVENTIONS.md`:

```json
{
  "track": "core/02_single_cell_omics",
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
    "name": "10x_pbmc_multiome",
    "n_source": 9631,
    "n_target": 11022,
    "source_dim": 50,
    "target_dim": 40
  },
  "hyperparams": {
    "M": 80,
    "epsilon": 0.005,
    "distance_mode": "landmark",
    "fgw_alpha": 0.5
  },
  "metrics": {
    "correctness": {
      "gw_cost": 0.0234,
      "gw_cost_vs_pot_relative": 0.008,
      "marginal_error": 1.2e-6
    },
    "task": {
      "label_transfer_accuracy": 0.912,
      "cell_type_f1": 0.883
    },
    "efficiency": {
      "wall_s": 14.3,
      "gpu_peak_gb": 12.4,
      "cpu_rss_gb": 4.1,
      "iterations": 237
    },
    "stability": {
      "seed_std_gw_cost": null
    }
  },
  "artifacts": {
    "transport_plan_path": "results/plans/...",
    "plot_path": "results/plots/..."
  }
}
```

Rules:

- A track fills in the `metrics` sub-trees it can compute; missing keys are
  left out. The reporter renders a blank cell for missing keys.
- **Failures still write a JSON record** with `"status": "fail"` and an
  `"error"` string. This makes failures visible in the report rather than
  silently missing.
- A `"status": "skip"` value signals "baseline env not available" or "data
  missing" — these are also visible in the report but not counted as failures.
- **There is no Python validator for this schema.** CONVENTIONS.md is the only
  source of truth. The reporter uses dict lookups with `.get()` throughout.

### 6.4 `run.py` minimal skeleton

The following ~50-line skeleton can be copy-pasted into each new track:

```python
#!/usr/bin/env python
"""Track: <tier>/<NN>_<name>
<one-line task description>"""
import argparse
import json
import time
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--subset", default="full")
    args = ap.parse_args()

    rec = {
        "track": "<tier>/<NN>_<name>",
        "solver": args.solver,
        "seed": args.seed,
        "subset": args.subset,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {},  # fill in (GPU name, torch version, ...)
        "status": "ok",
        "error": None,
        "dataset": {},
        "hyperparams": {},
        "metrics": {"correctness": {}, "task": {}, "efficiency": {}, "stability": {}},
        "artifacts": {},
    }
    try:
        # --- Track-specific work: load data, run solver, compute metrics ---
        pass
    except Exception as e:
        rec["status"] = "fail"
        rec["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        out_file = args.out / f"<tier>_<NN>_<name>__{args.solver}__seed{args.seed}.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(rec, indent=2))

if __name__ == "__main__":
    main()
```

Tracks may deviate freely — this is a starter, not a requirement.

## 7. Track Catalog

### 7.1 Core (6 tracks — paper-depth)

Every Core track runs with **10 seeds**, **all applicable baselines**, and
**all metric families** (correctness, task quality, efficiency, stability,
robustness).

#### C1 — Foundation: synthetic + methodology

| Aspect        | Details                                                               |
|---------------|-----------------------------------------------------------------------|
| Task          | (a) Spiral → Swiss roll (2D → 3D) with ground-truth arc-length Spearman; (b) Gaussian blobs (3D → 5D); (c) Torus → Sphere manifolds |
| Data          | Fully synthetic (generated in `run.py`, no downloads)                 |
| Scale sweep   | 400×500 → 4k×5k → 20k×25k → 50k×60k (5 sizes)                         |
| torchgw       | `sampled_gw` × {precomputed, dijkstra, landmark} × {mixed_precision on/off} × {multiscale on/off}; `sampled_lowrank_gw` at ≥50k |
| Baselines     | POT exact GW (small only), POT entropic GW, CNT-GW, OTT-JAX LR-GW     |
| Metrics       | `gw_cost_relative_err` vs POT exact, `T_frobenius`, `Spearman(arclen)`, `wall_s`, `gpu_peak_gb`, `iterations`, `time_to_rho_0.99`, seed std, param sensitivity |
| Ablations     | Triton on/off, mixed precision on/off, Lambda EMA on/off, multiscale on/off, distance_mode triple |
| Deliverables  | Scaling plot, ablation table, speedup table, correctness table        |

#### C2 — Single-Cell Multi-Omics Integration

| Aspect        | Details                                                               |
|---------------|-----------------------------------------------------------------------|
| Task          | Cross-omics label transfer via FGW alignment                          |
| Data          | 10x PBMC Multiome, SHARE-seq mouse skin, CITE-seq 10k BMNC            |
| Scale         | 2k → 50k cells                                                        |
| torchgw       | `sampled_gw(fgw_alpha=0.5, C_linear=cosine(features))` with landmark distance |
| Baselines     | POT GW, POT FGW, scOT, Pamona, SCOT, Harmony (non-OT control)         |
| Metrics       | `label_transfer_accuracy`, `cell_type_f1`, `omic_mixing_metric`, `FOSCTTM`, efficiency, dropout/imbalance robustness |
| Deliverables  | Accuracy-vs-scale curve, bar chart vs scOT/Pamona                     |

#### C3 — Cryo-EM Conformational Alignment

| Aspect        | Details                                                               |
|---------------|-----------------------------------------------------------------------|
| Task          | (a) 2D class-average alignment across conformations; (b) 3D volume alignment for heterogeneous reconstruction |
| Data          | EMPIAR-10076 (80S ribosome), EMPIAR-10028 (Thermus 70S), CryoBench synthetic |
| Scale         | 2D: 1k–50k particles; 3D: 100³–200³ voxels                            |
| torchgw       | `sampled_gw` with landmark; optional FGW blending structural + grey-level |
| Baselines     | POT GW, CryoDRGN latent-space matching, RELION class averaging alignment |
| Metrics       | `FSC_0.143`, `fraction_correctly_assigned`, `angular_error_deg`       |
| Risk          | Medium–High (data format and pre-processing are non-trivial)          |
| Deliverables  | FSC curve + assignment accuracy table                                 |

#### C4 — Medical Image Multimodal Registration

| Aspect        | Details                                                               |
|---------------|-----------------------------------------------------------------------|
| Task          | MRI ↔ CT / MRI ↔ PET patch-level OT → deformable warp                 |
| Data          | OASIS brain MRI, Learn2Reg Task 1 (lung CT) and Task 3 (brain MR)     |
| Scale         | 1k–50k key points per scan (voxel down-sampling)                      |
| torchgw       | FGW blending spatial coordinates + intensity features                 |
| Baselines     | POT FGW, NiftyReg B-spline, VoxelMorph (deep-learning control)        |
| Metrics       | `DICE`, `HD95`, `TRE`                                                 |
| Deliverables  | DICE table + TRE scatter + post-registration overlay                  |

#### C5 — TU Graph Classification

| Aspect        | Details                                                               |
|---------------|-----------------------------------------------------------------------|
| Task          | Graph kernel SVM using GW distance                                    |
| Data          | MUTAG (188 graphs, avg 17 nodes), PTC_MR (344 graphs, avg 14 nodes). ENZYMES/PROTEINS are excluded from Core due to per-pair compute cost and may later be added as optional opt-in large benchmarks (not shown in Core summary tables). |
| Scale         | Per-pair 20×20 – 40×40                                                |
| torchgw       | `sampled_gw(distance_mode="precomputed")` with shortest-path          |
| Baselines     | POT GW, CNT-GW (KPCA), GWL, Weisfeiler-Lehman kernel                  |
| Metrics       | 10-fold CV accuracy, pairwise matrix compute time                     |
| Deliverables  | Accuracy table + pairwise compute-time table                          |

#### C6 — 3D Shape Correspondence & Point Cloud Registration

| Aspect        | Details                                                               |
|---------------|-----------------------------------------------------------------------|
| Task          | (a) TOSCA/FAUST non-rigid shape correspondence with GT; (b) ModelNet40 intra-class point-cloud matching |
| Data          | TOSCA-ISO (9 animal classes), FAUST (100 human scans), ModelNet40 (10 classes × 100 items) |
| Scale         | TOSCA: ~3k–50k vertices; ModelNet: 1k–10k points                      |
| torchgw       | `sampled_gw` with landmark distance (natural for geometry)            |
| Baselines     | POT GW, GW-on-mesh (PyMeshLab), BCICP, ICP                            |
| Metrics       | `geodesic_error_cdf`, `correspondence_accuracy@5%_diameter`           |
| Deliverables  | Geodesic error CDF plot, 3D correspondence visualization              |

### 7.2 Extended (10 tracks — breadth + buffer)

Every Extended track runs with **3 seeds**, **1 primary baseline + POT**,
and **main metric families** (correctness where applicable, task quality,
efficiency). Docs have one dedicated section per track.

| #  | Track                           | Data                                 | Metrics                                    | torchgw variant          | Primary baseline            | Risk    |
|----|---------------------------------|--------------------------------------|--------------------------------------------|--------------------------|-----------------------------|---------|
| E1 | Spatial transcriptomics ↔ scRNA | Visium mouse brain, MERFISH, Slide-seqV2 | `label_transfer_acc`, `spatial_coherence_score` | FGW coord + expression    | PASTE / PASTE2              | Low     |
| E2 | Histopathology WSI patch align  | CAMELYON16, TCGA-BRCA                | `patch_overlap_iou`, `tumor_region_agreement` | landmark FGW              | POT FGW, HoVer-Net clustering | Medium  |
| E3 | fMRI inter-subject alignment    | HCP WM task, StudyForrest            | `inter_subject_accuracy`, `time_series_isc` | FGW on parcel features    | SRM, Procrustes             | Medium  |
| E4 | Protein contact-map FGW         | CATH-S40 × SCOPe                     | `fold_retrieval_TP`, `TM_score`             | precomputed FGW           | DALI, TM-align, POT GW      | Low     |
| E5 | Connectomics brain-network align| Allen C.elegans, HCP, Mouse connectome | `node_matching_precision`, `graph_edit_distance` | precomputed GW          | GWL, SpecGW, QAP            | Medium  |
| E6 | Domain Adaptation (Office-31)   | Office-31, MNIST↔USPS                | `target_acc`                                | FGW with feature cost + label soft constraint | POT DA (OT-Laplace), DeepJDOT | Low     |
| E7 | SMLM / dSTORM cross-session     | SMLM Challenge MT0/MT2               | `registration_RMSE`, `cluster_overlap`      | landmark GW, 10k–1M points | POT entropic OT, GMM align | Medium  |
| E8 | KG entity alignment (DBP15k)    | DBP15k ZH/JA/FR-EN                   | `Hit@1/@10`, `MRR`                          | FGW with pretrained emb + neighbor structure | BootEA, RDGCN (non-OT) | Medium  |
| E9 | Generative model GW-FID         | CIFAR-10, FFHQ-256 samples           | Spearman of model ranking vs FID/KID        | GW as generative quality metric | FID, KID (as baselines) | **High** (novel idea) |
| E10| Protein AlphaFold DB alignment  | AlphaFoldDB human → mouse orthologs  | `orthologous_structure_alignment_TP`        | FGW on AF-predicted structures | TM-align, FoldSeek     | Low     |

**Rationale for the E9 high-risk slot**: GW-FID is novel and may not produce
useful rankings. It is deliberately included to absorb the expected 2–3
failures in Extended — if it works, it's a paper contribution; if it fails,
it drops out cleanly and Extended still carries 9 working tracks.

### 7.3 Gallery (10+ tracks — showcase breadth)

Each Gallery track is a minimum-effort deliverable: a README, one notebook,
and one figure. Gallery tracks are **not run through `run_tier.sh`** and
**do not produce JSON records**; they exist purely to populate
`docs/tier_gallery.md`.

| #   | Track                          | Data                                 | Minimal demo                                       |
|-----|--------------------------------|--------------------------------------|----------------------------------------------------|
| G1  | Color transfer                 | Any two images                        | LAB pixel OT → stylized output                     |
| G2  | Morphometric landmarks         | FaceScape / MPG landmarks             | Small-sample Procrustes vs GW comparison           |
| G3  | Molecular conformer space      | GEOM-QM9 (5 molecules)                | Conformer ensemble GW alignment → UMAP             |
| G4  | RL state distribution match    | MuJoCo HalfCheetah expert vs novice   | GW distance as policy similarity metric            |
| G5  | Weather ensemble forecasts     | ECMWF ensemble subset                 | Pairwise ensemble-member GW → clustering dendrogram |
| G6  | Financial correlation networks | S&P 500 daily 2010 vs 2020            | Time-segment correlation GW → "market drift" figure |
| G7  | Music melody alignment         | MAESTRO two piano pieces              | MIDI note-sequence GW → piano roll visualization   |
| G8  | Cosmology halo matching        | IllustrisTNG300 low-res slice         | Halo point-cloud GW across two simulations         |
| G9  | Hi-C contact-map alignment     | ENCODE GM12878 vs K562                | Chromosome N×N matrix GW alignment                  |
| G10 | Phylogenetic tree alignment    | OpenTreeOfLife small clade            | Tree node FGW correspondence                       |

Gallery is **explicitly open to extension**: any new discipline that can be
demonstrated in a single notebook is welcome, with near-zero maintenance cost.

## 8. Baseline Integration Strategy

Four baseline classes are supported, each with a different integration pattern
(selected per-track based on compatibility):

### 8.1 Three access patterns

**Pattern A — Direct import (same env)**  
For pure-Python baselines compatible with PyTorch: POT, CNT-GW (KeOps-based),
simple sklearn-style baselines. `run.py` imports the baseline directly and
runs it in the same Python process as torchgw.

**Pattern B — Isolated env + subprocess**  
For OTT-JAX (JAX/CUDA conflicts with torch-CUDA), R-dependent tools (scOT),
and deep-learning baselines with restrictive dependency trees (VoxelMorph,
CryoDRGN, BrainIAK). The track defines two Python entry points:

```
run.py                 # torchgw part only, env=base
run_baseline_ott.py    # OTT-JAX part, env=jax (separate activation)
```

`scripts/run_tier.sh` iterates over a track's `env.yaml` declaration, which
may list multiple envs to activate in sequence. Each entry point writes its
own JSON; `make_report.py` merges them by `track` key.

**Pattern C — Precomputed artifacts on disk**  
For non-Python tools (DALI, TM-align, NiftyReg) or expensive baselines
(CryoDRGN training runs). The baseline is run offline, results saved to
`data/baseline_precomputed/<track>/<solver>.npz`, and `run.py` reads them as
pre-computed ground truth or comparison points. Each track's README documents
the offline command.

### 8.2 Baseline ↔ track matrix

See **Appendix A** for the full matrix. Key highlights:

- **POT** (pattern A, `base` env) is used by **every Core and most Extended**
  tracks as the "correctness anchor" and as an entropic-GW competitor.
- **CNT-GW** (pattern A, `base` env) is used by C1, C5, C6 — the benchmark
  wrappers already exist in torchgw's `examples/` directory and will be
  ported into the respective tracks.
- **OTT-JAX** (pattern B, `jax` env) is used **only** on the largest scaling
  points of C1 and C2, and on E9 — not every track — to keep the JAX
  environment burden contained.
- **Domain-specific SOTAs** (scOT, PASTE, SRM, DALI, GWL, BootEA, etc.) are
  used **at most once per Extended track** on a per-domain basis.

### 8.3 Graceful degradation

If an isolated env fails to build on the user's machine, the affected
subprocess exits with `"status": "skip"` and an error reason. The primary
torchgw path still runs, and the Extended track keeps its JSON record with
reduced baseline coverage. The reporter renders `skip` rows in grey.

## 9. Environment Management

### 9.1 Tool: conda/mamba

`mamba` is the required environment tool:

- Bio and imaging baselines require conda-only packages (R, SimpleITK, KeOps
  pinned to torch-CUDA).
- `pixi` and `uv` lack the bio/imaging ecosystem coverage.
- `mamba` is 10× faster than `conda` with identical semantics.

### 9.2 Env catalog (duplicated from §5 for reference)

| Env           | Key contents                                          | Used by                                   |
|---------------|-------------------------------------------------------|-------------------------------------------|
| `base`        | torch 2.6, torchgw, POT, numpy, scipy, sklearn, matplotlib | C1, C4, C5, C6, E6, E9, most Gallery      |
| `jax`         | JAX + CUDA 12, OTT-JAX                                | C1 large scales, C2, E9                   |
| `bio`         | base + scanpy + anndata + scvi-tools + harmonypy + scOT-R bridge | C2, E1                                    |
| `graph`       | base + torch_geometric + dgl + networkx + GWL         | C5, E5, E8                                |
| `imaging`     | base + SimpleITK + nibabel + cryosparc-tools + BrainIAK | C3, C4, E2, E3, E7                        |
| `vision3d`    | base + open3d + trimesh + pymeshlab                   | C6, G1, G2                                |
| `nlp`         | base + huggingface + gensim + openea                  | E8                                        |

### 9.3 Bootstrap

`scripts/bootstrap_envs.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
for f in envs/*.yaml; do
    name=$(basename "$f" .yaml)
    mamba env create -f "$f" -n "tgwbench-$name" \
        || mamba env update -f "$f" -n "tgwbench-$name"
done
```

Every track's `env.yaml` points to one of these envs by name, for example:

```yaml
# tracks/core/02_single_cell_omics/env.yaml
env: bio
# optional: additional envs for baselines under pattern B
baseline_envs:
  - jax   # used by run_baseline_ott.py
```

`scripts/run_tier.sh` parses this YAML and activates each env in turn before
running its associated script.

## 10. Reporting Layer (`make_report.py`)

### 10.1 Inputs

- `results/*.json` — all current JSON records.
- Optional `results_prev/*.json` — the previous snapshot, used only with
  `--format diff`.

### 10.2 Three output formats (v1)

**Format 1 — Docs markdown** (`--format docs`)

Renders per-tier markdown pages (`docs/tier_core.md`, `docs/tier_extended.md`,
`docs/tier_gallery.md`) with:

- Per-track section containing task description, dataset, solver/baseline
  comparison table, and any plots found in `results/plots/`.
- Auto-generated markdown headers (`## Core Track: …`).
- Explicit "failed" and "skipped" row annotations.

Example fragment:

```markdown
## Core Track: Single-Cell Multi-Omics

**Task**: Cross-omics label transfer via FGW alignment

**Dataset**: 10x PBMC Multiome (9631 RNA × 11022 ATAC cells)

| Solver              | Label Transfer Acc | Wall Time | GPU Peak |
|---------------------|:------------------:|----------:|---------:|
| POT entropic GW     | 0.901              |      89 s |    14 GB |
| scOT                | 0.887              |     240 s |     8 GB |
| **torchgw (landmark)** | **0.912**       | **12.1 s**| **4.2 GB** |

![scaling plot](plots/core_02_scaling.png)
```

**Format 2 — CI JSON summary** (`--format ci`)

One JSON file summarizing the entire run for manual release inspection:

```json
{
  "version": "0.4.2+abc1234",
  "summary": {"total_tracks": 26, "ok": 24, "fail": 1, "skip": 1},
  "core": {"ok": 6, "fail": 0, "regressions": []},
  "extended": {"ok": 8, "fail": 1, "skip": 1},
  "gallery": {"ok": 10, "fail": 0, "skip": 0},
  "key_metrics": {
    "spiral_400x500_wall_s": 1.04,
    "spiral_4000x5000_wall_s": 14.3,
    "tu_mutag_cv_acc": 0.867
  },
  "failures": [
    {
      "track": "extended/09_gen_model_gwfid",
      "error": "ImportError: ott_jax not available in jax env"
    }
  ]
}
```

**Format 3 — Regression diff** (`--format diff --from <ver> --to <ver>`)

Text-table comparison of two `artifacts/` snapshots:

```
Track                          Metric              v0.4.1    HEAD      Δ         Status
─────────────────────────────────────────────────────────────────────────────────────
core/01_foundation             gw_cost_relative    0.008     0.008     +0.0%     ✓
core/01_foundation             wall_s              14.3      13.9      -2.8%     ✓ faster
core/02_single_cell_omics      label_acc           0.912     0.906     -0.7%     ⚠ within tol
core/05_tu_graph               cv_acc (MUTAG)      0.867     0.851     -1.8%     ✗ REGRESSION
```

Tolerance thresholds (task metric ±0.5%, efficiency +10%, correctness ±1%)
live in `CONVENTIONS.md`.

### 10.3 Implementation notes

- `make_report.py` is a **single standalone script** of ~300–500 lines.
- Dependencies: `json`, `pathlib`, `jinja2`, `matplotlib`. Nothing else.
- It does **not** import anything from `tracks/` or `envs/`.
- It uses `dict.get()` everywhere for lenient parsing — missing JSON keys
  render blank cells.

## 11. Metric Families (v1 scope)

Metrics are grouped into three families. A track implements whichever subset
makes sense for its task; the reporter displays what it finds.

### 11.1 Correctness + task quality

- **Correctness** (vs POT reference where feasible):
  - `gw_cost_relative_err` = |gw_cost - gw_cost_pot| / |gw_cost_pot|
  - `T_frobenius_diff` = ‖T - T_pot‖_F
  - `marginal_error_source`, `marginal_error_target`
- **Task quality** (task-specific):
  - Classification: `accuracy`, `f1`, `AUC`
  - Retrieval: `Precision@k`, `Hit@k`, `MRR`
  - Alignment: `label_transfer_acc`, `FOSCTTM`, `cell_type_f1`
  - Registration: `DICE`, `HD95`, `TRE`
  - Correspondence: `geodesic_error_cdf`, `correspondence_accuracy@x%`
  - Structure: `TM_score`, `fold_retrieval_TP`
  - fMRI: `inter_subject_accuracy`, `time_series_isc`
  - Generative: `Spearman_vs_fid`

### 11.2 Efficiency

- `wall_s` (end-to-end wall clock, excluding dataset loading)
- `gpu_peak_gb` (torch.cuda.max_memory_allocated at peak)
- `cpu_rss_gb` (resource.getrusage)
- `iterations` (outer GW iterations)
- `time_to_accuracy` (seconds to reach 95% of final task metric)

### 11.3 Stability + robustness

- `seed_std_<metric>` (cross-seed standard deviation, computed post-hoc by
  `make_report.py` across multiple seed JSONs)
- Parameter-sensitivity sweeps (for C1 only): vary `epsilon`, `M`,
  `d_landmark` and report metric curves
- Robustness probes (for C2 only in v1): dropout rate, class imbalance

**Differentiability metrics are explicitly out of scope for v1** but may be
added as a fourth family in a future revision.

## 12. Development Workflow

### 12.1 Adding a new track

1. `mkdir tracks/<tier>/<NN>_<name>/`
2. Copy the ~50-line `run.py` skeleton from §6.4 of this spec into the new
   directory.
3. Write `README.md`, `env.yaml`, optional `fetch.sh`.
4. Implement the track-specific work inside `run.py`.
5. Test locally: `python tracks/<tier>/<NN>_<name>/run.py --solver torchgw-x
   --seed 0 --out results/`
6. Verify the output JSON appears in `results/`.
7. `python scripts/make_report.py --format docs --tier <tier>` to see it in
   the rendered docs.
8. Commit the new directory.

### 12.2 Running a tier

```bash
# Bootstrap (once)
bash scripts/bootstrap_envs.sh

# Fetch data for Core
bash scripts/fetch_data.sh core

# Run all Core tracks with seed 0
bash scripts/run_tier.sh core --seed 0

# Generate docs markdown
python scripts/make_report.py --format docs --out docs/

# Manual CI gate
python scripts/make_report.py --format ci --out ci_summary.json
```

### 12.3 Regression check before release

```bash
# Assume v0.4.1 snapshot lives in artifacts/v0.4.1/results/
bash scripts/run_tier.sh core --seed 0
bash scripts/run_tier.sh extended --seed 0

python scripts/diff_report.py \
    --from artifacts/v0.4.1/results \
    --to   results/

# Inspect, decide, then freeze:
cp -r results artifacts/v0.4.2/
git lfs track artifacts/v0.4.2/
git add artifacts/v0.4.2/
git commit -m "snapshot: freeze results for torchgw v0.4.2"
```

## 13. Rollout Plan

The system is built incrementally over **six phases**, each delivering a
visible milestone. Fast iteration beats a big-bang drop.

### Phase 1 — Bootstrap + prove architecture

1. Create `torchgw-bench` GitHub repo.
2. Write `CONVENTIONS.md`, `README.md`, top-level layout.
3. Land `envs/base.yaml` and `envs/bio.yaml`.
4. Implement **C1 Foundation** track (ported from
   `torchgw/examples/benchmark_scale.py`), compared to POT + CNT-GW.
5. Implement `scripts/make_report.py` v1 (docs markdown only).
6. Run C1 end-to-end: fetch → run → report.

**Milestone 1**: `docs/tier_core.md` contains the first table with real numbers.

### Phase 2 — Migrate existing benchmarks

7. Port C5 TU graph classification from `examples/benchmark_tu.py`.
8. Implement C6 TOSCA shape correspondence.
9. Extend `make_report.py` to aggregate 3 tracks.

**Milestone 2**: Core 3/6 tracks are green.

### Phase 3 — Complete Core

10. C2 single-cell multi-omics (requires `bio` env).
11. C4 medical imaging registration (requires `imaging` env).
12. C3 Cryo-EM (highest risk within Core — data wrangling + baseline setup).

**Milestone 3**: Core 6/6 tracks are green.

### Phase 4 — Extended roll-out

13. Extended tracks are added in order of **decreasing confidence**:
    E6 → E1 → E4 → E10 → E3 → E2 → E5 → E7 → E8 → E9.
14. Tracks that fail are marked with `status: fail` but kept in the tree.

**Milestone 4**: At least 7/10 Extended tracks are green (the failure budget
agreed in §3.3 tolerates up to 3 losses).

### Phase 5 — Gallery

15. Batch-write all 10 Gallery notebooks in one focused session.
16. Deploy docs site to GitHub Pages.

**Milestone 5**: All 26 tracks online; Gallery entries are figure-only.

### Phase 6 — Stabilization + release

17. Freeze first snapshot to `artifacts/v0.1.0/`.
18. Implement `scripts/diff_report.py` (format 3).
19. Link `torchgw-bench` docs from the main `torchgw` README and docs site.
20. Tag `v0.1.0` release.

**Milestone 6**: `torchgw-bench v0.1.0` published.

## 14. Open Questions & Risks

### 14.1 Open questions

- **Who owns `torchgw-bench`?** If it stays under Stanford copyright like
  torchgw, the same commercial-license split applies. Needs Stanford OTL
  confirmation before the repo is made public.
- **Data hosting for large assets.** CryoBench, EMPIAR, HCP, AlphaFoldDB
  subsets, Learn2Reg volumes — these cannot live in `data/` (git) and should
  not even live in git LFS for the raw form. Options:
  (a) per-track `fetch.sh` that pulls from upstream,
  (b) a Zenodo mirror under `torchgw-bench/`,
  (c) Hugging Face Datasets hosting.
  Recommendation: start with (a); escalate to (b) or (c) if upstream endpoints
  prove unreliable.
- **Baseline pinning.** Baselines evolve (POT, scanpy, etc.). Pin exact
  versions in `envs/*.yaml` and document the pin rationale in CONVENTIONS.md.
- **Gallery evaluation rigor**: are static figures enough, or should Gallery
  tracks also write a minimal `metadata.json` so they can appear in the
  regression diff? Initial answer: figure-only; revisit if users ask.
- **Snapshot storage cost**: frozen `artifacts/v*/` inside git LFS will grow
  over time. Rotate after N releases, or move to external storage (Zenodo).

### 14.2 Risks

- **Cryo-EM (C3)** is the single highest-risk Core track. Mitigation: if C3
  cannot produce meaningful comparisons within 2 weeks of starting, demote to
  Extended and promote medical imaging's depth.
- **GW-FID (E9)** is a novel and unproven idea. Mitigation: already scoped as
  sacrificial lamb in the Extended buffer.
- **Environment hell** for OTT-JAX / R-based scOT / VoxelMorph may delay
  early phases. Mitigation: baselines are optional in every track — torchgw
  runs even if all baselines are skipped.
- **Maintenance burden scales with track count.** Mitigation: zero-shared
  architecture caps per-track maintenance at O(1); Gallery tracks are
  explicitly marked low-maintenance.
- **Docs drift** when `make_report.py` regenerates `docs/tier_*.md` and
  hand-edited content is lost. Mitigation: all auto-generated files carry a
  `<!-- AUTO-GENERATED — do not edit -->` header; hand-edited content lives
  in sibling files (`docs/tier_core_notes.md`) included by sphinx.

## 15. Success Criteria

v0.1.0 of `torchgw-bench` is successful if:

1. **Core 6/6 tracks run end-to-end** with at least torchgw + POT comparisons.
2. **Extended ≥7/10 tracks** produce JSON records (not necessarily green) —
   the over-provisioning goal is met.
3. **Gallery ≥8/10 tracks** have a committed notebook and figure.
4. `scripts/make_report.py --format docs` produces `docs/tier_core.md` with
   no hand-editing required.
5. `scripts/diff_report.py` can compare v0.1.0 to a hypothetical v0.1.1
   snapshot without crashing on partially missing data.
6. A new discipline (e.g., a G11 seismic waveform track) can be added in
   **under 2 hours of effort** from `mkdir` to figure in docs.

---

## Appendix A — Baseline × Track Matrix

| Baseline                | Pattern | Env     | Tracks                                       |
|-------------------------|---------|---------|----------------------------------------------|
| POT exact GW            | A       | base    | C1 (small), C5, E6                           |
| POT entropic GW         | A       | base    | C1–C6 (all Core), E1–E6, E8–E10              |
| POT FGW                 | A       | base    | C2, C4, E1, E2, E6                           |
| POT OT-DA               | A       | base    | E6                                            |
| CNT-GW (KPCA)           | A       | base    | C1, C5, C6                                    |
| OTT-JAX LR-GW           | B       | jax     | C1 (large), C2 (large), C6 (large), E9       |
| OTT-JAX Sinkhorn        | B       | jax     | C1, C4                                        |
| scOT                    | B       | bio+R   | C2                                            |
| Pamona / SCOT           | A       | bio     | C2                                            |
| Harmony                 | A       | bio     | C2 (non-OT control)                          |
| PASTE / PASTE2          | A       | bio     | E1                                            |
| VoxelMorph              | B       | imaging | C4                                            |
| NiftyReg                | C       | system  | C4                                            |
| SRM (BrainIAK)          | B       | imaging | E3                                            |
| DALI / TM-align         | C       | system  | E4, E10                                       |
| CryoDRGN                | B       | imaging | C3                                            |
| GWL / SpecGW            | B       | graph   | C5, E5                                        |
| BootEA / RDGCN          | B       | nlp     | E8                                            |
| FID / KID               | A       | base    | E9 (as baseline targets)                     |

## Appendix B — Glossary

- **track**: a single evaluation task living in
  `tracks/<tier>/<NN>_<name>/`, self-contained and independent.
- **tier**: one of Core / Extended / Gallery, defining evaluation depth and
  baseline coverage.
- **solver**: a specific algorithm variant being measured (e.g.,
  `torchgw-landmark`, `pot-entropic`, `ott-jax-lr`).
- **record**: the JSON file produced by one `run.py --solver X --seed Y`
  invocation.
- **snapshot**: the full set of records for a release, frozen under
  `artifacts/v<X.Y.Z>/`.
- **pattern A/B/C**: baseline integration strategies (same-env import /
  isolated-env subprocess / precomputed-on-disk).
