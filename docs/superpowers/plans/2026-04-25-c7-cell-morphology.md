# C7 Cell Morphology vs CAJAL — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `tracks/core/07_cell_morphology` that reuses CAJAL's preprocessing (SWC → sample → intracell geodesic distance matrix per cell) and benchmarks four solvers (`cajal-native`, `pot-entropic-gpu`, `pot-exact-gpu`, `torchgw-precomputed`) on the *pairwise GW* step, sweeping `N_per_cell ∈ {50, 200, 500, 1000}` across two stages (NeuroMorpho hand-picked subset, Allen CTDB ~1000 cells).

**Architecture:** Mirror C2/C5: per-track `run.py` writes one JSON per (stage, solver, N_per_cell, seed); a sweep shell script enumerates the matrix; a plotting script consumes the JSONs. The new wrinkle is "many tiny GW" instead of "few large GW" — `run.py` solves all `N_cells × (N_cells - 1) / 2` pairs serially per cell to keep the swap point clean. CAJAL is installed in a dedicated `c7_morph` env so its POT pin cannot break C2/C3/C5/C6.

**Tech Stack:** CAJAL ≥ 1.0 (`pip install cajal`), POT (`pot[gpu]` already in `dl2025`), torchgw, numpy, networkx, scikit-learn (clustering + kNN), umap-learn, matplotlib, pytest.

**Spec reference:** `docs/superpowers/specs/2026-04-25-c7-cell-morphology-design.md` — every numbered section below maps to that spec.

---

### Task 1: Track scaffolding + env bootstrap + CAJAL backend probe

**Files:**
- Create: `tracks/core/07_cell_morphology/README.md`
- Create: `tracks/core/07_cell_morphology/env.yaml`
- Modify: `scripts/bootstrap_envs.sh`
- Create: `tracks/core/07_cell_morphology/probe_cajal_backend.py`

- [ ] **Step 1: One-paragraph README**

```markdown
# C7 — Cell morphology vs CAJAL

Reuses CAJAL's intracell-geodesic preprocessing and swaps only the pairwise-GW
solver. Compares `cajal-native`, `pot-entropic-gpu`, `pot-exact-gpu`, and
`torchgw-precomputed` on two stages (NeuroMorpho hand-picked + Allen CTDB).
See `docs/superpowers/specs/2026-04-25-c7-cell-morphology-design.md` for the
design and `docs/experiments/2026-04-25-c7-cell-morphology.md` for results.
```

- [ ] **Step 2: env.yaml**

```yaml
name: c7_morph
channels: [conda-forge]
dependencies:
  - python=3.11
  - pip
  - pip:
      - cajal
      - "pot[gpu]"
      - torchgw
      - umap-learn
      - scikit-learn
      - psutil
      - matplotlib
      - pytest
```

- [ ] **Step 3: Append bootstrap stanza**

Append to `scripts/bootstrap_envs.sh`:

```bash
# C7 cell morphology — isolated to keep CAJAL's POT pin off C2/C3/C5/C6
if ! micromamba env list | grep -q '^c7_morph '; then
    micromamba env create -f tracks/core/07_cell_morphology/env.yaml -y
fi
```

- [ ] **Step 4: Backend probe script**

```python
"""Print CAJAL's default pairwise-GW backend so the writeup can name it."""
import inspect
import cajal
import cajal.run_gw as run_gw

print(f"cajal version: {cajal.__version__}")
target = run_gw.compute_gw_distance_matrix
print(f"signature: {inspect.signature(target)}")
print(f"defaults  : {target.__defaults__}")
src = inspect.getsource(target)
for kw in ("entropic", "epsilon", "log", "loss_fun", "gromov_wasserstein"):
    if kw in src:
        print(f"mentions {kw!r} in body: yes")
```

- [ ] **Step 5: Run env bootstrap + probe**

```bash
bash scripts/bootstrap_envs.sh
micromamba run -n c7_morph python tracks/core/07_cell_morphology/probe_cajal_backend.py | tee /tmp/cajal_probe.txt
```

Expected: prints version, signature, defaults, and which GW variant CAJAL
calls. **Save the output verbatim** — Task 11's writeup quotes it.

- [ ] **Step 6: Commit**

```bash
git add tracks/core/07_cell_morphology/README.md \
        tracks/core/07_cell_morphology/env.yaml \
        tracks/core/07_cell_morphology/probe_cajal_backend.py \
        scripts/bootstrap_envs.sh
git commit -m "feat(C7): scaffolding + c7_morph env + CAJAL backend probe"
```

---

### Task 2: Cell-ID manifests + fetch script

**Files:**
- Create: `tracks/core/07_cell_morphology/stage_a_manifest.txt`
- Create: `tracks/core/07_cell_morphology/stage_b_manifest.txt`
- Create: `tracks/core/07_cell_morphology/fetch.sh`

Manifests pin the dataset by ID, not by class query (spec §10 caveat 3).

- [ ] **Step 1: Stage A manifest (NeuroMorpho hand-picked)**

300 cells across 3 morphologically distinct classes — 100 each. Format is
TSV: `neuron_name<TAB>class_label`. Choose neurons whose `.swc` is known to
exist on NeuroMorpho.org (verify with HEAD requests in Step 3). Suggested
classes:
- `pyramidal` (cortical pyramidal): 100 IDs from Allen 2010 mouse cortex
  archive on NeuroMorpho
- `basket` (cortical basket): 100 IDs from Markram 2015 BBP archive
- `purkinje` (cerebellar Purkinje): 100 IDs from Smith 2007 archive

Actual IDs go in the file; produce the manifest by querying NeuroMorpho's
search API once and freezing the result. Example first 3 lines:

```
neuron_name	class
H16-03-001-11-08-04_565789419_m	pyramidal
H16-03-001-13-12-01_572319974_m	pyramidal
...
```

- [ ] **Step 2: Stage B manifest (Allen CTDB)**

1000 cells from Allen Brain Atlas Cell Types Database morphology-labelled
release. Fetch the metadata CSV from
`http://celltypes.brain-map.org/api/v2/data/query.csv?...` once; freeze
`(specimen_id, dendrite_type, layer)` joined column. Use `dendrite_type`
(`spiny`, `aspiny`, `sparsely spiny`) as the 3-class label. Example:

```
specimen_id	class
313862167	spiny
314642645	aspiny
...
```

- [ ] **Step 3: fetch.sh — by-ID downloader**

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$REPO_ROOT/data/core_07_cell_morphology"
mkdir -p "$DATA_DIR/swc/stage_a" "$DATA_DIR/swc/stage_b"

fetch_neuromorpho() {
    local manifest="$1" outdir="$2"
    while IFS=$'\t' read -r name cls; do
        [[ "$name" == "neuron_name" ]] && continue
        local out="$outdir/${name}.swc"
        [[ -s "$out" ]] && continue
        curl -fsSL --retry 3 -o "$out" \
          "https://neuromorpho.org/dableFiles/${name}/CNG%20version/${name}.CNG.swc" \
          || { echo "[c7-fetch] WARN: missing $name"; rm -f "$out"; }
    done < "$manifest"
}

fetch_allen() {
    local manifest="$1" outdir="$2"
    while IFS=$'\t' read -r sid cls; do
        [[ "$sid" == "specimen_id" ]] && continue
        local out="$outdir/${sid}.swc"
        [[ -s "$out" ]] && continue
        curl -fsSL --retry 3 -o "$out" \
          "http://api.brain-map.org/api/v2/well_known_file_download/specimen/${sid}/recon.swc" \
          || { echo "[c7-fetch] WARN: missing $sid"; rm -f "$out"; }
    done < "$manifest"
}

fetch_neuromorpho "$SCRIPT_DIR/stage_a_manifest.txt" "$DATA_DIR/swc/stage_a"
fetch_allen       "$SCRIPT_DIR/stage_b_manifest.txt" "$DATA_DIR/swc/stage_b"

# Sanity: at least 80% of each manifest must have downloaded
for stage in stage_a stage_b; do
    expected=$(( $(wc -l < "$SCRIPT_DIR/${stage}_manifest.txt") - 1 ))
    actual=$(ls "$DATA_DIR/swc/$stage"/*.swc 2>/dev/null | wc -l)
    threshold=$(( expected * 80 / 100 ))
    if (( actual < threshold )); then
        echo "[c7-fetch] ERROR: $stage has $actual/$expected (need ≥ $threshold)"
        exit 1
    fi
done
echo "[c7-fetch] done."
```

- [ ] **Step 4: Make executable + dry-run**

```bash
chmod +x tracks/core/07_cell_morphology/fetch.sh
bash tracks/core/07_cell_morphology/fetch.sh
ls data/core_07_cell_morphology/swc/stage_a | wc -l
ls data/core_07_cell_morphology/swc/stage_b | wc -l
```

Expected: ≥ 240 `.swc` in stage_a, ≥ 800 in stage_b.

- [ ] **Step 5: Commit**

```bash
git add tracks/core/07_cell_morphology/{stage_a,stage_b}_manifest.txt \
        tracks/core/07_cell_morphology/fetch.sh
git commit -m "feat(C7): manifests + by-ID fetch script"
```

---

### Task 3: io.py — SWC reader passthrough

**Files:**
- Create: `tracks/core/07_cell_morphology/io.py`
- Create: `tracks/core/07_cell_morphology/tests/__init__.py`
- Create: `tracks/core/07_cell_morphology/tests/conftest.py`
- Create: `tracks/core/07_cell_morphology/tests/test_io.py`

- [ ] **Step 1: Write failing io test**

`tests/conftest.py`:

```python
import sys, pathlib
TRACK = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACK))
```

`tests/test_io.py`:

```python
import textwrap, pathlib
import numpy as np
import io as track_io  # tracks/core/07_cell_morphology/io.py


def test_read_swc_returns_node_array(tmp_path):
    swc = tmp_path / "tiny.swc"
    swc.write_text(textwrap.dedent("""\
        # comment
        1 1 0.0 0.0 0.0 1.0 -1
        2 3 1.0 0.0 0.0 0.5  1
        3 3 1.0 1.0 0.0 0.5  2
        4 3 1.0 1.0 1.0 0.5  3
    """))
    nodes = track_io.read_swc(swc)
    assert nodes.shape == (4, 7)
    assert nodes.dtype == np.float64
    assert nodes[0, 6] == -1
    assert nodes[3, 6] == 3
```

- [ ] **Step 2: Run test, expect failure**

```bash
micromamba run -n c7_morph pytest tracks/core/07_cell_morphology/tests/test_io.py -v
```

Expected: `ModuleNotFoundError` or `AttributeError: module 'io' has no attribute 'read_swc'`.

- [ ] **Step 3: Implement io.py**

```python
"""Thin SWC reader. Returns (n, 7) float64 array of [id, type, x, y, z, r, parent]."""
from __future__ import annotations
import pathlib
import numpy as np


def read_swc(path: str | pathlib.Path) -> np.ndarray:
    rows = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 7:
                continue
            rows.append([float(p) for p in parts[:7]])
    if not rows:
        raise ValueError(f"no node rows in {path}")
    return np.asarray(rows, dtype=np.float64)
```

- [ ] **Step 4: Re-run tests**

```bash
micromamba run -n c7_morph pytest tracks/core/07_cell_morphology/tests/test_io.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tracks/core/07_cell_morphology/io.py \
        tracks/core/07_cell_morphology/tests/__init__.py \
        tracks/core/07_cell_morphology/tests/conftest.py \
        tracks/core/07_cell_morphology/tests/test_io.py
git commit -m "feat(C7): io.py SWC reader + tests"
```

---

### Task 4: intracell.py — per-cell geodesic D_i with disk cache

**Files:**
- Create: `tracks/core/07_cell_morphology/intracell.py`
- Create: `tracks/core/07_cell_morphology/tests/test_intracell.py`

- [ ] **Step 1: Write failing test**

```python
import numpy as np, pathlib
import io as track_io
import intracell


def test_compute_intracell_returns_square(tmp_path):
    # build a 5-node Y-fork SWC: linear 0-1-2 + branch 1-3-4
    swc = tmp_path / "y.swc"
    swc.write_text(
        "1 1 0 0 0 1 -1\n"
        "2 3 1 0 0 1  1\n"
        "3 3 2 0 0 1  2\n"
        "4 3 1 1 0 1  2\n"
        "5 3 1 2 0 1  4\n"
    )
    D = intracell.compute_intracell(swc, n_per_cell=4, seed=0,
                                    cache_dir=tmp_path / "cache")
    assert D.shape == (4, 4)
    assert np.allclose(D, D.T, atol=1e-6)
    assert (np.diag(D) == 0).all()
    # cache hit: second call must return identical
    D2 = intracell.compute_intracell(swc, n_per_cell=4, seed=0,
                                     cache_dir=tmp_path / "cache")
    assert np.array_equal(D, D2)
```

- [ ] **Step 2: Run test, expect failure**

```bash
micromamba run -n c7_morph pytest tracks/core/07_cell_morphology/tests/test_intracell.py -v
```

- [ ] **Step 3: Implement intracell.py — call CAJAL, cache by content hash**

```python
"""Compute the per-cell intracell geodesic distance matrix using CAJAL."""
from __future__ import annotations
import hashlib
import pathlib
import numpy as np


def _cache_key(swc_path: pathlib.Path, n_per_cell: int, seed: int) -> str:
    h = hashlib.sha256()
    h.update(pathlib.Path(swc_path).read_bytes())
    h.update(f"|n={n_per_cell}|seed={seed}".encode())
    return h.hexdigest()[:16]


def compute_intracell(
    swc_path: str | pathlib.Path,
    n_per_cell: int,
    seed: int,
    cache_dir: str | pathlib.Path,
) -> np.ndarray:
    swc_path = pathlib.Path(swc_path)
    cache_dir = pathlib.Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(swc_path, n_per_cell, seed)
    cache_file = cache_dir / f"{swc_path.stem}__n{n_per_cell}__s{seed}__{key}.npy"
    if cache_file.exists():
        return np.load(cache_file)
    # CAJAL pipeline: load SWC → sample → geodesic distance matrix
    from cajal.swc import read_swc as cajal_read_swc
    from cajal.sample_swc import icdm_geodesic
    forest = cajal_read_swc(str(swc_path))
    rng = np.random.default_rng(seed)
    D = icdm_geodesic(forest, n_sample=n_per_cell, rng=rng).astype(np.float64)
    np.save(cache_file, D)
    return D
```

> **Note:** CAJAL's exact public API names may differ slightly between
> versions. If `cajal.sample_swc.icdm_geodesic` does not exist, run
> `python -c "import cajal.sample_swc as m; print(dir(m))"` and pick the
> equivalent (typical names: `geodesic_distance_matrix`,
> `compute_intracell_distance`). Update both the import and the call site;
> do **not** add a fallback chain — fail loudly if the API moved.

- [ ] **Step 4: Re-run test**

```bash
micromamba run -n c7_morph pytest tracks/core/07_cell_morphology/tests/test_intracell.py -v
```

Expected: PASS. If it fails on the import, follow the note above.

- [ ] **Step 5: Commit**

```bash
git add tracks/core/07_cell_morphology/intracell.py \
        tracks/core/07_cell_morphology/tests/test_intracell.py
git commit -m "feat(C7): intracell.py — CAJAL geodesic D_i with content-hash cache"
```

---

### Task 5: eval.py — ARI/NMI/kNN from N×N distance matrix

**Files:**
- Create: `tracks/core/07_cell_morphology/eval.py`
- Create: `tracks/core/07_cell_morphology/tests/test_eval.py`

- [ ] **Step 1: Write failing test**

```python
import numpy as np
import eval as track_eval


def _block_distance_matrix(n_per_class: int = 10, n_classes: int = 3,
                           sep: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    classes = np.repeat(np.arange(n_classes), n_per_class)
    points = rng.normal(size=(len(classes), 2)) + sep * classes[:, None]
    D = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    return D, classes


def test_eval_block_recovers_clusters():
    D, y = _block_distance_matrix()
    out = track_eval.eval_distance_matrix(D, y, k_classes=3, knn_k=5)
    assert out["ARI_ward"] > 0.95
    assert out["NMI_ward"] > 0.90
    assert out["knn_acc_k5"] > 0.95


def test_eval_random_is_chance():
    rng = np.random.default_rng(1)
    D = rng.uniform(size=(30, 30)); D = (D + D.T) / 2; np.fill_diagonal(D, 0)
    y = np.repeat(np.arange(3), 10)
    out = track_eval.eval_distance_matrix(D, y, k_classes=3, knn_k=5)
    assert out["ARI_ward"] < 0.30
    assert out["knn_acc_k5"] < 0.55
```

- [ ] **Step 2: Run test, expect failure**

```bash
micromamba run -n c7_morph pytest tracks/core/07_cell_morphology/tests/test_eval.py -v
```

- [ ] **Step 3: Implement eval.py**

```python
"""Downstream evaluation from an N×N GW distance matrix + ground-truth labels."""
from __future__ import annotations
import numpy as np


def eval_distance_matrix(
    D: np.ndarray,
    labels: np.ndarray,
    k_classes: int,
    knn_k: int = 5,
) -> dict:
    from sklearn.cluster import AgglomerativeClustering, SpectralClustering
    from sklearn.metrics import (
        adjusted_rand_score, normalized_mutual_info_score,
        accuracy_score, f1_score,
    )
    from sklearn.neighbors import KNeighborsClassifier

    # Ward needs vector input or precomputed with linkage='average';
    # use 'average' on precomputed distances (Ward requires Euclidean).
    ward = AgglomerativeClustering(
        n_clusters=k_classes, metric="precomputed", linkage="average"
    ).fit_predict(D)

    # Spectral on similarity = exp(-D / median(D))
    med = float(np.median(D[D > 0])) if np.any(D > 0) else 1.0
    S = np.exp(-D / max(med, 1e-12))
    spec = SpectralClustering(
        n_clusters=k_classes, affinity="precomputed",
        assign_labels="kmeans", random_state=0,
    ).fit_predict(S)

    # Leave-one-out kNN on the precomputed distance matrix
    knn = KNeighborsClassifier(n_neighbors=knn_k, metric="precomputed")
    n = D.shape[0]
    preds = np.empty(n, dtype=labels.dtype)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        D_train = D[np.ix_(mask, mask)]
        knn.fit(D_train, labels[mask])
        preds[i] = knn.predict(D[i:i+1, mask])[0]

    return {
        "ARI_ward":     float(adjusted_rand_score(labels, ward)),
        "NMI_ward":     float(normalized_mutual_info_score(labels, ward)),
        "ARI_spectral": float(adjusted_rand_score(labels, spec)),
        "NMI_spectral": float(normalized_mutual_info_score(labels, spec)),
        "knn_acc_k5":   float(accuracy_score(labels, preds)),
        "knn_macro_f1_k5": float(f1_score(labels, preds, average="macro")),
    }
```

- [ ] **Step 4: Re-run test**

```bash
micromamba run -n c7_morph pytest tracks/core/07_cell_morphology/tests/test_eval.py -v
```

Expected: PASS both.

- [ ] **Step 5: Commit**

```bash
git add tracks/core/07_cell_morphology/eval.py \
        tracks/core/07_cell_morphology/tests/test_eval.py
git commit -m "feat(C7): eval.py — ARI/NMI + leave-one-out kNN from N×N D"
```

---

### Task 6: solvers.py — pairwise GW dispatch (4 solvers)

**Files:**
- Create: `tracks/core/07_cell_morphology/solvers.py`
- Create: `tracks/core/07_cell_morphology/tests/test_solvers.py`

This module exposes one function per solver, each taking two `D_i, D_j`
arrays (already CAJAL-preprocessed) and returning a single scalar GW
distance plus per-pair wall time. The full N×N loop lives in `run.py`.

- [ ] **Step 1: Write failing test**

```python
import numpy as np
import solvers


def _two_blocks(n: int = 30):
    rng = np.random.default_rng(0)
    A = rng.uniform(size=(n, n)); A = (A + A.T) / 2; np.fill_diagonal(A, 0)
    return A, A.copy()  # identical → GW should be ~0


def test_pot_entropic_identical_is_small():
    D1, D2 = _two_blocks()
    out = solvers.gw_pair("pot-entropic-gpu", D1, D2,
                          epsilon=5e-3, M_samples=None, seed=0)
    assert out["gw"] < 1e-2
    assert out["wall_s"] > 0


def test_torchgw_identical_is_small():
    D1, D2 = _two_blocks()
    out = solvers.gw_pair("torchgw-precomputed", D1, D2,
                          epsilon=5e-3, M_samples=20, seed=0)
    assert out["gw"] < 5e-2  # MC noise floor
```

- [ ] **Step 2: Run test, expect failure**

```bash
micromamba run -n c7_morph pytest tracks/core/07_cell_morphology/tests/test_solvers.py -v
```

- [ ] **Step 3: Implement solvers.py**

```python
"""Single-pair GW dispatch for the four C7 solvers."""
from __future__ import annotations
import time
import numpy as np


def _uniform(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n, dtype=np.float64)


def _gw_pot_entropic(D1, D2, epsilon, seed):
    import ot, torch
    a, b = _uniform(D1.shape[0]), _uniform(D2.shape[0])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C1 = torch.as_tensor(D1, dtype=torch.float32, device=dev)
    C2 = torch.as_tensor(D2, dtype=torch.float32, device=dev)
    pa = torch.as_tensor(a,  dtype=torch.float32, device=dev)
    pb = torch.as_tensor(b,  dtype=torch.float32, device=dev)
    T, log = ot.gromov.entropic_gromov_wasserstein(
        C1, C2, pa, pb, "square_loss",
        epsilon=epsilon, log=True, max_iter=500,
    )
    return float(log.get("gw_dist", log.get("loss", float("nan"))))


def _gw_pot_exact(D1, D2, seed):
    import ot, torch
    a, b = _uniform(D1.shape[0]), _uniform(D2.shape[0])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C1 = torch.as_tensor(D1, dtype=torch.float32, device=dev)
    C2 = torch.as_tensor(D2, dtype=torch.float32, device=dev)
    pa = torch.as_tensor(a,  dtype=torch.float32, device=dev)
    pb = torch.as_tensor(b,  dtype=torch.float32, device=dev)
    T, log = ot.gromov.gromov_wasserstein(
        C1, C2, pa, pb, "square_loss", log=True, max_iter=500,
    )
    return float(log.get("gw_dist", log.get("loss", float("nan"))))


def _gw_torchgw_precomputed(D1, D2, epsilon, M_samples, seed):
    import torch, torchgw
    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C1 = torch.as_tensor(D1, dtype=torch.float32, device=dev)
    C2 = torch.as_tensor(D2, dtype=torch.float32, device=dev)
    n, m = C1.shape[0], C2.shape[0]
    pa = torch.full((n,), 1.0 / n, device=dev)
    pb = torch.full((m,), 1.0 / m, device=dev)
    M = M_samples if M_samples is not None else max(min(n, 1000), 3 * n // 4)
    T, log = torchgw.sampled_gw(
        C1, C2, pa, pb,
        distance_mode="precomputed",
        epsilon=epsilon, M_samples=M, log=True,
    )
    return float(log.get("gw_cost", float("nan")))


def _gw_cajal_native(D1, D2, seed):
    """Single-pair through CAJAL's GW. CAJAL works at the matrix-of-pairs
    level (not single pair); for the test path we call its low-level POT
    wrapper directly. The real run.py path goes through batch CAJAL — see
    `gw_full_matrix_cajal`."""
    import ot
    a, b = _uniform(D1.shape[0]), _uniform(D2.shape[0])
    T, log = ot.gromov.gromov_wasserstein(
        D1.astype(np.float64), D2.astype(np.float64), a, b,
        "square_loss", log=True, max_iter=500,
    )
    return float(log.get("gw_dist", log.get("loss", float("nan"))))


_DISPATCH = {
    "pot-entropic-gpu":      lambda D1, D2, epsilon, M, seed:
                                 _gw_pot_entropic(D1, D2, epsilon, seed),
    "pot-exact-gpu":         lambda D1, D2, epsilon, M, seed:
                                 _gw_pot_exact(D1, D2, seed),
    "torchgw-precomputed":   _gw_torchgw_precomputed,
    "cajal-native":          lambda D1, D2, epsilon, M, seed:
                                 _gw_cajal_native(D1, D2, seed),
}


def gw_pair(solver: str, D1: np.ndarray, D2: np.ndarray, *,
            epsilon: float, M_samples: int | None, seed: int) -> dict:
    if solver not in _DISPATCH:
        raise ValueError(f"unknown solver {solver!r}")
    t0 = time.perf_counter()
    val = _DISPATCH[solver](D1, D2, epsilon, M_samples, seed)
    return {"gw": val, "wall_s": time.perf_counter() - t0}


def gw_full_matrix_cajal(D_list: list[np.ndarray], n_jobs: int = -1) -> np.ndarray:
    """The real cajal-native path: hand the whole list of D_i to CAJAL's
    parallel pairwise routine so we measure CAJAL's actual end-to-end speed
    (multiprocessing across pairs) rather than the per-pair POT cost."""
    from cajal.run_gw import gw_pairwise_parallel
    n = len(D_list)
    M = gw_pairwise_parallel(D_list, [_uniform(d.shape[0]) for d in D_list],
                             n_processes=n_jobs)
    return np.asarray(M, dtype=np.float64)
```

> **Note:** `cajal.run_gw.gw_pairwise_parallel` argument names may shift
> across CAJAL versions. Use the probe output from Task 1 Step 5 to confirm
> the call signature; adjust both `import` and call accordingly. If CAJAL
> uses a `combined_metric_lookup` style API, switch to that — the principle
> is "let CAJAL drive its own batch, measure end-to-end."

- [ ] **Step 4: Re-run test**

```bash
micromamba run -n c7_morph pytest tracks/core/07_cell_morphology/tests/test_solvers.py -v
```

Expected: both PASS. If `torchgw.sampled_gw`'s kwarg names differ, copy the
working call from `tracks/core/02_single_cell_omics/run.py`.

- [ ] **Step 5: Commit**

```bash
git add tracks/core/07_cell_morphology/solvers.py \
        tracks/core/07_cell_morphology/tests/test_solvers.py
git commit -m "feat(C7): solvers.py — 4-solver GW dispatch + CAJAL parallel path"
```

---

### Task 7: run.py — full-matrix benchmark per (stage, solver, N, seed)

**Files:**
- Create: `tracks/core/07_cell_morphology/run.py`

This is the entry point invoked by the sweep script. One run = one JSON.

- [ ] **Step 1: Write run.py**

```python
#!/usr/bin/env python
"""C7 cell-morphology benchmark — one (stage, solver, N_per_cell, seed) cell."""
from __future__ import annotations
import argparse, json, os, pathlib, socket, threading, time
import datetime as _dt
import numpy as np

TRACK = pathlib.Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(TRACK))

import io as track_io  # noqa: F401  (kept for symmetry with other tracks)
import intracell, eval as track_eval, solvers


def _read_manifest(stage: str) -> tuple[list[pathlib.Path], np.ndarray, list[str]]:
    repo = TRACK.parents[2]
    manifest = TRACK / f"{stage}_manifest.txt"
    swc_dir = repo / "data" / "core_07_cell_morphology" / "swc" / stage
    paths, lbls = [], []
    with open(manifest) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("neuron_name") or line.startswith("specimen_id"):
                continue
            name, cls = line.split("\t")
            p = swc_dir / f"{name}.swc"
            if p.exists():
                paths.append(p); lbls.append(cls)
    classes_sorted = sorted(set(lbls))
    cls_to_int = {c: i for i, c in enumerate(classes_sorted)}
    y = np.asarray([cls_to_int[c] for c in lbls], dtype=np.int64)
    return paths, y, classes_sorted


def _peak_rss_gb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 2**30
    except ImportError:
        return float("nan")


def _gw_full_matrix(solver: str, D_list, *, epsilon, M_samples, seed):
    """Build N×N GW matrix. cajal-native uses CAJAL's parallel batch; the
    other three loop pairs serially (no batched-small-GW API in torchgw)."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    n = len(D_list)
    pair_walls = []
    if solver == "cajal-native":
        t0 = time.perf_counter()
        M = solvers.gw_full_matrix_cajal(D_list)
        wall_total = time.perf_counter() - t0
        # CAJAL gives the full matrix at once; we estimate per-pair wall from total.
        pair_walls = [wall_total / max(n * (n - 1) / 2, 1)] * (n * (n - 1) // 2)
    else:
        M = np.zeros((n, n), dtype=np.float64)
        t0 = time.perf_counter()
        for i in range(n):
            for j in range(i + 1, n):
                out = solvers.gw_pair(solver, D_list[i], D_list[j],
                                      epsilon=epsilon, M_samples=M_samples, seed=seed)
                M[i, j] = M[j, i] = out["gw"]
                pair_walls.append(out["wall_s"])
        wall_total = time.perf_counter() - t0
    gpu_peak = (torch.cuda.max_memory_allocated() / 2**30
                if torch.cuda.is_available() else None)
    return M, {
        "wall_full_matrix_s": float(wall_total),
        "wall_per_pair_ms":   float(np.mean(pair_walls) * 1000),
        "gpu_peak_gb":        gpu_peak,
        "cpu_peak_gb":        _peak_rss_gb(),
        "n_pairs":            int(n * (n - 1) // 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["A", "B"])
    ap.add_argument("--solver", required=True, choices=[
        "cajal-native", "pot-entropic-gpu", "pot-exact-gpu", "torchgw-precomputed",
    ])
    ap.add_argument("--n-per-cell", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epsilon", type=float, default=5e-3)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    if args.solver == "pot-exact-gpu" and args.n_per_cell > 200:
        print(f"[c7] skip pot-exact-gpu at n_per_cell={args.n_per_cell} > 200")
        return

    stage = f"stage_{args.stage.lower()}"
    paths, y, classes = _read_manifest(stage)
    cache_dir = (TRACK.parents[2] / "results" / "c7_cell_morphology"
                 / "_intracell_cache" / stage)

    rec = {
        "track": "core/07_cell_morphology",
        "stage": stage, "solver": args.solver,
        "n_per_cell": args.n_per_cell, "seed": args.seed,
        "epsilon": args.epsilon,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "n_cells": len(paths), "classes": classes,
        "status": "ok", "error": None,
        "metrics": {}, "efficiency": {},
    }
    out_file = args.out / (
        f"core_07_cell_morphology__{args.solver}__{stage}"
        f"__n{args.n_per_cell}__seed{args.seed}.json"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        D_list = [intracell.compute_intracell(p, args.n_per_cell, args.seed, cache_dir)
                  for p in paths]
        M, eff = _gw_full_matrix(args.solver, D_list,
                                 epsilon=args.epsilon, M_samples=None, seed=args.seed)
        ev = track_eval.eval_distance_matrix(M, y, k_classes=len(classes), knn_k=5)
        rec["metrics"]    = ev
        rec["efficiency"] = eff
        # save matrix only for seed 0 to keep results dir small
        if args.seed == 0:
            np.save(out_file.with_suffix(".npy"), M)
    except Exception as e:
        rec["status"] = "fail"; rec["error"] = f"{type(e).__name__}: {e}"

    with open(out_file, "w") as fh:
        json.dump(rec, fh, indent=2, default=str)
    print(f"[c7] wrote {out_file} (status={rec['status']})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test on tiny subset**

```bash
mkdir -p /tmp/c7_smoke
# Build a temporary 10-cell smoke manifest from stage_a
head -11 tracks/core/07_cell_morphology/stage_a_manifest.txt \
    > /tmp/stage_smoke_manifest.txt
cp /tmp/stage_smoke_manifest.txt tracks/core/07_cell_morphology/stage_smoke_manifest.txt

# Run a 10-cell × N=50 smoke for the fastest solver (pot-entropic-gpu)
micromamba run -n c7_morph python tracks/core/07_cell_morphology/run.py \
    --stage A --solver pot-entropic-gpu --n-per-cell 50 --seed 0 \
    --out /tmp/c7_smoke

cat /tmp/c7_smoke/core_07_cell_morphology__pot-entropic-gpu__stage_a__n50__seed0.json | python -m json.tool | head -30
rm tracks/core/07_cell_morphology/stage_smoke_manifest.txt
```

Expected JSON contains `"status": "ok"`, all six metric keys, and non-zero
`wall_full_matrix_s`. If it fails on the cajal-native or torchgw paths,
fix per Task 6's notes; do **not** add try/except fallbacks.

- [ ] **Step 3: Commit**

```bash
git add tracks/core/07_cell_morphology/run.py
git commit -m "feat(C7): run.py — full-matrix bench per (stage, solver, N, seed)"
```

---

### Task 8: Stage A sweep script + Stage A run

**Files:**
- Create: `scripts/run_c7_stage_a.sh`

- [ ] **Step 1: Write sweep script**

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT="$REPO_ROOT/results/c7_cell_morphology"
mkdir -p "$OUT"

SOLVERS=(cajal-native pot-entropic-gpu pot-exact-gpu torchgw-precomputed)
N_VALUES=(50 200 500 1000)
SEEDS=(0 1 2)

for solver in "${SOLVERS[@]}"; do
    for n in "${N_VALUES[@]}"; do
        # spec §5: pot-exact-gpu skipped above N=200
        if [[ "$solver" == "pot-exact-gpu" && "$n" -gt 200 ]]; then continue; fi
        for seed in "${SEEDS[@]}"; do
            tag="${solver}__stage_a__n${n}__seed${seed}"
            json="$OUT/core_07_cell_morphology__${tag}.json"
            if [[ -s "$json" ]]; then
                echo "[c7-A] skip done: $tag"; continue
            fi
            echo "[c7-A] running $tag"
            micromamba run -n c7_morph python \
                "$REPO_ROOT/tracks/core/07_cell_morphology/run.py" \
                --stage A --solver "$solver" --n-per-cell "$n" \
                --seed "$seed" --out "$OUT" \
                2>&1 | tee -a "$REPO_ROOT/logs/c7_stage_a.log"
        done
    done
done
echo "[c7-A] done."
```

- [ ] **Step 2: Make executable + run**

```bash
chmod +x scripts/run_c7_stage_a.sh
mkdir -p logs
bash scripts/run_c7_stage_a.sh
```

Expected: ~6 h on a single H100. The cajal-native × N=1000 cell is the long
pole. Verify all expected JSONs land:

```bash
ls results/c7_cell_morphology/*stage_a*.json | wc -l
# expected: 4 solvers × 3 seeds × {50,200,500,1000}
#   minus pot-exact-gpu × {500,1000} skipped
#   = 4*3*4 - 1*3*2 = 48 - 6 = 42
```

- [ ] **Step 3: Stage A success gate (spec §11)**

```bash
python -c "
import json, glob
for p in sorted(glob.glob('results/c7_cell_morphology/*cajal-native*stage_a*seed0.json')):
    d = json.load(open(p))
    print(p.split('__')[-3], d['metrics'].get('ARI_ward'))
"
```

Expected: cajal-native ARI_ward > 0.8 at the highest N_per_cell. If not,
**stop here, do not proceed to Stage B** — pipeline is broken; debug.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_c7_stage_a.sh
git commit -m "feat(C7): Stage A sweep script (NeuroMorpho 300 × 4 solvers × 4 N × 3 seeds)"
```

---

### Task 9: Stage A plotting

**Files:**
- Create: `scripts/experiments/make_c7_plots.py`

- [ ] **Step 1: Write plotting script**

```python
#!/usr/bin/env python
"""C7 plots: quality vs N_per_cell, wall vs N_per_cell, per-pair latency."""
from __future__ import annotations
import argparse, glob, json, pathlib
import matplotlib.pyplot as plt
import numpy as np

SOLVER_ORDER = ["cajal-native", "pot-entropic-gpu", "pot-exact-gpu",
                "torchgw-precomputed"]
SOLVER_COLOR = {
    "cajal-native":         "#444",
    "pot-entropic-gpu":     "#1f77b4",
    "pot-exact-gpu":        "#9467bd",
    "torchgw-precomputed":  "#d62728",
}


def _load(stage: str, results_dir: pathlib.Path) -> list[dict]:
    out = []
    for p in sorted(results_dir.glob(f"core_07*{stage}*.json")):
        d = json.load(open(p))
        if d.get("status") != "ok":
            continue
        out.append(d)
    return out


def _aggregate(records: list[dict], metric_path: tuple[str, ...]):
    """Return dict[(solver, n_per_cell)] = (mean, std) over seeds."""
    bins: dict = {}
    for r in records:
        v = r
        for k in metric_path:
            v = v.get(k) if isinstance(v, dict) else None
        if v is None:
            continue
        bins.setdefault((r["solver"], r["n_per_cell"]), []).append(float(v))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in bins.items()}


def _plot_metric(records, metric_path, ylabel, title, out_path, log_y=False):
    agg = _aggregate(records, metric_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    n_values = sorted({n for (_, n) in agg.keys()})
    for solver in SOLVER_ORDER:
        xs, ys, errs = [], [], []
        for n in n_values:
            if (solver, n) not in agg: continue
            m, s = agg[(solver, n)]
            xs.append(n); ys.append(m); errs.append(s)
        if xs:
            ax.errorbar(xs, ys, yerr=errs, marker="o",
                        label=solver, color=SOLVER_COLOR[solver])
    ax.set_xscale("log")
    if log_y: ax.set_yscale("log")
    ax.set_xlabel("N_per_cell")
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["A", "B"])
    args = ap.parse_args()

    repo = pathlib.Path(__file__).resolve().parents[2]
    rdir = repo / "results" / "c7_cell_morphology"
    figdir = repo / "docs" / "figures"; figdir.mkdir(exist_ok=True, parents=True)
    stage = f"stage_{args.stage.lower()}"

    recs = _load(stage, rdir)
    if not recs:
        raise SystemExit(f"no records for {stage} in {rdir}")

    _plot_metric(recs, ("metrics", "ARI_ward"),
                 "ARI (Ward, vs ground truth)",
                 f"C7 {stage} — clustering quality vs sample size",
                 figdir / f"c7_{stage}_ari.png")
    _plot_metric(recs, ("metrics", "knn_acc_k5"),
                 "kNN accuracy (LOO, k=5)",
                 f"C7 {stage} — kNN type recovery vs sample size",
                 figdir / f"c7_{stage}_knn.png")
    _plot_metric(recs, ("efficiency", "wall_full_matrix_s"),
                 "Full-matrix wall (s)",
                 f"C7 {stage} — full N×N GW wall vs sample size",
                 figdir / f"c7_{stage}_wall.png", log_y=True)
    _plot_metric(recs, ("efficiency", "wall_per_pair_ms"),
                 "Per-pair wall (ms)",
                 f"C7 {stage} — per-pair GW latency vs sample size",
                 figdir / f"c7_{stage}_per_pair.png", log_y=True)

    # UMAP figure for the highest-N seed-0 of each solver
    from umap import UMAP
    n_top = max({r["n_per_cell"] for r in recs})
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, solver in zip(axes, SOLVER_ORDER):
        candidates = [r for r in recs if r["solver"] == solver
                      and r["n_per_cell"] == n_top and r["seed"] == 0]
        if not candidates:
            ax.set_title(f"{solver}\n(no data)"); ax.axis("off"); continue
        npy = pathlib.Path(rdir / (
            f"core_07_cell_morphology__{solver}__{stage}"
            f"__n{n_top}__seed0.npy"
        ))
        if not npy.exists():
            ax.set_title(f"{solver}\n(matrix not saved)"); ax.axis("off"); continue
        D = np.load(npy)
        emb = UMAP(metric="precomputed", random_state=0,
                   n_neighbors=min(15, D.shape[0] - 1)).fit_transform(D)
        labels = candidates[0]["classes"]
        # we don't have y in JSON per-cell, but classes index = label int
        # rebuild y by reading the manifest:
        manifest = repo / "tracks" / "core" / "07_cell_morphology" / f"{stage}_manifest.txt"
        cls_to_int = {c: i for i, c in enumerate(labels)}
        y = []
        for line in open(manifest):
            line = line.strip()
            if not line or line.startswith("neuron_name") or line.startswith("specimen_id"):
                continue
            cls = line.split("\t")[1]
            y.append(cls_to_int[cls])
        y = np.asarray(y[:emb.shape[0]])
        ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap="tab10", s=8, alpha=0.8)
        ax.set_title(f"{solver}, N={n_top}"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"C7 {stage} — UMAP of GW distance matrices")
    fig.tight_layout()
    fig.savefig(figdir / f"c7_{stage}_umap.png", dpi=150)
    print(f"wrote {figdir / f'c7_{stage}_umap.png'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate Stage A figures**

```bash
micromamba run -n c7_morph python scripts/experiments/make_c7_plots.py --stage A
ls docs/figures/c7_stage_a*.png
```

Expected: 5 PNGs (`ari`, `knn`, `wall`, `per_pair`, `umap`).

- [ ] **Step 3: Commit**

```bash
git add scripts/experiments/make_c7_plots.py docs/figures/c7_stage_a*.png
git commit -m "feat(C7): plotting + Stage A figures"
```

---

### Task 10: Stage B sweep + run

**Files:**
- Create: `scripts/run_c7_stage_b.sh`

- [ ] **Step 1: Write Stage B script**

Identical structure to Stage A; just change `--stage A` to `--stage B`,
output tag to `stage_b`, and log file. Copy `scripts/run_c7_stage_a.sh` to
`scripts/run_c7_stage_b.sh` and `sed -i 's/stage_a/stage_b/g; s/--stage A/--stage B/g' scripts/run_c7_stage_b.sh`. Verify the diff is exactly those token swaps.

```bash
cp scripts/run_c7_stage_a.sh scripts/run_c7_stage_b.sh
sed -i 's/stage_a/stage_b/g; s/--stage A/--stage B/g' scripts/run_c7_stage_b.sh
diff scripts/run_c7_stage_a.sh scripts/run_c7_stage_b.sh
chmod +x scripts/run_c7_stage_b.sh
```

- [ ] **Step 2: Run Stage B (only after Stage A success gate passed)**

```bash
bash scripts/run_c7_stage_b.sh
```

Expected: ~24 h; cajal-native × N=1000 is the long pole. Re-runnable —
existing JSONs are skipped (idempotent).

- [ ] **Step 3: Verify completion**

```bash
ls results/c7_cell_morphology/*stage_b*.json | wc -l
# expected 42 (same arithmetic as Stage A)
```

- [ ] **Step 4: Generate Stage B figures**

```bash
micromamba run -n c7_morph python scripts/experiments/make_c7_plots.py --stage B
ls docs/figures/c7_stage_b*.png
```

- [ ] **Step 5: Commit**

```bash
git add scripts/run_c7_stage_b.sh docs/figures/c7_stage_b*.png
git commit -m "feat(C7): Stage B sweep + figures (Allen CTDB ~1000 cells)"
```

---

### Task 11: Writeup + cross-track index update

**Files:**
- Create: `docs/experiments/2026-04-25-c7-cell-morphology.md`
- Modify: `docs/experiments/README.md`

- [ ] **Step 1: Draft the writeup**

Skeleton (fill from the actual numbers in `results/c7_cell_morphology/`):

```markdown
# C7 — Cell morphology vs CAJAL (2026-04-25)

**Setup.** Reused CAJAL's preprocessing (SWC → sample N points → intracell
geodesic distance matrix per cell), then swapped only the pairwise-GW step
across four solvers: cajal-native, pot-entropic-gpu, pot-exact-gpu,
torchgw-precomputed. Two stages: NeuroMorpho hand-picked 3-class subset
(stage A, ~300 cells) and Allen CTDB dendrite-type labels (stage B, ~1000
cells). Sample-size sweep N_per_cell ∈ {50, 200, 500, 1000}, 3 seeds.

**CAJAL backend**: <quote from the Task 1 probe output verbatim>.

## Stage A — sanity
![ARI](../figures/c7_stage_a_ari.png)
![wall](../figures/c7_stage_a_wall.png)

cajal-native at the highest N achieves ARI = <fill>; sanity gate passed.

## Stage B — benchmark
![ARI](../figures/c7_stage_b_ari.png)
![per-pair](../figures/c7_stage_b_per_pair.png)
![UMAP](../figures/c7_stage_b_umap.png)

| solver | ARI (N=1000) | kNN acc | wall (s) | per-pair (ms) |
|---|---|---|---|---|
| cajal-native       | … | … | … | … |
| pot-entropic-gpu   | … | … | … | … |
| pot-exact-gpu      | — (skip > N=200) | … | … | … |
| torchgw-precomputed| … | … | … | … |

## The sample-size threshold

[Identify the smallest N_per_cell at which torchgw-precomputed beats
cajal-native on full-matrix wall without losing more than 0.02 ARI. State
explicitly. If no such N exists, write "torchgw never wins on this regime"
and explain why — the per-pair GPU launch overhead figure is the smoking
gun.]

## Take-home

1. [Threshold finding or negative-result statement.]
2. [Whether the GPU-vs-CPU axis or the POT-vs-torchgw axis dominated the
   speed delta — read off pot-entropic-gpu vs cajal-native vs
   torchgw-precomputed.]
3. ["many tiny GW" deployment rule: <state>.]

## Caveats

- torchgw runs pairs serially on GPU; CAJAL parallelizes pairs on CPU. The
  per-pair latency plot is the apples-to-apples comparison; the
  full-matrix plot rewards CAJAL's parallelism. Both shown for honesty.
- pot-exact-gpu is skipped beyond N_per_cell = 200 (CG memory).
- Manifests pin specific cell IDs; results are reproducible only against
  those exact IDs.

## Reproducing

\`\`\`bash
micromamba activate c7_morph
bash tracks/core/07_cell_morphology/fetch.sh
bash scripts/run_c7_stage_a.sh
python scripts/experiments/make_c7_plots.py --stage A
bash scripts/run_c7_stage_b.sh
python scripts/experiments/make_c7_plots.py --stage B
\`\`\`
```

- [ ] **Step 2: Add C7 row to cross-track table in `docs/experiments/README.md`**

Insert into the synthesis table (after C6 column):

```markdown
| C7 (cell morpho, many tiny GW) |
| <ARI winner at N=1000>         |
| <wall winner>                  |
| 1000 cells × {50…1000} pts     |
| intracell geodesic (sparse)    |
| 5e-3                           |
| max(N, 3N/4) capped 1000       |
| <dominant failure>             |
| precomputed (only fair mode)   |
```

Also add a one-paragraph C7 section under the existing track sections,
linking to the writeup.

- [ ] **Step 3: Commit**

```bash
git add docs/experiments/2026-04-25-c7-cell-morphology.md docs/experiments/README.md
git commit -m "docs(C7): writeup + cross-track index entry"
```

- [ ] **Step 4: Push**

```bash
git push origin main
```

---

## Self-review notes

- **Spec §3 swap point**: enforced by `solvers.gw_pair` taking `D1, D2`
  pre-built only — no solver-side cost construction; `landmark`/`dijkstra`
  modes are absent from the dispatch table on purpose.
- **Spec §5 M_samples rule**: implemented in `_gw_torchgw_precomputed`
  (`M = max(min(n, 1000), 3·n//4)`).
- **Spec §6 metrics**: all six (ARI_ward, NMI_ward, ARI_spectral,
  NMI_spectral, knn_acc_k5, knn_macro_f1_k5) emitted by `eval.py`.
  Efficiency dict carries the four required fields.
- **Spec §7 skip rule**: `run.py` early-returns for pot-exact at N>200;
  sweep script skips the same.
- **Spec §10 caveat 1 (no batching)**: solver loop is intentionally serial
  for non-CAJAL solvers; CAJAL gets its native parallel path.
- **Spec §10 caveat 3 (manifest reproducibility)**: by-ID `fetch.sh`
  enforced; manifest committed.
- **Spec §11 success gate**: Stage A gate is an explicit checkpoint in
  Task 8 Step 3 — Stage B does not start until it passes.
- **Risk: CAJAL API drift** — flagged in Task 4 and Task 6 with a clear
  recovery path (re-import name only, no fallbacks).
- **Risk: NeuroMorpho IDs vanish** — `fetch.sh` warns per-ID and only
  errors if < 80% of the manifest survived. Manifest can be patched.
