# C1 Large-Scale Point-Cloud Scalability Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement task-by-task.

**Goal:** Build `tracks/core/01_point_cloud_scale` to prove torchgw's scalability claim — it runs at N where POT OOMs. Sweep N ∈ {10k, 20k, 50k, 100k} on ModelNet40 point clouds; POT is expected to OOM between N=20k and N=50k; torchgw-landmark expected to run all the way to N=100k.

**Cost matrix design (REVISED 2026-04-19):** kNN-hop geodesic (k=20, symmetrize, unweighted Dijkstra, max-normalize). Raw Euclidean cost on ModelNet point clouds was found to give torchgw P@1 ≈ 0 regardless of ε — same MC-SNR pathology as C5 cosine. kNN-hop cost is sparse + long-tailed (matches C2 SCOT recipe), giving torchgw partial recovery (P@1 ~ 0.25, P@5 ~ 0.6 at N=2000); POT-exact still wins on quality (P@1 ~ 0.97) but is bounded by O(N²) memory + O(N³) CG complexity.

**Story:** This is NOT "torchgw beats POT on quality at scale". It's "POT wins quality where it can run; torchgw-precomp/landmark are the only viable choice at N where POT OOMs, with mediocre but non-zero quality acceptable as the scalability tradeoff."

**Architecture:** Same structural pattern as C5. `fetch.sh` downloads ModelNet40. `run.py` loads a shape, does farthest-point sampling to N points, applies a known random rotation, runs GW, evaluates P@1 correspondence accuracy against the identity GT. Solver lineup adapts by N: 5 standard solvers at N≤20k; landmark/dijkstra/lowrank-only at N>20k (where `precomputed`'s N×N cost matrix is intractable).

**Tech Stack:** numpy, torch, POT, torchgw, `open3d` or `trimesh` for mesh I/O + FPS, matplotlib for plots. No new env — existing `dl2025` venv should cover.

**Literature/spec targets:** Original spec C1 calls for "400×500 → 4k×5k → 20k×25k → 50k×60k" scale sweep with all torchgw ablations. We simplify to a single-axis N sweep (source and target same N) because that's what reveals the POT OOM transition cleanly.

---

### Task 1: Track scaffolding + ModelNet40 fetch + mesh I/O module

**Files:**
- Create: `tracks/core/01_point_cloud_scale/fetch.sh`
- Create: `tracks/core/01_point_cloud_scale/README.md`
- Create: `tracks/core/01_point_cloud_scale/io.py`

- [x] **Step 1: `fetch.sh`**

Downloads ModelNet40 (`ModelNet40.zip` from Princeton or mirror). Cache under `data/core_01_point_cloud/`. Extract and verify at least one `.off` file per target class is present.

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$REPO_ROOT/data/core_01_point_cloud"
mkdir -p "$DATA_DIR"

ZIP_PATH="$DATA_DIR/ModelNet40.zip"
if [[ ! -s "$ZIP_PATH" ]]; then
    echo "[c1-fetch] downloading ModelNet40 (~440 MB)"
    curl -sSL -o "$ZIP_PATH" \
        "https://modelnet.cs.princeton.edu/ModelNet40.zip"
fi
if [[ ! -d "$DATA_DIR/ModelNet40" ]]; then
    echo "[c1-fetch] extracting"
    unzip -q "$ZIP_PATH" -d "$DATA_DIR"
fi
# Sanity: verify target classes present
for cls in airplane car lamp table sofa; do
    count=$(ls "$DATA_DIR/ModelNet40/$cls/train/"*.off 2>/dev/null | wc -l)
    if (( count < 10 )); then
        echo "[c1-fetch] ERROR: expected >=10 .off files for class $cls, found $count"
        exit 1
    fi
done
echo "[c1-fetch] done."
```

- [x] **Step 2: `README.md`** — one-paragraph stub, mention this is the scalability proof, link forward to `docs/experiments/2026-04-19-c1-point-cloud-scale.md`.

- [x] **Step 3: `io.py`** — two functions:

```python
def read_off(path) -> np.ndarray:
    """Parse a ModelNet .off mesh file, return (M, 3) vertex array (float32)."""
    # Parse: "OFF\n<nv> <nf> 0\n<vertex rows>\n<face rows>"
    # We only need vertices.

def fps_downsample(points: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Farthest-point sampling. O(n*N) — fine for N<=100k, n<=100k.
    If n >= len(points), return all points (no upsampling)."""
```

OFF format is simple text: line 1 "OFF", line 2 counts, then vertex coords, then face indices. We only need vertex coords. Robust to whitespace.

FPS: pick a random start, iteratively add the farthest point from the already-selected set. Store selected as indices.

- [x] **Step 4: Unit tests** at `tests/test_io.py`:
  - `read_off`: tiny fixture file → correct shape
  - `fps_downsample`: 1000 random points → 100 FPS samples → max pairwise distance should be ~90% of full point cloud's max distance (rough coverage check)

- [x] **Step 5: Smoke test**

```bash
bash tracks/core/01_point_cloud_scale/fetch.sh
python -c "
import sys; sys.path.insert(0, 'tracks/core/01_point_cloud_scale')
import importlib.util
spec = importlib.util.spec_from_file_location('io', 'tracks/core/01_point_cloud_scale/io.py')
io = importlib.util.module_from_spec(spec); spec.loader.exec_module(io)
P = io.read_off('data/core_01_point_cloud/ModelNet40/airplane/train/airplane_0001.off')
print('shape:', P.shape, 'dtype:', P.dtype)
P_sub = io.fps_downsample(P, 1000, seed=0)
print('FPS shape:', P_sub.shape)
"
```

Expected: (big, 3) for raw, (1000, 3) for FPS output.

- [x] **Step 6: Commit**

```bash
git add tracks/core/01_point_cloud_scale/
git commit -m "feat(C1): scaffolding + ModelNet40 fetch + OFF/FPS I/O"
```

---

### Task 2: Rotation-based pair generation + correspondence metric

**Files:**
- Create: `tracks/core/01_point_cloud_scale/pair.py`
- Create: `tracks/core/01_point_cloud_scale/eval.py`

- [x] **Step 1: `pair.py::make_pair(P, n, seed)`**

```python
def make_pair(P: np.ndarray, n: int, seed: int):
    """Create (source, target, R_gt) where target = R_gt @ source.

    Pipeline: FPS-downsample P to n points (deterministic seed).
    Generate a random rotation R via uniform SO(3) sampling.
    target = R @ source. Identity index correspondence: source[i] <-> target[i].
    Returns: (source: (n,3) float32, target: (n,3) float32, R_gt: (3,3) float32)
    """
```

Uniform SO(3) via Shoemake's algorithm or scipy.spatial.transform.Rotation.random.

- [x] **Step 2: `eval.py::correspondence_accuracy(T)`**

```python
def correspondence_accuracy(T: np.ndarray) -> float:
    """Fraction of rows where argmax equals the row index.
    GT correspondence is identity (source[i] <-> target[i])."""
    return float((np.argmax(T, axis=1) == np.arange(T.shape[0])).mean())
```

Also add `correspondence_recall_at_k(T, k=5)` for P@5.

- [x] **Step 3: `eval.py::chamfer_distance(projected, target)`**

Projected = barycentric projection of source through T: `(T / T.sum(1).clip(1e-30)) @ target`. Chamfer = `mean(min_j ||projected[i] - target[j]||²) + symmetric term`.

- [x] **Step 4: Unit tests** at `tests/test_pair.py`, `tests/test_eval.py`:
  - `make_pair`: output shapes correct, `||target - R @ source|| < 1e-5`
  - `correspondence_accuracy`: identity plan → 1.0; shuffled plan → 0.0
  - `chamfer_distance`: projected == target → 0; random projected → positive

Run: `pytest tracks/core/01_point_cloud_scale/tests/ -v`. Expect all pass.

- [x] **Step 5: Commit**

```bash
git add tracks/core/01_point_cloud_scale/pair.py \
        tracks/core/01_point_cloud_scale/eval.py \
        tracks/core/01_point_cloud_scale/tests/
git commit -m "feat(C1): pair generation (rotation) + correspondence P@k + Chamfer"
```

---

### Task 3: `run.py` main pipeline with conditional solver dispatch by N

**Files:**
- Create: `tracks/core/01_point_cloud_scale/run.py`

Mirror C5/C2's structure but with **N-conditional solver lineup**:

- [x] **Step 1: CLI**

```
--shape-class        airplane | car | lamp | table | sofa  (default airplane)
--instance-idx       int (default 0)  — picks airplane_{idx:04d}.off from train/
--n-points           N (default 5000)
--solver             one of: pot-entropic-gpu, pot-exact-gpu, torchgw-landmark,
                     torchgw-dijkstra, torchgw-precomputed, torchgw-lowrank-landmark,
                     torchgw-lowrank-dijkstra
--seed               int (default 0) — rotation + FPS seed
--epsilon            ε (default 5e-3, C1-specific — geometric Euclidean cost wants
                     moderate regularization; to be confirmed by sweep)
--M-samples          int | None (torchgw only; default None → use the cell script's
                     M=max(1000, 3N/4) rule, capped at N)
--lowrank-rank       int (default 20; only for lowrank variants)
--out                output dir
```

- [x] **Step 2: Cost matrix construction**

For each input point cloud `P` of shape (n, 3): `C = pairwise Euclidean distance`. **Do NOT normalize** beyond default — for rotation-invariant task, both C1 and C2 are already on the same scale (same point cloud, rotated). Keep float32.

For `torchgw-precomputed` and both POT solvers: pass `(C_src, C_tgt)` to solver.

For `torchgw-landmark`, `torchgw-dijkstra`, and the lowrank variants: pass raw point-cloud coordinates; torchgw builds internal cost.

- [x] **Step 3: N-conditional solver skip**

If `N > 20000` AND solver is POT or `torchgw-precomputed`: write a JSON record with `status="skipped_oom_risk"` and `error="N>20000, OOM risk — not attempted"` and exit. This makes the cache-skip logic in the bench script clean and makes the JSON summary honest about what was tested.

If `N > 50000` AND solver is `torchgw-landmark` / `torchgw-dijkstra`: also allow, but log a warning (may OOM on kNN graph at 100k).

- [x] **Step 4: Two torchgw modes for lowrank**

```python
def run_torchgw_lowrank_landmark(P_src, P_tgt, ...): 
    from torchgw import sampled_lowrank_gw
    T, log = sampled_lowrank_gw(
        X_source=P_src, X_target=P_tgt,
        distance_mode="landmark",
        fgw_alpha=0.0, epsilon=eps, rank=rank, ...)
    ...

def run_torchgw_lowrank_dijkstra(P_src, P_tgt, ...):
    # similar but distance_mode="dijkstra"
```

Sanity: log `rank` and `lr_max_iter` in `hyperparams`.

- [x] **Step 5: Metrics recorded**

```python
rec["metrics"]["task"] = {
    "correspondence_accuracy": float,  # P@1
    "correspondence_recall_at_5": float,
    "chamfer_distance": float,
    "n_points": int,
}
rec["metrics"]["efficiency"] = {
    "wall_preprocess_s": float,
    "wall_solve_s": float,
    "gpu_peak_gb": float,
    "ram_peak_gb": float,
}
```

- [x] **Step 6: Smoke test**

```bash
python tracks/core/01_point_cloud_scale/run.py \
    --shape-class airplane --instance-idx 0 --n-points 1000 \
    --solver pot-entropic-gpu --epsilon 5e-3 --seed 0 \
    --out results/c1_smoke
```

Expected at N=1000: P@1 ≈ 1.0 (rotation is a trivial task for any working GW solver at small N), wall_solve < 10s.

Then one torchgw variant at N=5000:

```bash
python tracks/core/01_point_cloud_scale/run.py \
    --shape-class airplane --instance-idx 0 --n-points 5000 \
    --solver torchgw-landmark --seed 0 \
    --out results/c1_smoke
```

Expected: P@1 ≥ 0.9, wall_solve < 30s.

If smoke test P@1 << 1.0, pause and debug before proceeding.

- [x] **Step 7: Commit**

```bash
git add tracks/core/01_point_cloud_scale/run.py
git commit -m "feat(C1): main run.py with N-conditional solver dispatch + lowrank variants"
```

---

### Task 4: Bench sweep script (the scalability proof)

**Files:**
- Create: `scripts/run_c1_bench.sh`

- [x] **Step 1: Sweep iteration**

```
shape_class × instance_idx × n_points × seed × solver
```
- classes: 1 (airplane, fix for v1 simplicity)
- instance_idx: 3 (0, 1, 2)
- n_points: {1000, 5000, 10000, 20000, 50000, 100000}
- seeds: 3 (0, 1, 2)
- solvers: N-conditional

Total theoretical cells = 1 × 3 × 6 × 3 × 7 = 378, but N-conditional pruning cuts the POT/precomputed at N>20k, so actual ~250 cells.

- [x] **Step 2: N-conditional solver list**

In the bash loop:

```bash
if (( n <= 20000 )); then
    SOLVERS=(pot-entropic-gpu pot-exact-gpu \
             torchgw-landmark torchgw-dijkstra torchgw-precomputed)
else
    SOLVERS=(torchgw-landmark torchgw-dijkstra \
             torchgw-lowrank-landmark torchgw-lowrank-dijkstra)
fi
```

- [x] **Step 3: M_samples rule**

Same as C2/C5: `M = max(1000, 3N/4)` capped at N, for all torchgw variants.

- [x] **Step 4: Cache-skip**

Same pattern as C5: cache JSON if `status != fail` and torchgw cells have `M_samples ≥ 1000`. Skip `"skipped_oom_risk"` cells without re-running.

- [x] **Step 5: Run**

```bash
nohup bash scripts/run_c1_bench.sh > logs/c1_bench.log 2>&1 &
```

Estimated wall time: small N fast (~1 min per cell × 60 small cells = 1 h); large N potentially 10+ min per cell × 30 large cells = 5+ h. **Total ~4-6 hours.**

- [x] **Step 6: Commit script (before running — so script exists in git even if bench takes long)**

```bash
git add scripts/run_c1_bench.sh
git commit -m "feat(C1): bench sweep script (N ∈ {1k..100k} × 3 seeds × N-conditional solvers)"
```

---

### Task 5: Plotting + headline figures

**Files:**
- Create: `scripts/experiments/make_c1_plots.py`

- [x] **Step 1: Aggregate JSONs**

Read `results/c1_point_cloud_scale/*.json`, filter by `status=="ok"`, group by (solver, n_points), compute mean ± std across seeds+instances.

- [x] **Step 2: Figure 1 — wall-time vs N (log-log)**

- x-axis: N (log scale, 1000 to 100000)
- y-axis: wall_solve_s (log scale)
- one line per solver; truncate line where solver becomes OOM (marker "×" at first skipped cell)
- annotation: "POT OOM threshold" vertical band

Save: `docs/figures/c1_scalability_wall.png`

- [x] **Step 3: Figure 2 — P@1 vs N**

- x-axis: N (log scale)
- y-axis: P@1 correspondence accuracy
- horizontal line at 1.0 = perfect (rotation-invariant GW should hit this at small N)
- Show degradation patterns at large N by solver

Save: `docs/figures/c1_scalability_quality.png`

- [x] **Step 4: Figure 3 — GPU memory vs N (log-log)**

- x-axis: N (log)
- y-axis: gpu_peak_gb (log)
- annotations: "80GB H100 ceiling" horizontal line; "POT N² curve" theoretical fit
- one line per solver

Save: `docs/figures/c1_scalability_memory.png`

- [x] **Step 5: Commit**

```bash
git add scripts/experiments/make_c1_plots.py \
        results/c1_point_cloud_scale docs/figures/c1_*.png
git commit -m "feat(C1): bench execution + headline scalability figures"
```

---

### Task 6: Writeup + experiments index update

**Files:**
- Create: `docs/experiments/2026-04-19-c1-point-cloud-scale.md`
- Modify: `docs/experiments/README.md`

- [x] **Step 1: Writeup** (same structure as C5):

1. **Positioning**: C1 tests torchgw at N where POT cannot run. All other tracks stop at N≤10k in POT's comfort zone; this is the first track to enter the post-POT regime.

2. **Task**: same-shape rotation → identity correspondence GT, ModelNet40 airplane class.

3. **Pipeline**: FPS downsample → apply random SO(3) → cost matrices → solve GW → evaluate P@1.

4. **Headline table**: solver × N. Include POT OOM transition, torchgw N=100k entries.

5. **Observations**:
   - Where does POT OOM?
   - Does torchgw quality degrade with N? How gracefully?
   - `sampled_lowrank_gw` vs standard at N=50k: which wins?
   - `multiscale` benefit at large N?

6. **Scalability take-home**: validated or not. Quote the max N achieved.

- [x] **Step 2: Update `docs/experiments/README.md`**

Add C1 section (promote to first/top track since it's "foundation"). Update cross-track synthesis to include the scalability data point.

- [x] **Step 3: Commit**

```bash
git add docs/experiments/2026-04-19-c1-point-cloud-scale.md docs/experiments/README.md
git commit -m "docs(C1): large-scale scalability writeup + index update"
```

---

## Status

**Complete (2026-04-25).** All six tasks landed; writeup at
`docs/experiments/2026-04-19-c1-point-cloud-scale.md`; index entry in
`docs/experiments/README.md`. Open follow-ups left to a future track,
not blocking C1 closure:
- pot-exact CG instability at N=20k (Spearman 0.88 → 0.76) — needs a
  max-iter / tolerance sweep before being trusted at scale.
- `torchgw-lowrank-dijkstra` only has seed=0 cells at N=10k/20k; either
  backfill seeds 1/2 or drop the variant from headline plots.
- ModelNet40 path was abandoned for synthetic spiral after raw
  Euclidean cost killed torchgw P@1 (kept here for archival context;
  the spiral pivot is the actual delivered experiment).

## Self-review notes

- **Spec coverage:** scale sweep → Task 4; lowrank test → Task 3/4; ablations (multiscale, mixed_precision) deliberately **deferred** to a follow-up mini-sweep, not in main plan, to keep scope manageable.
- **Not in scope:** Chamfer-only metric on different instances (deferred to an optional follow-up); cross-class matching; non-rigid deformation (FAUST-style). All deferred to keep v1 focused on pure scalability.
- **Risk:** `sampled_lowrank_gw` at N=100k may still OOM if landmark graph too dense. Task 3 step 6 has a smoke test at N=5000 to catch API issues early; if N=50k lowrank fails, Task 4 will log it as skipped and we document the real ceiling.
- **Risk:** ModelNet `.off` files are text format — parsing is fast but not trivial for some edge cases (comments, multi-line headers). `io.py` should handle these; Task 1 unit tests need a realistic fixture.
- **Risk:** torchgw's `torchgw-precomputed` and POT at N=20k may both barely fit (N² float32 cost = 1.6GB × 2 = 3.2GB, plus plan + intermediates ~5GB). If the 20k cell is itself borderline OOM, the transition point shifts to N=15k. The bench will expose this honestly.
