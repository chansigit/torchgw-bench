# C1 Scale Sweep Implementation Plan (Phase 1b)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the C1 Foundation track to run at four scales (400×500, 4k×5k, 10k×12k, 20k×25k), add a POT memory guard that skips large scales automatically, update the reporter to show scale as a table column, and commit the multi-scale benchmark table to `docs/tier_core.md`.

**Architecture:** All changes are confined to two files: `tracks/core/01_foundation/run.py` (solver wrappers + CLI) and `scripts/make_report.py` (reporter renderer). A new `scripts/run_scale_sweep.sh` drives the multi-scale runs. The zero-shared-code invariant is preserved.

**Tech Stack:** Python 3.10, torchgw (landmark mode, k=5), POT 0.9.6, numpy, bash

**Working directory:** `/scratch/users/chensj16/projects/torchgw-bench/`

**Active Python env:** `/scratch/users/chensj16/venvs/dl2025/.venv`

Run tests with:
```bash
source /scratch/users/chensj16/venvs/dl2025/.venv/bin/activate
cd /scratch/users/chensj16/projects/torchgw-bench
python -m pytest -v
```

---

## Scale matrix

| N × K | torchgw-landmark | pot-entropic | Notes |
|---|---|---|---|
| 400 × 500 | ✓ already | ✓ already | Phase 1 baseline |
| 4,000 × 5,000 | ✓ | ✓ (C1²=128 MB, feasible) | Both solvers |
| 10,000 × 12,000 | ✓ | ✗ skipped (C1²=800 MB, OOM risk) | torchgw only |
| 20,000 × 25,000 | ✓ | ✗ skipped (C1²=3.2 GB) | torchgw only |

**torchgw hyperparams:**
- N ≤ 20k: `M=80, k=5, n_landmarks=50, epsilon=5e-3, max_iter=300` (same as Phase 1)
- N > 20k (reserved for Phase 1c): `M=100`

**POT guard threshold:** skip POT when `max(N_source, N_target) > 5_000`. Writes a record with `status="skip"` and `error="skipped: POT O(N²) memory guard (N=...)"` rather than crashing.

---

## File structure

| File | Change |
|---|---|
| `tracks/core/01_foundation/run.py` | Add `_pot_too_large()` guard; update `main()` to emit skip record |
| `tracks/core/01_foundation/tests/test_run.py` | Tests for the skip record path |
| `scripts/make_report.py` | Add `N×K` column to `render_track_section` table |
| `scripts/tests/test_make_report.py` | Tests for multi-scale table rendering |
| `scripts/run_scale_sweep.sh` | New: drive multi-scale runs |

---

## Task 1: POT memory guard — emit `status=skip` for large scales

**Files:**
- Modify: `tracks/core/01_foundation/run.py`
- Modify: `tracks/core/01_foundation/tests/test_run.py`

The `main()` function in `run.py` calls `run_pot_entropic` unconditionally. At 10k×12k, POT needs an 800 MB cost matrix; at 20k×25k it's 3.2 GB — both crash with OOM. We want to skip POT gracefully and write an informative JSON record instead of crashing.

- [ ] **Step 1: Append failing tests**

Append to `tracks/core/01_foundation/tests/test_run.py`:

```python
# ---- POT memory guard ---------------------------------------------------

def test_pot_too_large_returns_true_above_threshold():
    assert run.pot_too_large(n_source=6000, n_target=7000, threshold=5000) is True


def test_pot_too_large_returns_false_below_threshold():
    assert run.pot_too_large(n_source=400, n_target=500, threshold=5000) is False


def test_pot_too_large_exact_threshold_is_ok():
    # max(5000, 4000) == 5000, not strictly greater → should NOT skip
    assert run.pot_too_large(n_source=5000, n_target=4000, threshold=5000) is False
```

- [ ] **Step 2: Run and confirm 3 failures**

```bash
cd /scratch/users/chensj16/projects/torchgw-bench
source /scratch/users/chensj16/venvs/dl2025/.venv/bin/activate
python -m pytest tracks/core/01_foundation/tests/test_run.py -k "pot_too_large" -v
```

Expected: 3 failures with `AttributeError: module 'run' has no attribute 'pot_too_large'`.

- [ ] **Step 3: Implement `pot_too_large`**

Add to `tracks/core/01_foundation/run.py`, above `def run_pot_entropic`:

```python
def pot_too_large(n_source: int, n_target: int, threshold: int = 5_000) -> bool:
    """Return True if POT's O(N²) cost matrices would exceed the memory guard.

    POT's entropic_gromov_wasserstein builds two dense float64 (N×N) and
    (K×K) cost matrices. At threshold=5_000, the larger matrix is
    5000×5000×8 bytes = 200 MB, which is borderline acceptable. Above that
    the risk of OOM or multi-minute wall time grows rapidly.
    """
    return max(n_source, n_target) > threshold
```

- [ ] **Step 4: Re-run and confirm 3 pass**

```bash
python -m pytest tracks/core/01_foundation/tests/test_run.py -k "pot_too_large" -v
```

Expected: 3 pass.

- [ ] **Step 5: Wire the guard into `main()`**

In `tracks/core/01_foundation/run.py`, find the `main()` body where `run_pot_entropic` is called and replace the relevant block:

Current (lines ~305–309 in `main()`):
```python
        if args.solver == "torchgw-landmark":
            result = run_torchgw_landmark(X, Y, seed=args.seed)
        elif args.solver == "pot-entropic":
            result = run_pot_entropic(X, Y, seed=args.seed)
        else:
            raise ValueError(f"unknown solver: {args.solver}")
```

Replace with:
```python
        if args.solver == "torchgw-landmark":
            result = run_torchgw_landmark(X, Y, seed=args.seed)
        elif args.solver == "pot-entropic":
            if pot_too_large(args.n_source, args.n_target):
                rec["status"] = "skip"
                rec["error"] = (
                    f"skipped: POT O(N²) memory guard "
                    f"(max(N,K)={max(args.n_source, args.n_target)} > 5000)"
                )
                out_path.write_text(json.dumps(rec, indent=2))
                print(f"[C1] skipped (POT memory guard) → {out_path}")
                return
            result = run_pot_entropic(X, Y, seed=args.seed)
        else:
            raise ValueError(f"unknown solver: {args.solver}")
```

Note: this `return` exits `main()` early — it is inside the `try` block but before the `else` clause, which is fine because we wrote the record manually before returning.

Actually, to avoid the early-return cutting across the try/else structure, restructure the guard to be outside the try:

Replace the entire try/except/else block in `main()` with:

```python
    # POT memory guard — must check before entering the solver try block
    if args.solver == "pot-entropic" and pot_too_large(args.n_source, args.n_target):
        rec["status"] = "skip"
        rec["error"] = (
            f"skipped: POT O(N²) memory guard "
            f"(max(N,K)={max(args.n_source, args.n_target)} > 5000)"
        )
        out_path.write_text(json.dumps(rec, indent=2))
        print(f"[C1] skipped (POT memory guard) → {out_path}")
        return

    try:
        X, src_angles = sample_spiral(args.n_source, seed=args.seed)
        Y, tgt_angles = sample_swiss_roll(args.n_target, seed=args.seed + 1)

        if args.solver == "torchgw-landmark":
            result = run_torchgw_landmark(X, Y, seed=args.seed)
        elif args.solver == "pot-entropic":
            result = run_pot_entropic(X, Y, seed=args.seed)
        else:
            raise ValueError(f"unknown solver: {args.solver}")

        # Pull hyperparams + version into the record
        rec["hyperparams"] = result["hyperparams"]
        rec["solver_version"] = result["solver_version"]

        # Fill metrics sub-dicts
        rec["metrics"]["correctness"] = {
            "gw_cost": result["gw_cost"],
            "marginal_error": result["marginal_error"],
        }
        rec["metrics"]["task"] = {
            "spearman_arclen": arclen_spearman(result["T"], src_angles, tgt_angles),
        }
        rec["metrics"]["efficiency"] = {
            "wall_s": result["wall_s"],
            "gpu_peak_gb": result["gpu_peak_gb"],
            "iterations": result["iterations"],
        }
    except Exception as e:
        rec["status"] = "fail"
        rec["error"] = f"{type(e).__name__}: {e}"
        out_path.write_text(json.dumps(rec, indent=2))
        raise
    else:
        out_path.write_text(json.dumps(rec, indent=2))
        print(f"[C1] wrote {out_path}")
```

- [ ] **Step 6: Run the full test suite — all 20 must still pass**

```bash
python -m pytest tracks/core/01_foundation/tests/test_run.py -v
```

Expected: 23 pass (20 old + 3 new).

- [ ] **Step 7: Quick smoke test of the skip path**

```bash
source /scratch/users/chensj16/venvs/dl2025/.venv/bin/activate
cd /scratch/users/chensj16/projects/torchgw-bench
python tracks/core/01_foundation/run.py \
    --solver pot-entropic --seed 0 --out /tmp/tgwbench_smoke/ \
    --n-source 10000 --n-target 12000
```

Expected stdout: `[C1] skipped (POT memory guard) → /tmp/tgwbench_smoke/...json`
Inspect the JSON:
```bash
python -c "import json; r=json.load(open('/tmp/tgwbench_smoke/core_01_foundation__pot-entropic__seed0.json')); print(r['status'], r['error'])"
```
Expected: `skip  skipped: POT O(N²) memory guard (max(N,K)=12000 > 5000)`

- [ ] **Step 8: Commit**

```bash
cd /scratch/users/chensj16/projects/torchgw-bench
git add tracks/core/01_foundation/run.py tracks/core/01_foundation/tests/test_run.py
git commit -m "feat(C1): add POT memory guard for large-scale runs

pot_too_large(n_source, n_target, threshold=5000) returns True when
max(N,K) > threshold. main() emits status='skip' with an informative
error string before entering the solver try-block, so the output JSON
is always written regardless of outcome."
```

---

## Task 2: Update reporter — add `N×K` scale column to the table

**Files:**
- Modify: `scripts/make_report.py`
- Modify: `scripts/tests/test_make_report.py`

Currently `render_track_section` shows `**Scale:** N=400, K=500` as a header and has no scale column in the table. When results from multiple scales live in the same track, there is no way to tell which row belongs to which scale.

- [ ] **Step 1: Append failing test**

Append to `scripts/tests/test_make_report.py`:

```python
# ---- multi-scale table --------------------------------------------------

def test_render_track_section_shows_scale_column_when_multiple_scales():
    records = [
        {
            "track": "core/01_foundation",
            "solver": "torchgw-landmark",
            "status": "ok",
            "dataset": {"name": "spiral_400_swissroll_500", "n_source": 400, "n_target": 500},
            "metrics": {"correctness": {"gw_cost": 0.001}, "task": {"spearman_arclen": 0.999},
                        "efficiency": {"wall_s": 7.1, "gpu_peak_gb": 0.04, "iterations": 300}},
        },
        {
            "track": "core/01_foundation",
            "solver": "torchgw-landmark",
            "status": "ok",
            "dataset": {"name": "spiral_4000_swissroll_5000", "n_source": 4000, "n_target": 5000},
            "metrics": {"correctness": {"gw_cost": 0.002}, "task": {"spearman_arclen": 0.998},
                        "efficiency": {"wall_s": 12.3, "gpu_peak_gb": 0.5, "iterations": 280}},
        },
    ]
    md = make_report.render_track_section("core/01_foundation", records)
    # Scale column header must appear
    assert "N×K" in md or "Scale" in md
    # Both scale values must appear as data
    assert "400×500" in md or "400" in md
    assert "4000×5000" in md or "4000" in md


def test_render_track_section_skip_record_shows_skip_status():
    records = [
        {
            "track": "core/01_foundation",
            "solver": "pot-entropic",
            "status": "skip",
            "error": "skipped: POT O(N²) memory guard (max(N,K)=12000 > 5000)",
            "dataset": {"name": "spiral_10000_swissroll_12000", "n_source": 10000, "n_target": 12000},
            "metrics": {},
        },
    ]
    md = make_report.render_track_section("core/01_foundation", records)
    assert "pot-entropic" in md
    # skip status marker must appear
    assert "skip" in md.lower() or "⊘" in md
```

- [ ] **Step 2: Run and confirm 2 failures**

```bash
cd /scratch/users/chensj16/projects/torchgw-bench
source /scratch/users/chensj16/venvs/dl2025/.venv/bin/activate
python -m pytest scripts/tests/test_make_report.py -k "scale" -v
```

Expected: 2 failures.

- [ ] **Step 3: Update `render_track_section`**

In `scripts/make_report.py`, replace the table header and row-building block inside `render_track_section`:

Find:
```python
    lines.append("| Solver | Status | GW cost | Spearman | Wall (s) | GPU peak (GB) | Iterations |")
    lines.append("|---|:---:|---:|---:|---:|---:|---:|")
    for r in records:
        solver = r.get("solver", "?")
        status = r.get("status", "?")
        metrics = r.get("metrics", {}) or {}
        correctness = metrics.get("correctness", {}) or {}
        task = metrics.get("task", {}) or {}
        efficiency = metrics.get("efficiency", {}) or {}

        status_cell = {"ok": "✓", "fail": "✗ FAIL", "skip": "⊘ skip"}.get(status, status)

        row = (
            f"| `{solver}` | {status_cell} | "
            f"{_fmt(correctness.get('gw_cost'), '.4f')} | "
            f"{_fmt(task.get('spearman_arclen'), '.4f')} | "
            f"{_fmt(efficiency.get('wall_s'), '.2f')} | "
            f"{_fmt(efficiency.get('gpu_peak_gb'), '.2f')} | "
            f"{_fmt(efficiency.get('iterations'))} |"
        )
        lines.append(row)

        if status == "fail" and r.get("error"):
            lines.append(f"|     | error: `{r['error']}` |||||||")
```

Replace with:
```python
    lines.append("| N×K | Solver | Status | GW cost | Spearman | Wall (s) | GPU peak (GB) | Iterations |")
    lines.append("|---|---|:---:|---:|---:|---:|---:|---:|")
    for r in records:
        solver = r.get("solver", "?")
        status = r.get("status", "?")
        ds = r.get("dataset", {}) or {}
        n_src = ds.get("n_source")
        n_tgt = ds.get("n_target")
        scale_cell = f"{n_src}×{n_tgt}" if (n_src and n_tgt) else "—"
        metrics = r.get("metrics", {}) or {}
        correctness = metrics.get("correctness", {}) or {}
        task = metrics.get("task", {}) or {}
        efficiency = metrics.get("efficiency", {}) or {}

        status_cell = {"ok": "✓", "fail": "✗ FAIL", "skip": "⊘ skip"}.get(status, status)

        row = (
            f"| {scale_cell} | `{solver}` | {status_cell} | "
            f"{_fmt(correctness.get('gw_cost'), '.4f')} | "
            f"{_fmt(task.get('spearman_arclen'), '.4f')} | "
            f"{_fmt(efficiency.get('wall_s'), '.2f')} | "
            f"{_fmt(efficiency.get('gpu_peak_gb'), '.2f')} | "
            f"{_fmt(efficiency.get('iterations'))} |"
        )
        lines.append(row)

        if status in ("fail", "skip") and r.get("error"):
            lines.append(f"|     |     | note: `{r['error']}` ||||||")
```

- [ ] **Step 4: Run all reporter tests — all 12 must pass**

```bash
python -m pytest scripts/tests/test_make_report.py -v
```

Expected: 12 pass (10 old + 2 new).

Note: the existing test `test_render_track_section_header_and_table_columns` checks for `"| Solver"` — this now becomes `"| N×K | Solver"`. Update that assertion in the existing test:

In `scripts/tests/test_make_report.py`, find:
```python
    assert "| Solver" in md
```
Replace with:
```python
    assert "Solver" in md
```

After that change all 12 tests should pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/make_report.py scripts/tests/test_make_report.py
git commit -m "feat(reporter): add N×K scale column and ⊘ skip status to table

Multi-scale runs produce records with different dataset.n_source/n_target.
The new N×K column lets readers compare across scales in one table.
Skip records (from POT memory guard) now surface as '⊘ skip' with the
guard message in a follow-up note row."
```

---

## Task 3: `scripts/run_scale_sweep.sh` — multi-scale runner

**Files:**
- Create: `scripts/run_scale_sweep.sh`

This script calls `run.py` at each scale × solver combination and prints a summary at the end.

- [ ] **Step 1: Write the script**

Write `/scratch/users/chensj16/projects/torchgw-bench/scripts/run_scale_sweep.sh`:

```bash
#!/usr/bin/env bash
# Run C1 Foundation track at all Phase 1b scales and regenerate tier_core.md.
#
# Usage:
#     bash scripts/run_scale_sweep.sh [--out RESULTS_DIR] [--seed N] [--quick]
#
# Options:
#     --out DIR    directory to write JSON records (default: results/)
#     --seed N     random seed (default: 0)
#     --quick      only run 400x500 and 4000x5000 (skip 10k and 20k)
#
# Uses the active Python environment. Does NOT require mamba/conda.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RUN_PY="$REPO_ROOT/tracks/core/01_foundation/run.py"

OUT="$REPO_ROOT/results"
SEED=0
QUICK=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)   OUT="$2"; shift 2 ;;
        --seed)  SEED="$2"; shift 2 ;;
        --quick) QUICK=1; shift ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "$OUT"

# Scale definitions: "N_source K_target"
SCALES=("400 500" "4000 5000" "10000 12000" "20000 25000")
if [[ "$QUICK" -eq 1 ]]; then
    SCALES=("400 500" "4000 5000")
fi

SOLVERS=("torchgw-landmark" "pot-entropic")

echo "[sweep] Starting C1 scale sweep  seed=$SEED  out=$OUT"
echo "[sweep] Scales: ${SCALES[*]}"
echo ""

for scale in "${SCALES[@]}"; do
    N=$(echo "$scale" | awk '{print $1}')
    K=$(echo "$scale" | awk '{print $2}')
    for solver in "${SOLVERS[@]}"; do
        echo "[sweep] N=$N K=$K  solver=$solver"
        python "$RUN_PY" \
            --solver "$solver" \
            --seed "$SEED" \
            --out "$OUT" \
            --n-source "$N" \
            --n-target "$K" \
            2>&1 | grep -E "^\[C1\]" || true
    done
done

echo ""
echo "[sweep] Done. Regenerate report with:"
echo "    python $SCRIPT_DIR/make_report.py --format docs --results $OUT --out $REPO_ROOT/docs/ --tier core"
```

- [ ] **Step 2: Make it executable and syntax-check**

```bash
chmod +x /scratch/users/chensj16/projects/torchgw-bench/scripts/run_scale_sweep.sh
bash -n /scratch/users/chensj16/projects/torchgw-bench/scripts/run_scale_sweep.sh
```

Expected: no output (syntax OK).

- [ ] **Step 3: Run the quick sweep as a smoke test (400×500 + 4k×5k only)**

```bash
source /scratch/users/chensj16/venvs/dl2025/.venv/bin/activate
cd /scratch/users/chensj16/projects/torchgw-bench
bash scripts/run_scale_sweep.sh --quick --seed 0
```

Expected output (approximately):
```
[sweep] Starting C1 scale sweep  seed=0  out=.../results
[sweep] Scales: 400 500 4000 5000

[sweep] N=400 K=500  solver=torchgw-landmark
[C1] wrote results/core_01_foundation__torchgw-landmark__seed0.json
[sweep] N=400 K=500  solver=pot-entropic
[C1] wrote results/core_01_foundation__pot-entropic__seed0.json
[sweep] N=4000 K=5000  solver=torchgw-landmark
[C1] wrote results/core_01_foundation__torchgw-landmark__seed0.json
[sweep] N=4000 K=5000  solver=pot-entropic
[C1] wrote results/core_01_foundation__pot-entropic__seed0.json
```

Wait — the file naming uses `--n-source` and `--n-target` in the name? Check `main()` in run.py:

```python
out_path = args.out / f"core_01_foundation__{args.solver}__seed{args.seed}.json"
```

**Problem:** the current filename does NOT include scale. At 4k×5k, it would overwrite the 400×500 result. Must fix this in `main()`.

- [ ] **Step 4: Fix the output filename to include scale**

In `tracks/core/01_foundation/run.py`, in `main()`, find:
```python
    out_path = args.out / f"core_01_foundation__{args.solver}__seed{args.seed}.json"
```
Replace with:
```python
    out_path = args.out / (
        f"core_01_foundation__{args.solver}"
        f"__n{args.n_source}k{args.n_target}"
        f"__seed{args.seed}.json"
    )
```

This produces filenames like `core_01_foundation__torchgw-landmark__n400k500__seed0.json`.

- [ ] **Step 5: Run the full test suite to confirm no regressions**

```bash
source /scratch/users/chensj16/venvs/dl2025/.venv/bin/activate
cd /scratch/users/chensj16/projects/torchgw-bench
python -m pytest -v
```

Expected: 23 pass (tracks) + 12 pass (reporter) = 35 pass total.

- [ ] **Step 6: Re-run the quick smoke test with the fixed filename**

```bash
bash scripts/run_scale_sweep.sh --quick --seed 0
ls results/
```

Expected files:
```
core_01_foundation__pot-entropic__n400k500__seed0.json
core_01_foundation__pot-entropic__n4000k5000__seed0.json
core_01_foundation__torchgw-landmark__n400k500__seed0.json
core_01_foundation__torchgw-landmark__n4000k5000__seed0.json
```

- [ ] **Step 7: Commit**

```bash
git add scripts/run_scale_sweep.sh tracks/core/01_foundation/run.py
git commit -m "feat(C1+scripts): add run_scale_sweep.sh and scale-stamped filenames

run_scale_sweep.sh iterates scales × solvers, respects the POT memory
guard, and prints a report regen command at the end. Output filenames
now include __nNkK__ so results from different scales don't overwrite
each other."
```

---

## Task 4: Run full scale sweep and commit `docs/tier_core.md`

**Files:**
- Creates/updates: `results/*.json` (gitignored)
- Updates: `docs/tier_core.md` (committed)

This task runs the actual benchmarks and commits the results page. It is the only task with significant wall time (~15–30 minutes total).

- [ ] **Step 1: Clear old results (they use the old filename format)**

```bash
cd /scratch/users/chensj16/projects/torchgw-bench
rm -f results/*.json
```

- [ ] **Step 2: Run the full sweep (all 4 scales)**

```bash
source /scratch/users/chensj16/venvs/dl2025/.venv/bin/activate
cd /scratch/users/chensj16/projects/torchgw-bench
bash scripts/run_scale_sweep.sh --seed 0
```

Expected: 8 JSON files written (4 scales × 2 solvers; some pot-entropic rows will be `status=skip`).

Expected console output:
```
[sweep] N=400 K=500  solver=torchgw-landmark   → [C1] wrote ...
[sweep] N=400 K=500  solver=pot-entropic        → [C1] wrote ...
[sweep] N=4000 K=5000  solver=torchgw-landmark  → [C1] wrote ...
[sweep] N=4000 K=5000  solver=pot-entropic      → [C1] wrote ...
[sweep] N=10000 K=12000  solver=torchgw-landmark → [C1] wrote ...
[sweep] N=10000 K=12000  solver=pot-entropic     → [C1] skipped (POT memory guard) → ...
[sweep] N=20000 K=25000  solver=torchgw-landmark → [C1] wrote ...
[sweep] N=20000 K=25000  solver=pot-entropic     → [C1] skipped (POT memory guard) → ...
```

- [ ] **Step 3: Quick sanity check on the results**

```bash
cd /scratch/users/chensj16/projects/torchgw-bench
python -c "
import json, glob
for f in sorted(glob.glob('results/*.json')):
    r = json.load(open(f))
    ds = r.get('dataset', {})
    m = r.get('metrics', {})
    sp = m.get('task', {}).get('spearman_arclen')
    wall = m.get('efficiency', {}).get('wall_s')
    print(f\"{ds.get('n_source','?'):>6}x{ds.get('n_target','?'):<6} {r['solver']:<22} {r['status']:<5}  spearman={sp!s:<8} wall={wall!s}\")
"
```

Verify:
- All `torchgw-landmark` rows have `status=ok` and `spearman_arclen >= 0.95`
- All large-scale `pot-entropic` rows have `status=skip`
- `400×500 pot-entropic` has `status=ok`
- `4000×5000 pot-entropic` has `status=ok`

If any `torchgw-landmark` row has `spearman < 0.95`, investigate before proceeding. Common causes: OOM causing partial T, or wrong k value for a particular scale.

- [ ] **Step 4: Regenerate `docs/tier_core.md`**

```bash
source /scratch/users/chensj16/venvs/dl2025/.venv/bin/activate
cd /scratch/users/chensj16/projects/torchgw-bench
python scripts/make_report.py --format docs --results results/ --out docs/ --tier core
```

Expected stdout: `[reporter] wrote docs/tier_core.md`

- [ ] **Step 5: Visually inspect the table**

```bash
cat docs/tier_core.md
```

Expected shape (excerpt):
```markdown
| N×K | Solver | Status | GW cost | Spearman | Wall (s) | GPU peak (GB) | Iterations |
|---|---|:---:|---:|---:|---:|---:|---:|
| 400×500 | `torchgw-landmark` | ✓ | 0.0010 | 0.9989 | 7.08 | 0.04 | 300 |
| 400×500 | `pot-entropic` | ✓ | 0.0060 | 0.9994 | 2.27 | — | 3 |
| 4000×5000 | `torchgw-landmark` | ✓ | ... | ... | ... | ... | ... |
| 4000×5000 | `pot-entropic` | ✓ | ... | ... | ... | ... | ... |
| 10000×12000 | `torchgw-landmark` | ✓ | ... | ... | ... | ... | ... |
| 10000×12000 | `pot-entropic` | ⊘ skip | — | — | — | — | — |
|     |     | note: `skipped: POT O(N²) memory guard...` | | | | | |
| 20000×25000 | `torchgw-landmark` | ✓ | ... | ... | ... | ... | ... |
| 20000×25000 | `pot-entropic` | ⊘ skip | — | — | — | — | — |
```

- [ ] **Step 6: Commit**

```bash
cd /scratch/users/chensj16/projects/torchgw-bench
git add docs/tier_core.md
git commit -m "docs: update tier_core.md with Phase 1b scale sweep results

C1 Foundation track at 4 scales (400×500, 4k×5k, 10k×12k, 20k×25k).
torchgw-landmark passes spearman>=0.95 at all scales; pot-entropic skipped
above 5k via memory guard."
```

- [ ] **Step 7: Tag Phase 1b**

```bash
cd /scratch/users/chensj16/projects/torchgw-bench
git tag -a v0.1.0-m1b -m "Milestone 1b: C1 scale sweep at 4 scales (400x500 to 20kx25k)

torchgw-landmark benchmarked at 400×500, 4k×5k, 10k×12k, 20k×25k.
pot-entropic benchmarked at 400×500 and 4k×5k; skipped above 5k via
POT O(N²) memory guard. docs/tier_core.md updated with multi-scale table."
```

---

## Acceptance criteria

1. `python -m pytest` reports **≥35 tests passing** (23 track + 12 reporter).
2. `bash scripts/run_scale_sweep.sh --quick` produces 4 JSON files in `results/` with scale-stamped names.
3. All `torchgw-landmark` result JSONs have `status=ok` and `spearman_arclen >= 0.95`.
4. POT results at `max(N,K) > 5000` have `status=skip`.
5. `docs/tier_core.md` contains an 8-row table (4 scales × 2 solvers) with the `N×K` column.
6. Tag `v0.1.0-m1b` exists.

---

## Deferred

- **N > 25k scales** — GPU memory and wall time grow. Phase 1c.
- **POT at 4k×5k wall-time cap** — currently runs to completion (~2–3 min). If it times out on slower machines, add `--timeout` parameter. Phase 1c.
- **Multi-seed stability** — `stability.seed_std_spearman` requires 3+ seeds per scale. Phase 1c.
- **CNT-GW at large scales** — requires egw-solvers sys.path wiring. Phase 2.
