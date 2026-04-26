# C8 fMRI Brain Alignment vs FUGW — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add fully-unbalanced (`rho_a, rho_b`) Sinkhorn to torchgw upstream (Phase A, ~150 LOC PR), then build `tracks/core/08_brain_alignment` that benchmarks 4 solvers (`fugw-native`, `pot-entropic-fgw`, `torchgw-balanced`, `torchgw-unbalanced`) on IBC fMRI inter-subject cortical alignment across fsaverage5/6/7 (Phase B).

**Architecture:** Two phases. Phase A modifies `/scratch/users/chensj16/projects/sgw/torchgw/_solver.py` (and 1 helper) to thread two-sided KL damping symmetrically through the existing single-`rho` semi-relaxed path; tests against POT and FUGW baselines. Phase B mirrors C2/C5/C7 structure: per-track `run.py` writes one JSON per (resolution, solver, pair, seed); a sweep script enumerates the matrix; a plotting script consumes the JSONs.

**Tech Stack:** torchgw (modified), POT, FUGW package, nilearn, nibabel, gdist, numpy, scikit-learn, matplotlib, pytest. New env `c8_brain` parallel to existing `c7_morph` (env isolation).

**Spec reference:** `docs/superpowers/specs/2026-04-26-c8-brain-alignment-design.md` — every numbered section below maps to that spec.

---

## Phase A — torchgw upstream PR (`/scratch/users/chensj16/projects/sgw/`)

### Task A1: Symmetrize `_sinkhorn_loop_pytorch` and `_sinkhorn_loop` to (tau_a, tau_b)

**Files:**
- Modify: `/scratch/users/chensj16/projects/sgw/torchgw/_solver.py:47-104` (`_sinkhorn_loop`)
- Modify: `/scratch/users/chensj16/projects/sgw/torchgw/_solver.py:106-144` (`_sinkhorn_loop_pytorch`)
- Test: `/scratch/users/chensj16/projects/sgw/tests/test_sinkhorn.py` (extend with new test)

The current implementation (line 124–128) applies `tau` only to `log_v`; `log_u` is always strict. We need both sides damped when fully unbalanced.

- [ ] **Step 1: Write a failing test for symmetric tau**

Append to `/scratch/users/chensj16/projects/sgw/tests/test_sinkhorn.py`:

```python
def test_sinkhorn_unbalanced_two_sided_marginals():
    """With small rho_a == rho_b, both source and target marginals relax."""
    import torch
    from torchgw._solver import _sinkhorn_torch
    n, m = 30, 40
    a = torch.full((n,), 1.0 / n, dtype=torch.float64)
    b = torch.full((m,), 1.0 / m, dtype=torch.float64)
    rng = torch.Generator().manual_seed(0)
    C = torch.rand(n, m, generator=rng, dtype=torch.float64)
    reg = 0.05
    # Balanced reference
    T_balanced = _sinkhorn_torch(a, b, C.clone(), reg, max_iter=500, tol=0,
                                 semi_relaxed=False, rho_a=1.0, rho_b=1.0)
    # Fully unbalanced (small rho relaxes both sides)
    T_unbal = _sinkhorn_torch(a, b, C.clone(), reg, max_iter=500, tol=0,
                              semi_relaxed=True, rho_a=0.1, rho_b=0.1)
    # Source marginal must drift away from a
    src_err_balanced = (T_balanced.sum(dim=1) - a).abs().max().item()
    src_err_unbal = (T_unbal.sum(dim=1) - a).abs().max().item()
    assert src_err_balanced < 1e-3, f"balanced source err {src_err_balanced}"
    assert src_err_unbal > 5e-3, f"unbalanced source should drift, got {src_err_unbal}"
    # Target marginal also drifts
    tgt_err_unbal = (T_unbal.sum(dim=0) - b).abs().max().item()
    assert tgt_err_unbal > 5e-3, f"unbalanced target should drift, got {tgt_err_unbal}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source ~/s/bin/activate-c7_morph 2>/dev/null || env -u PYTHONPATH micromamba run -n c7_morph bash -c '
  cd /scratch/users/chensj16/projects/sgw && pytest tests/test_sinkhorn.py::test_sinkhorn_unbalanced_two_sided_marginals -v
'
```

Expected: `TypeError: _sinkhorn_torch() got an unexpected keyword argument 'rho_a'` (since current API takes single `rho`).

- [ ] **Step 3: Modify `_sinkhorn_loop_pytorch` for two-sided damping**

Replace lines 122–142 of `/scratch/users/chensj16/projects/sgw/torchgw/_solver.py`:

```python
def _sinkhorn_loop_pytorch(
    log_K: torch.Tensor,
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    tau_a: float,
    tau_b: float,
    max_iter: int,
    tol: float,
    check_every: int,
    a: torch.Tensor,
    verbose: bool = False,
    log_u_init: torch.Tensor | None = None,
    log_v_init: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch Sinkhorn fallback. tau_a, tau_b control KL damping per side
    (1.0 = strict balanced; <1 = unbalanced KL relaxation)."""
    log_u = log_u_init if log_u_init is not None else torch.zeros_like(log_a)
    log_v = log_v_init if log_v_init is not None else torch.zeros_like(log_b)
    is_balanced_a = (tau_a == 1.0)
    is_balanced_b = (tau_b == 1.0)

    for it in range(max_iter):
        log_u_raw = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_u = log_u_raw if is_balanced_a else tau_a * log_u_raw + (1 - tau_a) * log_u
        log_v_raw = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
        log_v = log_v_raw if is_balanced_b else tau_b * log_v_raw + (1 - tau_b) * log_v

        if tol > 0 and (it + 1) % check_every == 0:
            log_marginal = log_u + torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
            marginal_err = torch.abs(torch.exp(log_marginal) - a).max().item()
            if verbose:
                print(f"    sinkhorn {it+1:>4}/{max_iter} | marginal_err: {marginal_err:.4e}")
            if marginal_err < tol:
                if verbose:
                    print(f"    sinkhorn converged at {it+1} (err={marginal_err:.4e})")
                break
    return log_u, log_v
```

- [ ] **Step 4: Update `_sinkhorn_loop` (compiled-iter path) signature**

Modify `/scratch/users/chensj16/projects/sgw/torchgw/_solver.py:47-104`. Current signature passes single `tau`; change to `tau_a, tau_b` and update the `is_balanced` branch:

```python
def _sinkhorn_loop(
    log_K: torch.Tensor, log_a: torch.Tensor, log_b: torch.Tensor,
    tau_a: float, tau_b: float, max_iter: int, tol: float, check_every: int,
    a: torch.Tensor, verbose: bool = False,
    log_u_init: torch.Tensor | None = None,
    log_v_init: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch order: Triton (CUDA only, balanced or single-tau semi-relaxed) →
    torch.compile (CUDA, both sides supported) → pure PyTorch fallback."""
    fully_unbalanced = (tau_a != 1.0) and (tau_b != 1.0) and (tau_a != tau_b or tau_a < 1.0)
    # Triton path: only handles balanced or single-tau-on-v (legacy semi-relaxed).
    # Fall back to PyTorch when both sides are damped.
    if log_K.is_cuda and not fully_unbalanced:
        try:
            from torchgw._triton_sinkhorn import triton_sinkhorn_loop
            tau_legacy = tau_b  # legacy single-tau was on the v side
            return triton_sinkhorn_loop(log_K, log_a, log_b, tau_legacy, max_iter,
                                        tol, check_every, a, verbose,
                                        log_u_init=log_u_init, log_v_init=log_v_init)
        except (ImportError, RuntimeError):
            pass
        # torch.compile path can handle both sides via the pure-PyTorch loop body
    return _sinkhorn_loop_pytorch(log_K, log_a, log_b, tau_a, tau_b,
                                   max_iter, tol, check_every, a, verbose,
                                   log_u_init=log_u_init, log_v_init=log_v_init)
```

> Note: the original `_sinkhorn_loop` had a torch.compile branch separate from
> the PyTorch fallback. Drop that branch — it duplicated the per-iter body and
> is incompatible with the two-tau API. PyTorch fallback already runs inside
> `torch.no_grad()` from the caller and is fast enough; we lose ~5-10 % single-
> case throughput in exchange for one consolidated loop body.

- [ ] **Step 5: Run test to verify it still fails (correctly, with new error)**

```bash
env -u PYTHONPATH micromamba run -n c7_morph bash -c '
  cd /scratch/users/chensj16/projects/sgw && pytest tests/test_sinkhorn.py::test_sinkhorn_unbalanced_two_sided_marginals -v
'
```

Expected: now fails inside `_sinkhorn_torch` because the caller still passes `rho` not `(rho_a, rho_b)`. That's what Task A2 fixes.

- [ ] **Step 6: Commit Task A1 partial (tests still failing, this is the "scaffolding" commit)**

```bash
cd /scratch/users/chensj16/projects/sgw
git add torchgw/_solver.py tests/test_sinkhorn.py
git commit -m "wip(sinkhorn): start two-sided unbalanced support — symmetric tau in loop body

Symmetrize _sinkhorn_loop_pytorch and _sinkhorn_loop to take (tau_a, tau_b)
instead of single tau. Triton fast-path falls back to PyTorch when fully
unbalanced. Caller signatures still single-rho — failing test
test_sinkhorn_unbalanced_two_sided_marginals tracks remaining work."
```

---

### Task A2: Thread `(rho_a, rho_b)` through `_sinkhorn_torch` / `_sinkhorn_unrolled` / `_sinkhorn_differentiable`

**Files:**
- Modify: `/scratch/users/chensj16/projects/sgw/torchgw/_solver.py:145-188` (`_sinkhorn_torch`)
- Modify: `/scratch/users/chensj16/projects/sgw/torchgw/_solver.py:266-294` (`_SinkhornApproximate.forward`)
- Modify: `/scratch/users/chensj16/projects/sgw/torchgw/_solver.py:294-380` (`_sinkhorn_unrolled` + `_sinkhorn_differentiable`)

- [ ] **Step 1: Update `_sinkhorn_torch` signature**

Replace at line 145–188:

```python
def _sinkhorn_torch(
    a: torch.Tensor, b: torch.Tensor, C: torch.Tensor, reg: float,
    max_iter: int = 100, tol: float = 5e-4, check_every: int = 10,
    semi_relaxed: bool = False,
    rho_a: float = 1.0, rho_b: float = 1.0,
    _inplace_C: bool = False, verbose: bool = False,
    log_u_init: torch.Tensor | None = None,
    log_v_init: torch.Tensor | None = None,
) -> torch.Tensor:
    """Log-domain Sinkhorn supporting balanced, single-side semi-relaxed,
    and fully-unbalanced via (rho_a, rho_b)."""
    log_K = C.neg_().div_(reg) if _inplace_C else -C / reg
    log_a = torch.log(a.clamp(min=1e-30))
    log_b = torch.log(b.clamp(min=1e-30))
    if semi_relaxed:
        tau_a = rho_a / (rho_a + reg)
        tau_b = rho_b / (rho_b + reg)
    else:
        tau_a = tau_b = 1.0
    log_u, log_v = _sinkhorn_loop(log_K, log_a, log_b, tau_a, tau_b,
                                   max_iter, tol, check_every, a,
                                   verbose=verbose,
                                   log_u_init=log_u_init, log_v_init=log_v_init)
    if log_K.is_cuda:
        try:
            from torchgw._triton_sinkhorn import triton_materialize_T
            T = triton_materialize_T(log_u, log_K, log_v)
        except (ImportError, RuntimeError):
            T = torch.exp(log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0))
    else:
        T = torch.exp(log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0))
    T._log_u = log_u.detach()  # type: ignore[attr-defined]
    T._log_v = log_v.detach()  # type: ignore[attr-defined]
    return T
```

- [ ] **Step 2: Update `_SinkhornApproximate.forward` signature**

Lines 274–294: replace `rho` parameter with `(rho_a, rho_b)`, compute both taus, and pass both into the loop. Pattern matches Step 1.

```python
@staticmethod
def forward(ctx, C, a, b, reg, max_iter, tol, check_every,
            semi_relaxed, rho_a, rho_b):
    if semi_relaxed:
        tau_a = rho_a / (rho_a + reg)
        tau_b = rho_b / (rho_b + reg)
    else:
        tau_a = tau_b = 1.0
    log_K = -C / reg
    log_a = torch.log(a.clamp(min=1e-30))
    log_b = torch.log(b.clamp(min=1e-30))
    log_u, log_v = _sinkhorn_loop(log_K, log_a, log_b, tau_a, tau_b,
                                   max_iter, tol, check_every, a, False)
    T = torch.exp(log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0))
    ctx.save_for_backward(T, a, b)
    ctx.reg = reg
    return T
```

- [ ] **Step 3: Update `_sinkhorn_unrolled` and `_sinkhorn_differentiable` signatures**

Both currently take single `rho`. Replace with `(rho_a, rho_b)`. Compute taus inside; otherwise the loop body code is identical to Step 1's pattern. Apply to both functions in lines 294–380.

```python
def _sinkhorn_unrolled(
    C, a, b, reg, max_iter=100, tol=5e-4, check_every=10,
    semi_relaxed=False, rho_a: float = 1.0, rho_b: float = 1.0,
    grad_mode="autograd", verbose=False,
):
    if semi_relaxed:
        tau_a = rho_a / (rho_a + reg)
        tau_b = rho_b / (rho_b + reg)
    else:
        tau_a = tau_b = 1.0
    log_K = -C / reg
    log_a = torch.log(a.clamp(min=1e-30))
    log_b = torch.log(b.clamp(min=1e-30))
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    is_a_balanced = (tau_a == 1.0)
    is_b_balanced = (tau_b == 1.0)
    for it in range(max_iter):
        log_u_raw = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_u = log_u_raw if is_a_balanced else tau_a * log_u_raw + (1 - tau_a) * log_u
        log_v_raw = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
        log_v = log_v_raw if is_b_balanced else tau_b * log_v_raw + (1 - tau_b) * log_v
        if tol > 0 and (it + 1) % check_every == 0:
            log_marginal = log_u + torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
            if torch.abs(torch.exp(log_marginal) - a).max().item() < tol:
                break
    return torch.exp(log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0))


def _sinkhorn_differentiable(
    C, a, b, reg, max_iter=100, tol=5e-4, check_every=10,
    semi_relaxed=False, rho_a: float = 1.0, rho_b: float = 1.0,
    grad_mode="autograd", verbose=False,
):
    if semi_relaxed:
        return _sinkhorn_unrolled(C, a, b, reg, max_iter, tol, check_every,
                                  semi_relaxed, rho_a, rho_b, grad_mode, verbose)
    if grad_mode == "implicit":
        return _SinkhornImplicit.apply(C, a, b, reg, max_iter, tol, check_every)
    if grad_mode == "approximate":
        return _SinkhornApproximate.apply(
            C, a, b, reg, max_iter, tol, check_every, semi_relaxed, rho_a, rho_b,
        )
    return _sinkhorn_unrolled(C, a, b, reg, max_iter, tol, check_every,
                              semi_relaxed, rho_a, rho_b, grad_mode, verbose)
```

- [ ] **Step 4: Run sinkhorn tests, expect failures only in `_gw_loop` callers**

```bash
env -u PYTHONPATH micromamba run -n c7_morph bash -c '
  cd /scratch/users/chensj16/projects/sgw && pytest tests/test_sinkhorn.py -v
'
```

Expected: `test_sinkhorn_unbalanced_two_sided_marginals` PASSES (Sinkhorn layer is now correct). Existing `test_semi_relaxed` may break — fix in Task A3 by updating the `_gw_loop` caller.

- [ ] **Step 5: Commit Task A2**

```bash
cd /scratch/users/chensj16/projects/sgw
git add torchgw/_solver.py
git commit -m "feat(sinkhorn): two-sided rho_a, rho_b in low-level Sinkhorn API

Sinkhorn machinery (_sinkhorn_torch, _sinkhorn_unrolled,
_sinkhorn_differentiable, _SinkhornApproximate) now takes (rho_a, rho_b)
and computes (tau_a, tau_b) internally. test_sinkhorn_unbalanced_two_sided
_marginals passes — both source and target marginals drift as expected
when rho_a == rho_b == 0.1.

Old single-rho semi_relaxed path callers in _gw_loop will be updated in
the next commit."
```

---

### Task A3: Update `_gw_loop` and public `sampled_gw` / `sampled_lowrank_gw` API

**Files:**
- Modify: `/scratch/users/chensj16/projects/sgw/torchgw/_solver.py:471-712` (`_gw_loop`)
- Modify: `/scratch/users/chensj16/projects/sgw/torchgw/_solver.py:772-940` (`sampled_gw` public)
- Modify: `/scratch/users/chensj16/projects/sgw/torchgw/_solver.py:960-1085` (`sampled_lowrank_gw` public)

- [ ] **Step 1: Replace `rho` with `(rho_a, rho_b)` in `_gw_loop` signature and forward to sinkhorn_fn**

In `_gw_loop` (~line 493), replace `rho: float` with `rho_a: float, rho_b: float`. Find every call to `sinkhorn_fn(...)` inside and replace `rho=rho` with `rho_a=rho_a, rho_b=rho_b`. There are two such calls (lines ~620 and ~633).

```python
def _gw_loop(
    *, N, K, provider, p_real, q_real, T_init, sinkhorn_fn, use_augmented,
    s_shared, fgw_alpha, C_lin_device, M, alpha, max_iter, tol, epsilon,
    min_iter_before_converge, device, verbose, verbose_every,
    semi_relaxed: bool, rho_a: float, rho_b: float,
    differentiable, lambda_ema_beta, mixed_precision,
):
    # ... existing body unchanged except sinkhorn_fn calls ...
    # Sinkhorn call(s):
    T_new = sinkhorn_fn(p_aug, q_aug, Lambda_aug, current_reg,
                        semi_relaxed=semi_relaxed, rho_a=rho_a, rho_b=rho_b,
                        max_iter=max_iter_inner, tol=tol_inner,
                        check_every=check_every_inner, _inplace_C=True)
```

- [ ] **Step 2: Update `sampled_gw` public signature**

Around line 772–798 in `/scratch/users/chensj16/projects/sgw/torchgw/_solver.py`. Replace the `rho` kwarg with `rho_a, rho_b`:

```python
def sampled_gw(
    X_source: np.ndarray | torch.Tensor | None = None,
    X_target: np.ndarray | torch.Tensor | None = None,
    p: np.ndarray | torch.Tensor | None = None,
    q: np.ndarray | torch.Tensor | None = None,
    *,
    distance_mode: str = "dijkstra",
    dist_source: np.ndarray | torch.Tensor | None = None,
    dist_target: np.ndarray | torch.Tensor | None = None,
    n_landmarks: int = 50,
    fgw_alpha: float = 0.0,
    C_linear: np.ndarray | torch.Tensor | None = None,
    s_shared: int | None = None,
    M: int = 50,
    alpha: float = 0.9,
    max_iter: int = 500,
    tol: float = 1e-5,
    epsilon: float = 0.001,
    k: int = 30,
    min_iter_before_converge: int = 50,
    device: torch.device | None = None,
    verbose: bool = False,
    verbose_every: int = 20,
    log: bool = False,
    semi_relaxed: bool = False,
    rho_a: float = 1.0,
    rho_b: float = 1.0,
    multiscale: bool = False,
    n_coarse: int | None = None,
    lambda_ema_beta: float | None = None,
    mixed_precision: bool = False,
    differentiable: bool = False,
    grad_mode: str = "autograd",
    T_init: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    # ... body unchanged except: pass rho_a, rho_b to _gw_loop where rho was passed
```

Inside the body, find `rho=rho` in the `_gw_loop(...)` call and replace with `rho_a=rho_a, rho_b=rho_b`.

- [ ] **Step 3: Update `sampled_lowrank_gw` public signature**

Same change as Step 2 but for `sampled_lowrank_gw` (line 960–1020). Replace `rho: float = 1.0` with `rho_a: float = 1.0, rho_b: float = 1.0`. Inside body, propagate via the existing semi_relaxed path. The lowrank path uses Dykstra projections — it does NOT support fully-unbalanced; if `rho_a != rho_b` and `semi_relaxed=True`, raise `NotImplementedError`:

```python
def sampled_lowrank_gw(
    # ... existing args, replace rho with rho_a, rho_b ...
    semi_relaxed: bool = False,
    rho_a: float = 1.0,
    rho_b: float = 1.0,
    # ... rest unchanged ...
):
    if semi_relaxed and rho_a != rho_b:
        raise NotImplementedError(
            "sampled_lowrank_gw does not yet support rho_a != rho_b "
            "(low-rank Dykstra requires symmetric KL). Use sampled_gw for "
            "fully-unbalanced (rho_a, rho_b) FGW."
        )
    # ... rest of body unchanged, pass rho_a, rho_b through ...
```

- [ ] **Step 4: Run all torchgw tests**

```bash
env -u PYTHONPATH micromamba run -n c7_morph bash -c '
  cd /scratch/users/chensj16/projects/sgw && pytest tests/ -v -x
'
```

Expected: all PASS, including legacy `test_semi_relaxed` (which now passes `rho` via `rho_a=rho_b=1.0` defaults — still relaxed because `semi_relaxed=True`). If any existing test breaks because it passed `rho=...`, update those test calls to use `rho_a=..., rho_b=...`.

- [ ] **Step 5: Commit Task A3**

```bash
cd /scratch/users/chensj16/projects/sgw
git add torchgw/_solver.py tests/
git commit -m "feat(api): public (rho_a, rho_b) kwargs in sampled_gw

sampled_gw and _gw_loop now expose two-sided KL marginal damping.
sampled_lowrank_gw raises NotImplementedError for rho_a != rho_b
(Dykstra path requires symmetric KL — out of scope). All existing tests
still pass; rho-only callers remain valid via default rho_a=rho_b=1.0."
```

---

### Task A4: Cross-validation tests against POT and FUGW + open PR

**Files:**
- Create: `/scratch/users/chensj16/projects/sgw/tests/test_unbalanced_crossval.py`

- [ ] **Step 1: Write a test that compares against POT's unbalanced Sinkhorn**

```python
"""Cross-validate the new (rho_a, rho_b) Sinkhorn against POT's reference
unbalanced Sinkhorn implementation."""
import numpy as np
import pytest
import torch


def test_sinkhorn_matches_pot_unbalanced_kl():
    """torchgw two-sided Sinkhorn at small reg should match POT's
    sinkhorn_unbalanced (KL divergence, balanced reg) within 1e-3."""
    pytest.importorskip("ot")
    import ot
    from torchgw._solver import _sinkhorn_torch
    rng = np.random.default_rng(0)
    n, m = 50, 60
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    C = rng.uniform(size=(n, m)).astype(np.float64)
    reg = 0.05
    rho = 0.5
    T_pot = ot.unbalanced.sinkhorn_unbalanced(
        a, b, C, reg=reg, reg_m=rho, method="sinkhorn", numItermax=2000,
    )
    T_torchgw = _sinkhorn_torch(
        torch.as_tensor(a), torch.as_tensor(b), torch.as_tensor(C),
        reg, max_iter=2000, tol=0,
        semi_relaxed=True, rho_a=rho, rho_b=rho,
    ).numpy()
    rel = np.linalg.norm(T_pot - T_torchgw) / (np.linalg.norm(T_pot) + 1e-12)
    assert rel < 5e-3, f"relative diff vs POT: {rel:.4e}"
```

- [ ] **Step 2: Run the cross-val test**

```bash
env -u PYTHONPATH micromamba run -n c7_morph bash -c '
  cd /scratch/users/chensj16/projects/sgw && pytest tests/test_unbalanced_crossval.py -v
'
```

Expected: PASS. If `rel` is in the 5e-3 to 5e-2 range, that's a sign of either (a) different stopping criteria — bump `numItermax` and `max_iter` higher, or (b) POT uses a slightly different parameterization (`reg_m` is the same as our `rho` only for KL divergence). If the disagreement is structural, document the difference rather than tuning the test to pass — the goal is to verify our implementation is correct, not to match POT exactly.

- [ ] **Step 3: Update CHANGELOG**

Append to `/scratch/users/chensj16/projects/sgw/CHANGELOG.md`:

```markdown
## [Unreleased]

### Added

- `sampled_gw(rho_a, rho_b, semi_relaxed=True)` — fully-unbalanced FGW with
  two-sided KL marginal damping. PyTorch fallback path only; Triton fast-
  path falls back to PyTorch when `rho_a != rho_b`. Lowrank
  (`sampled_lowrank_gw`) raises `NotImplementedError` when `rho_a != rho_b`.
- Cross-validation test against POT's `sinkhorn_unbalanced` to verify
  numerical correctness of the new path.

### Changed

- The `rho` kwarg on `sampled_gw` and `sampled_lowrank_gw` is now spelled
  `rho_a`, `rho_b`. Old `rho=X` calls should migrate to `rho_a=X, rho_b=X`.
```

- [ ] **Step 4: Commit + push branch + open PR**

```bash
cd /scratch/users/chensj16/projects/sgw
git add tests/test_unbalanced_crossval.py CHANGELOG.md
git commit -m "test(unbalanced): cross-validate against POT + CHANGELOG

The two-sided unbalanced Sinkhorn matches POT's sinkhorn_unbalanced KL
mode within 5e-3 relative error on random 50×60 problems. CHANGELOG
records the API change (rho → rho_a, rho_b) and the lowrank
NotImplementedError path."
git checkout -b feat-unbalanced-fgw
git push -u origin feat-unbalanced-fgw
# then open PR via gh
gh pr create --title "Add fully-unbalanced (rho_a, rho_b) FGW support" \
             --body "$(cat <<'EOF'
## Summary
Adds two-sided KL marginal damping to torchgw's Sinkhorn loop, exposed via
`(rho_a, rho_b)` kwargs on `sampled_gw` (and signature passthrough on
`sampled_lowrank_gw`). Enables fully-unbalanced FGW à la FUGW.

Pure-PyTorch fallback path only — Triton kernel still dispatched for
balanced and one-sided semi-relaxed cases.

## What changed
- `_sinkhorn_loop_pytorch` and `_sinkhorn_loop`: tau → (tau_a, tau_b)
- `_sinkhorn_torch`, `_sinkhorn_unrolled`, `_sinkhorn_differentiable`,
  `_SinkhornApproximate.forward`: signature carries (rho_a, rho_b)
- `_gw_loop`, `sampled_gw`: public API expose (rho_a, rho_b)
- `sampled_lowrank_gw`: raises NotImplementedError on rho_a != rho_b
- New cross-val test against POT's sinkhorn_unbalanced

## Test plan
- [x] `pytest tests/test_sinkhorn.py` — passes including new
      `test_sinkhorn_unbalanced_two_sided_marginals`
- [x] `pytest tests/test_unbalanced_crossval.py` — within 5e-3 of POT
- [x] `pytest tests/` — full suite passes, no regressions
EOF
)"
```

> The PR is opened on GitHub but does NOT need to be merged before C8 bench
> can use the new functionality — Phase B installs torchgw via `pip install
> -e /scratch/users/chensj16/projects/sgw` so the local working tree is the
> source of truth. The PR is for upstream visibility.

---

## Phase B — C8 bench track (`/scratch/users/chensj16/projects/torchgw-bench/`)

### Task B1: Track scaffolding + c8_brain env + FUGW backend probe

**Files:**
- Create: `tracks/core/08_brain_alignment/README.md`
- Create: `tracks/core/08_brain_alignment/env.yaml`
- Modify: `scripts/bootstrap_envs.sh` (append c8_brain stanza)
- Create: `tracks/core/08_brain_alignment/probe_fugw_backend.py`

- [ ] **Step 1: One-paragraph README**

```markdown
# C8 — fMRI brain alignment vs FUGW

4-solver shootout (fugw-native, pot-entropic-fgw, torchgw-balanced,
torchgw-unbalanced) on IBC inter-subject cortical alignment, three
freesurfer resolutions (fsaverage5/6/7). Tests whether torchgw's new
two-sided unbalanced Sinkhorn (upstream PR feat-unbalanced-fgw) closes
the quality gap to FUGW package, and whether sampled-MC scales past
the C1-found 30k vertex memory ceiling on real cortical meshes.
See `docs/superpowers/specs/2026-04-26-c8-brain-alignment-design.md`
for the design and `docs/experiments/2026-04-26-c8-brain-alignment.md`
for results.
```

- [ ] **Step 2: env.yaml**

```yaml
name: c8_brain
channels: [conda-forge]
dependencies:
  # Conda-only: things that need system libs / lmod conflicts.
  # The pip packages are installed by scripts/bootstrap_envs.sh with
  # PYTHONPATH unset (Sherlock lmod py-numpy bleeds in otherwise — same
  # issue as c7_morph).
  - python=3.11
  - hdf5
  - h5py
  - libgcc-ng
  - pip
```

- [ ] **Step 3: Append bootstrap stanza**

Append to `scripts/bootstrap_envs.sh`:

```bash
# C8 brain alignment — needs nilearn + fugw + gdist + editable torchgw.
# Editable torchgw install pulls our local PR branch's _solver.py.
if ! micromamba env list | grep -q '^c8_brain '; then
    env -u PYTHONPATH micromamba env create -f tracks/core/08_brain_alignment/env.yaml -y
    env -u PYTHONPATH micromamba run -n c8_brain pip install \
        nilearn nibabel fugw gdist scikit-learn matplotlib psutil pytest pot
    env -u PYTHONPATH micromamba run -n c8_brain pip install \
        -e /scratch/users/chensj16/projects/sgw
fi
```

- [ ] **Step 4: FUGW backend probe**

Create `tracks/core/08_brain_alignment/probe_fugw_backend.py`:

```python
"""Print FUGW's public API surface so the writeup can name the call site."""
import inspect
import importlib.metadata as md

print(f"fugw version: {md.version('fugw')}")
import fugw
print(f"top-level: {[x for x in dir(fugw) if not x.startswith('_')]}")

# The mappings module is the standard entry point in recent versions.
import fugw.mappings as fm
print(f"mappings: {[x for x in dir(fm) if not x.startswith('_')]}")

for cls_name in ("FUGW", "FUGWBarycenter", "FUGWSparse"):
    cls = getattr(fm, cls_name, None)
    if cls is None: continue
    print(f"\n=== {cls_name} ===")
    print(f"  __init__ sig: {inspect.signature(cls.__init__)}")
    if hasattr(cls, "fit"):
        print(f"  fit sig:      {inspect.signature(cls.fit)}")
    if hasattr(cls, "transform"):
        print(f"  transform sig: {inspect.signature(cls.transform)}")
```

- [ ] **Step 5: Run env bootstrap + probe**

```bash
bash scripts/bootstrap_envs.sh
env -u PYTHONPATH micromamba run -n c8_brain python \
    tracks/core/08_brain_alignment/probe_fugw_backend.py | tee \
    tracks/core/08_brain_alignment/fugw_probe.txt
```

Expected: prints version, top-level module contents, and at least one of
`FUGW`/`FUGWSparse` class signature. **Save the output verbatim** — Task
B5 (`solvers.py`) needs the actual call signature.

- [ ] **Step 6: Commit**

```bash
git add tracks/core/08_brain_alignment/README.md \
        tracks/core/08_brain_alignment/env.yaml \
        tracks/core/08_brain_alignment/probe_fugw_backend.py \
        tracks/core/08_brain_alignment/fugw_probe.txt \
        scripts/bootstrap_envs.sh
git commit -m "feat(C8): scaffolding + c8_brain env + FUGW backend probe"
```

---

### Task B2: IBC manifest + fetch.sh

**Files:**
- Create: `tracks/core/08_brain_alignment/manifest.txt`
- Create: `tracks/core/08_brain_alignment/fetch.sh`

- [ ] **Step 1: Write manifest skeleton with explicit subject IDs**

```text
# Manifest: 12 IBC subjects + train/test contrast split for C8 bench.
# Format: subject_id<TAB>train_contrasts<TAB>test_contrasts
# Train/test contrast indices are 0-based positions within the
# IBC contrast list returned by nilearn (which is alphabetically sorted).
# A 70/30 split by sorted index keeps it deterministic.
#
subject_id	train	test
sub-01	0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34	35,36,37,38,39,40,41,42,43,44,45,46,47,48,49
sub-04	0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34	35,36,37,38,39,40,41,42,43,44,45,46,47,48,49
sub-05	0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34	35,36,37,38,39,40,41,42,43,44,45,46,47,48,49
sub-06	0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34	35,36,37,38,39,40,41,42,43,44,45,46,47,48,49
sub-07	0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34	35,36,37,38,39,40,41,42,43,44,45,46,47,48,49
sub-08	0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34	35,36,37,38,39,40,41,42,43,44,45,46,47,48,49
sub-09	0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34	35,36,37,38,39,40,41,42,43,44,45,46,47,48,49
sub-11	0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34	35,36,37,38,39,40,41,42,43,44,45,46,47,48,49
sub-12	0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34	35,36,37,38,39,40,41,42,43,44,45,46,47,48,49
sub-13	0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34	35,36,37,38,39,40,41,42,43,44,45,46,47,48,49
sub-14	0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34	35,36,37,38,39,40,41,42,43,44,45,46,47,48,49
sub-15	0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34	35,36,37,38,39,40,41,42,43,44,45,46,47,48,49
```

> The 12 IBC subject IDs above are taken from the IBC public release
> (Pinho et al., as reused in the FUGW paper). If `nilearn`'s actual
> fetcher returns a different ID set, adjust accordingly during Step 2
> verification. The number of contrasts may also differ in the latest
> release — the 50-contrast assumption is a fallback; the real loader in
> `io.py` will use `len(contrasts)` and split 70/30 dynamically.

- [ ] **Step 2: fetch.sh — wrap nilearn IBC + fsaverage download**

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$REPO_ROOT/data/core_08_brain_alignment"
mkdir -p "$DATA_DIR"

env -u PYTHONPATH micromamba run -n c8_brain python - <<'PY'
import nilearn.datasets as nd
from nilearn import surface
import os, pathlib
data_dir = pathlib.Path(os.environ.get("DATA_DIR",
    "/scratch/users/chensj16/projects/torchgw-bench/data/core_08_brain_alignment"))
data_dir.mkdir(parents=True, exist_ok=True)
print(f"[c8-fetch] downloading fsaverage5/6/7 meshes ...")
for res in ("fsaverage5", "fsaverage6", "fsaverage7"):
    fs = nd.fetch_surf_fsaverage(mesh=res, data_dir=str(data_dir / "fsaverage"))
    print(f"  {res}: {fs.pial_left.split('/')[-1]}")
print(f"[c8-fetch] downloading IBC contrasts ...")
ibc = nd.fetch_ibc_contrasts(data_dir=str(data_dir / "ibc"))
print(f"  IBC release loaded: {len(ibc.images)} contrast files")
PY

echo "[c8-fetch] done."
```

- [ ] **Step 3: Make executable + verify download**

```bash
chmod +x tracks/core/08_brain_alignment/fetch.sh
bash tracks/core/08_brain_alignment/fetch.sh
ls data/core_08_brain_alignment/fsaverage/ | head -5
ls data/core_08_brain_alignment/ibc/ | head -5
```

Expected: fsaverage{5,6,7} subdirectories exist; IBC contrast files (.nii.gz or .gii) downloaded.

- [ ] **Step 4: Commit**

```bash
git add tracks/core/08_brain_alignment/manifest.txt \
        tracks/core/08_brain_alignment/fetch.sh
git commit -m "feat(C8): IBC + fsaverage fetch.sh + 12-subject manifest"
```

---

### Task B3: io.py — surface mesh + contrast map loaders

**Files:**
- Create: `tracks/core/08_brain_alignment/io.py`
- Create: `tracks/core/08_brain_alignment/__init__.py`
- Create: `tracks/core/08_brain_alignment/tests/__init__.py`
- Create: `tracks/core/08_brain_alignment/tests/conftest.py`
- Create: `tracks/core/08_brain_alignment/tests/test_io.py`

- [ ] **Step 1: Write conftest + failing test**

`tracks/core/08_brain_alignment/tests/conftest.py`:

```python
import sys, pathlib
TRACK = pathlib.Path(__file__).resolve().parents[1]
if str(TRACK) not in sys.path:
    sys.path.insert(0, str(TRACK))
```

`tracks/core/08_brain_alignment/__init__.py`: empty.

`tracks/core/08_brain_alignment/tests/__init__.py`: empty.

`tracks/core/08_brain_alignment/tests/test_io.py`:

```python
import numpy as np
import io_brain  # tracks/core/08_brain_alignment/io_brain.py


def test_load_fsaverage_mesh_returns_vertices_and_faces():
    verts, faces = io_brain.load_fsaverage_mesh("fsaverage5", hemi="left")
    assert verts.ndim == 2 and verts.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert verts.shape[0] == 10242  # fsaverage5 left hemi vertex count
    assert faces.dtype == np.int64
```

> Note: file is named `io_brain.py` not `io.py` to dodge the Python builtin
> `io` module conflict (lesson from C7).

- [ ] **Step 2: Run test, expect failure**

```bash
env -u PYTHONPATH micromamba run -n c8_brain pytest \
    tracks/core/08_brain_alignment/tests/test_io.py -v
```

Expected: `ModuleNotFoundError: No module named 'io_brain'`.

- [ ] **Step 3: Implement io_brain.py**

```python
"""IBC + fsaverage surface and contrast loaders. Thin wrapper over nilearn."""
from __future__ import annotations
import pathlib
import numpy as np
import nibabel as nb

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data" / "core_08_brain_alignment"


def _fsaverage_mesh_path(resolution: str, hemi: str) -> pathlib.Path:
    """Return path to a fsaverage white-matter surface .surf.gii or .surf."""
    base = DATA_ROOT / "fsaverage" / resolution
    # nilearn's fetch_surf_fsaverage stores meshes as
    # <res>/{lh,rh}.{pial,white,inflated,sulc} — pick white.
    candidates = list(base.rglob(f"{ 'lh' if hemi == 'left' else 'rh' }.white*"))
    if not candidates:
        raise FileNotFoundError(f"no white-matter mesh under {base} for {hemi}")
    return candidates[0]


def load_fsaverage_mesh(resolution: str, hemi: str = "left"
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices [N, 3] float32, faces [F, 3] int64)."""
    from nilearn import surface
    path = _fsaverage_mesh_path(resolution, hemi)
    verts, faces = surface.load_surf_mesh(str(path))
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def load_subject_contrasts(subject_id: str, resolution: str,
                           hemi: str = "left") -> np.ndarray:
    """Return (n_vertices × n_contrasts) float32 contrast matrix.

    Contrasts are sorted alphabetically by name for reproducibility.
    """
    from nilearn import surface, datasets
    ibc = datasets.fetch_ibc_contrasts(data_dir=str(DATA_ROOT / "ibc"))
    # Filter rows for this subject; sort by contrast name for determinism
    rows = sorted(
        [(c, p) for c, p, s in zip(ibc.contrasts, ibc.images, ibc.subjects)
         if s == subject_id],
        key=lambda r: r[0],
    )
    if not rows:
        raise ValueError(f"no IBC contrasts for {subject_id}")
    fsmesh = _fsaverage_mesh_path(resolution, hemi)
    cols = []
    for _name, vol_path in rows:
        v = surface.vol_to_surf(vol_path, str(fsmesh))
        cols.append(np.asarray(v, dtype=np.float32))
    return np.stack(cols, axis=1)  # (n_vertices, n_contrasts)
```

- [ ] **Step 4: Re-run test**

```bash
env -u PYTHONPATH micromamba run -n c8_brain pytest \
    tracks/core/08_brain_alignment/tests/test_io.py -v
```

Expected: PASS. If `nilearn.surface.load_surf_mesh` returns a different
shape, adjust the `np.asarray` calls.

- [ ] **Step 5: Commit**

```bash
git add tracks/core/08_brain_alignment/io_brain.py \
        tracks/core/08_brain_alignment/__init__.py \
        tracks/core/08_brain_alignment/tests/__init__.py \
        tracks/core/08_brain_alignment/tests/conftest.py \
        tracks/core/08_brain_alignment/tests/test_io.py
git commit -m "feat(C8): io_brain.py — fsaverage mesh + IBC contrast loaders"
```

---

### Task B4: precompute.py — cost matrices (sparse-aware) with disk cache

**Files:**
- Create: `tracks/core/08_brain_alignment/precompute.py`
- Create: `tracks/core/08_brain_alignment/tests/test_precompute.py`

- [ ] **Step 1: Write a failing test**

```python
import numpy as np
import precompute


def test_geodesic_matrix_small_mesh(tmp_path):
    # 5-vertex mesh on a line: 0-1-2-3-4
    verts = np.array([[i, 0.0, 0.0] for i in range(5)], dtype=np.float32)
    faces = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=np.int64)
    D = precompute.geodesic_matrix(verts, faces, sparse=False)
    assert D.shape == (5, 5)
    assert np.allclose(D, D.T)
    assert (np.diag(D) == 0).all()
    # 0 → 4 should be ~4 (along the line)
    assert 3.5 < D[0, 4] < 4.5
```

- [ ] **Step 2: Run test, expect failure**

```bash
env -u PYTHONPATH micromamba run -n c8_brain pytest \
    tracks/core/08_brain_alignment/tests/test_precompute.py -v
```

- [ ] **Step 3: Implement precompute.py**

```python
"""Per-subject precomputation: geodesic distance matrix + contrast features.

At fsaverage7 the dense geodesic matrix is 213 GB and cannot be stored;
sparse mode returns a (k-NN distance graph + Dijkstra) approximation, kept
as a CSR matrix only the FUGW solver consumes natively.
"""
from __future__ import annotations
import hashlib
import pathlib
import numpy as np
from scipy.sparse import csr_matrix


def _cache_key(verts: np.ndarray, faces: np.ndarray, sparse: bool) -> str:
    h = hashlib.sha256()
    h.update(verts.tobytes()); h.update(faces.tobytes())
    h.update(b"sparse" if sparse else b"dense")
    return h.hexdigest()[:16]


def geodesic_matrix(verts: np.ndarray, faces: np.ndarray,
                    sparse: bool = False, max_dist: float | None = None,
                    cache_dir: pathlib.Path | None = None
                    ) -> np.ndarray | csr_matrix:
    """Pairwise geodesic distance on a triangle mesh.

    sparse=False: dense (N, N) float64 matrix; only safe for N ≤ ~30 000.
    sparse=True:  CSR (N, N) up to max_dist; required for fsaverage7.
    """
    import gdist
    n = verts.shape[0]
    if cache_dir is not None:
        key = _cache_key(verts, faces, sparse)
        cache_file = pathlib.Path(cache_dir) / f"geo__{n}__{key}.npz"
        if cache_file.exists():
            from scipy.sparse import load_npz
            arr = load_npz(cache_file) if sparse else np.load(cache_file.with_suffix(".npy"))
            return arr
    if sparse:
        if max_dist is None:
            max_dist = 50.0  # fsaverage units; tune per-resolution if needed
        rows, cols, dists = [], [], []
        for src in range(n):
            d = gdist.compute_gdist(verts.astype(np.float64),
                                    faces.astype(np.int32),
                                    source_indices=np.array([src], dtype=np.int32),
                                    max_distance=max_dist)
            mask = d < max_dist
            rows.extend([src] * int(mask.sum()))
            cols.extend(np.where(mask)[0].tolist())
            dists.extend(d[mask].tolist())
        D = csr_matrix((dists, (rows, cols)), shape=(n, n))
    else:
        # Dense — call gdist per source vertex, fill row.
        D = np.zeros((n, n), dtype=np.float64)
        for src in range(n):
            D[src] = gdist.compute_gdist(verts.astype(np.float64),
                                         faces.astype(np.int32),
                                         source_indices=np.array([src], dtype=np.int32))
    if cache_dir is not None:
        if sparse:
            from scipy.sparse import save_npz
            save_npz(cache_file, D)
        else:
            np.save(cache_file.with_suffix(".npy"), D)
    return D


def feature_cost_matrix(F_a: np.ndarray, F_b: np.ndarray) -> np.ndarray:
    """Inter-subject vertex-vs-vertex feature cost = 1 - cosine similarity
    of train contrast vectors, range-normalized to [0, 1]."""
    Fa = F_a / (np.linalg.norm(F_a, axis=1, keepdims=True) + 1e-12)
    Fb = F_b / (np.linalg.norm(F_b, axis=1, keepdims=True) + 1e-12)
    C = 1.0 - Fa @ Fb.T  # (n_v_a, n_v_b)
    return C.astype(np.float64)
```

- [ ] **Step 4: Re-run test**

```bash
env -u PYTHONPATH micromamba run -n c8_brain pytest \
    tracks/core/08_brain_alignment/tests/test_precompute.py -v
```

Expected: PASS. If `gdist.compute_gdist` returns a different shape, follow
its docs (a 1-D array of length N for each source query is the standard).

- [ ] **Step 5: Commit**

```bash
git add tracks/core/08_brain_alignment/precompute.py \
        tracks/core/08_brain_alignment/tests/test_precompute.py
git commit -m "feat(C8): precompute.py — geodesic + feature cost matrices, sparse-aware"
```

---

### Task B5: solvers.py — 4-solver FGW dispatch

**Files:**
- Create: `tracks/core/08_brain_alignment/solvers.py`
- Create: `tracks/core/08_brain_alignment/tests/test_solvers.py`

- [ ] **Step 1: Write a failing test**

```python
import numpy as np
import solvers


def _two_random_costs(n: int = 20):
    rng = np.random.default_rng(0)
    A = rng.uniform(size=(n, n)); A = (A + A.T) / 2; np.fill_diagonal(A, 0)
    B = A.copy()
    Cl = np.zeros((n, n))
    return A, B, Cl


def test_pot_entropic_fgw_identical_is_small():
    A, B, Cl = _two_random_costs()
    out = solvers.fgw_pair("pot-entropic-fgw", A, B, Cl,
                           epsilon=5e-3, fgw_alpha=0.5, seed=0)
    assert out["fgw_objective"] < 0.05


def test_torchgw_balanced_identical_is_small():
    A, B, Cl = _two_random_costs()
    out = solvers.fgw_pair("torchgw-balanced", A, B, Cl,
                           epsilon=5e-3, fgw_alpha=0.5, seed=0)
    assert out["fgw_objective"] < 0.1


def test_torchgw_unbalanced_identical_is_small():
    A, B, Cl = _two_random_costs()
    out = solvers.fgw_pair("torchgw-unbalanced", A, B, Cl,
                           epsilon=5e-3, fgw_alpha=0.5, seed=0,
                           rho_a=1.0, rho_b=1.0)
    assert out["fgw_objective"] < 0.1
```

- [ ] **Step 2: Run test, expect failure**

```bash
env -u PYTHONPATH micromamba run -n c8_brain pytest \
    tracks/core/08_brain_alignment/tests/test_solvers.py -v
```

- [ ] **Step 3: Implement solvers.py**

```python
"""Single-pair FGW dispatch for the four C8 solvers.

All solvers consume the same (C_a, C_b, C_lin) triple; differences are
only in solver implementation.
"""
from __future__ import annotations
import time
import numpy as np


def _uniform(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n, dtype=np.float64)


def _fgw_pot(C_a, C_b, C_lin, epsilon, fgw_alpha, seed):
    import ot, torch
    a = _uniform(C_a.shape[0]); b = _uniform(C_b.shape[0])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Ca = torch.as_tensor(C_a, dtype=torch.float32, device=dev)
    Cb = torch.as_tensor(C_b, dtype=torch.float32, device=dev)
    Cl = torch.as_tensor(C_lin, dtype=torch.float32, device=dev)
    pa = torch.as_tensor(a, dtype=torch.float32, device=dev)
    pb = torch.as_tensor(b, dtype=torch.float32, device=dev)
    T, log = ot.gromov.entropic_fused_gromov_wasserstein(
        Cl, Ca, Cb, pa, pb,
        loss_fun="square_loss", alpha=fgw_alpha, epsilon=epsilon,
        log=True, max_iter=500,
    )
    return T.detach().cpu().numpy(), float(log.get("fgw_dist",
                                                   log.get("loss", float("nan"))))


def _fgw_torchgw(C_a, C_b, C_lin, epsilon, fgw_alpha, seed,
                 rho_a=None, rho_b=None):
    """torchgw-balanced if rho_{a,b} is None; torchgw-unbalanced otherwise."""
    import torch
    from torchgw import sampled_gw
    torch.manual_seed(seed)
    n, m = C_a.shape[0], C_b.shape[0]
    M_samples = max(min(n, 1000), 3 * n // 4)
    p = _uniform(n); q = _uniform(m)
    kwargs = dict(
        X_source=C_a, X_target=C_b, p=p, q=q,
        distance_mode="precomputed",
        dist_source=C_a.astype(np.float32), dist_target=C_b.astype(np.float32),
        fgw_alpha=fgw_alpha, C_linear=C_lin.astype(np.float32),
        mixed_precision=True,
        M=M_samples, epsilon=epsilon, max_iter=200,
        log=True, verbose=False,
    )
    if rho_a is not None:
        kwargs.update(semi_relaxed=True, rho_a=rho_a, rho_b=rho_b)
    T, log = sampled_gw(**kwargs)  # type: ignore[misc]
    return (T.detach().cpu().numpy(),
            float(log.get("gw_cost", log.get("fgw_cost", float("nan")))))


def _fgw_fugw(C_a, C_b, C_lin, epsilon, fgw_alpha, seed,
              rho_a=1.0, rho_b=1.0):
    """FUGW package call. Exact API verified at install via probe (Task B1)."""
    import torch
    from fugw.mappings import FUGW
    torch.manual_seed(seed)
    n, m = C_a.shape[0], C_b.shape[0]
    p = _uniform(n); q = _uniform(m)
    # API per probe; if signature differs adapt here per probe output.
    model = FUGW(alpha=fgw_alpha, rho=(rho_a, rho_b), eps=epsilon)
    model.fit(
        source_features=np.eye(n, dtype=np.float32),  # identity placeholder
        target_features=np.eye(m, dtype=np.float32),
        source_geometry=C_a.astype(np.float32),
        target_geometry=C_b.astype(np.float32),
        source_weights=p.astype(np.float32),
        target_weights=q.astype(np.float32),
        init_plan=None,
    )
    return model.pi.detach().cpu().numpy(), float(model.loss_steps[-1])


_DISPATCH = {
    "pot-entropic-fgw":    lambda Ca, Cb, Cl, eps, alpha, seed, **kw:
                                _fgw_pot(Ca, Cb, Cl, eps, alpha, seed),
    "torchgw-balanced":    lambda Ca, Cb, Cl, eps, alpha, seed, **kw:
                                _fgw_torchgw(Ca, Cb, Cl, eps, alpha, seed),
    "torchgw-unbalanced":  lambda Ca, Cb, Cl, eps, alpha, seed,
                                  rho_a=1.0, rho_b=1.0, **kw:
                                _fgw_torchgw(Ca, Cb, Cl, eps, alpha, seed,
                                             rho_a=rho_a, rho_b=rho_b),
    "fugw-native":         lambda Ca, Cb, Cl, eps, alpha, seed,
                                  rho_a=1.0, rho_b=1.0, **kw:
                                _fgw_fugw(Ca, Cb, Cl, eps, alpha, seed,
                                          rho_a, rho_b),
}


def fgw_pair(solver: str, C_a: np.ndarray, C_b: np.ndarray, C_lin: np.ndarray,
             *, epsilon: float, fgw_alpha: float, seed: int,
             rho_a: float = 1.0, rho_b: float = 1.0) -> dict:
    if solver not in _DISPATCH:
        raise ValueError(f"unknown solver {solver!r}")
    t0 = time.perf_counter()
    T, fgw_obj = _DISPATCH[solver](C_a, C_b, C_lin,
                                   epsilon, fgw_alpha, seed,
                                   rho_a=rho_a, rho_b=rho_b)
    return {"T": T, "fgw_objective": fgw_obj,
            "wall_s": time.perf_counter() - t0}
```

> Note: `fugw.mappings.FUGW` API based on the probe output of Task B1. If
> the probe shows different argument names (e.g. `rho` is a single float,
> not tuple, or `eps`/`epsilon` differs), adjust this file per the actual
> signature shown by `tracks/core/08_brain_alignment/fugw_probe.txt`. Do
> not invent fallbacks — fail loudly if the call doesn't work.

- [ ] **Step 4: Re-run tests**

```bash
env -u PYTHONPATH micromamba run -n c8_brain pytest \
    tracks/core/08_brain_alignment/tests/test_solvers.py -v
```

Expected: 3 passes (pot-entropic-fgw, torchgw-balanced, torchgw-unbalanced).
The fugw-native solver is NOT tested at this layer because its API may need
probe-driven adjustment first; add fugw smoke test in Task B7's run.py
smoke step instead.

- [ ] **Step 5: Commit**

```bash
git add tracks/core/08_brain_alignment/solvers.py \
        tracks/core/08_brain_alignment/tests/test_solvers.py
git commit -m "feat(C8): solvers.py — 4-way FGW dispatch (POT, torchgw bal/unbal, FUGW)"
```

---

### Task B6: eval.py — held-out functional correlation + retrieval + objective

**Files:**
- Create: `tracks/core/08_brain_alignment/eval_brain.py`
- Create: `tracks/core/08_brain_alignment/tests/test_eval.py`

> File named `eval_brain.py` to dodge the Python builtin `eval` function
> shadow risk (lesson from C7).

- [ ] **Step 1: Write a failing test**

```python
import numpy as np
import eval_brain


def test_eval_perfect_alignment_is_perfect():
    # Identity plan: predicted contrasts == actual contrasts → corr = 1
    rng = np.random.default_rng(0)
    F_test_a = rng.normal(size=(50, 4)).astype(np.float32)  # 50 vertices, 4 test contrasts
    F_test_b = F_test_a.copy()
    n = 50
    T = np.eye(n) / n  # identity plan
    out = eval_brain.eval_alignment(T, F_test_a, F_test_b)
    assert out["func_corr_holdout_mean"] > 0.99
    assert out["retrieval_top1"] == 1.0


def test_eval_random_alignment_is_chance():
    rng = np.random.default_rng(1)
    F_test_a = rng.normal(size=(50, 4)).astype(np.float32)
    F_test_b = rng.normal(size=(50, 4)).astype(np.float32)
    n = 50
    T = np.full((n, n), 1.0 / (n * n))  # uniform
    out = eval_brain.eval_alignment(T, F_test_a, F_test_b)
    assert abs(out["func_corr_holdout_mean"]) < 0.2
    assert out["retrieval_top1"] < 0.5
```

- [ ] **Step 2: Run test, expect failure**

```bash
env -u PYTHONPATH micromamba run -n c8_brain pytest \
    tracks/core/08_brain_alignment/tests/test_eval.py -v
```

- [ ] **Step 3: Implement eval_brain.py**

```python
"""Held-out functional correlation + retrieval evaluation.

Given an alignment plan T (n_a × n_b) and held-out test contrasts on each
subject, predict B's contrasts from A's via T and measure (a) vertex-wise
correlation, (b) contrast-level retrieval accuracy.
"""
from __future__ import annotations
import numpy as np


def _row_normalize(T: np.ndarray) -> np.ndarray:
    s = T.sum(axis=1, keepdims=True); s[s == 0] = 1.0
    return T / s


def eval_alignment(T: np.ndarray, F_test_a: np.ndarray, F_test_b: np.ndarray
                   ) -> dict:
    """T: (n_a, n_b) alignment plan; F_test_*: (n_v, n_test_contrasts)."""
    n_a, n_b = T.shape
    assert F_test_a.shape[0] == n_a, "F_test_a vertex count mismatch"
    assert F_test_b.shape[0] == n_b, "F_test_b vertex count mismatch"
    n_contrasts = F_test_a.shape[1]
    # Predict B's contrasts by transporting A's: F̂_b = (T.T @ F_a) / sum(T.T, axis=1)
    Tn = _row_normalize(T.T)  # (n_b, n_a) row-normalized
    F_pred_b = Tn @ F_test_a  # (n_b, n_contrasts)

    # Vertex-wise Pearson r per contrast, then mean over contrasts
    def _corr(x, y):
        xc = x - x.mean(); yc = y - y.mean()
        return float(np.dot(xc, yc) / (np.linalg.norm(xc) * np.linalg.norm(yc) + 1e-12))
    per_contrast_r = [_corr(F_pred_b[:, c], F_test_b[:, c]) for c in range(n_contrasts)]
    func_corr = float(np.mean(per_contrast_r))

    # Retrieval: for each predicted contrast, rank B's contrast set by cosine
    Fp = F_pred_b / (np.linalg.norm(F_pred_b, axis=0, keepdims=True) + 1e-12)
    Fb = F_test_b / (np.linalg.norm(F_test_b, axis=0, keepdims=True) + 1e-12)
    sim = Fp.T @ Fb  # (n_contrasts, n_contrasts)
    ranks = np.argsort(-sim, axis=1)
    top1 = float(np.mean(ranks[:, 0] == np.arange(n_contrasts)))
    top5 = float(np.mean(np.any(ranks[:, :5] == np.arange(n_contrasts)[:, None],
                                axis=1)))
    return {
        "func_corr_holdout_mean": func_corr,
        "func_corr_holdout_std":  float(np.std(per_contrast_r)),
        "retrieval_top1":         top1,
        "retrieval_top5":         top5,
    }
```

- [ ] **Step 4: Re-run test**

```bash
env -u PYTHONPATH micromamba run -n c8_brain pytest \
    tracks/core/08_brain_alignment/tests/test_eval.py -v
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add tracks/core/08_brain_alignment/eval_brain.py \
        tracks/core/08_brain_alignment/tests/test_eval.py
git commit -m "feat(C8): eval_brain.py — held-out functional corr + retrieval"
```

---

### Task B7: run.py — full pipeline per (resolution, solver, pair, seed)

**Files:**
- Create: `tracks/core/08_brain_alignment/run.py`

- [ ] **Step 1: Write run.py**

```python
#!/usr/bin/env python
"""C8 brain-alignment benchmark — one (resolution, solver, pair, seed) cell."""
from __future__ import annotations
import argparse
import datetime as _dt
import json
import os
import pathlib
import socket
import sys
import time
import numpy as np

TRACK = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TRACK))

import io_brain
import precompute
import eval_brain
import solvers


def _read_manifest() -> list[tuple[str, list[int], list[int]]]:
    out = []
    with open(TRACK / "manifest.txt") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("subject_id"):
                continue
            sid, train, test = line.split("\t")
            train_idx = [int(x) for x in train.split(",")]
            test_idx = [int(x) for x in test.split(",")]
            out.append((sid, train_idx, test_idx))
    return out


def _peak_rss_gb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 2**30
    except ImportError:
        return float("nan")


def _load_subject(sid: str, resolution: str, hemi: str, cache_dir: pathlib.Path):
    """Return dict with C_geo, F_train, F_test arrays for one subject."""
    F = io_brain.load_subject_contrasts(sid, resolution, hemi)
    verts, faces = io_brain.load_fsaverage_mesh(resolution, hemi)
    n_total = F.shape[1]
    # 70/30 split — agree with manifest's split-by-index encoding
    split_at = int(round(n_total * 0.7))
    F_train = F[:, :split_at]
    F_test = F[:, split_at:]
    sparse = (verts.shape[0] > 30000)
    C_geo = precompute.geodesic_matrix(verts, faces, sparse=sparse,
                                       cache_dir=cache_dir)
    return {"C_geo": C_geo, "F_train": F_train, "F_test": F_test, "n_v": verts.shape[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", required=True,
                    choices=["fsaverage5", "fsaverage6", "fsaverage7"])
    ap.add_argument("--solver", required=True, choices=[
        "fugw-native", "pot-entropic-fgw", "torchgw-balanced", "torchgw-unbalanced",
    ])
    ap.add_argument("--pair", required=True,
                    help="<sub_a>__<sub_b>, e.g. sub-01__sub-04")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epsilon", type=float, default=5e-3)
    ap.add_argument("--fgw-alpha", type=float, default=0.5)
    ap.add_argument("--rho", type=float, default=1.0,
                    help="(unbalanced solvers only) symmetric rho_a=rho_b")
    ap.add_argument("--hemi", default="left", choices=["left", "right"])
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    sub_a, sub_b = args.pair.split("__")
    cache_dir = (TRACK.parents[2] / "results" / "c8_brain_alignment"
                 / "_precompute_cache" / args.resolution)

    rec = {
        "track": "core/08_brain_alignment",
        "resolution": args.resolution, "solver": args.solver, "pair": args.pair,
        "seed": args.seed, "epsilon": args.epsilon, "fgw_alpha": args.fgw_alpha,
        "rho": args.rho, "hemi": args.hemi,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "status": "ok", "error": None,
        "metrics": {}, "efficiency": {},
    }
    out_file = args.out / (
        f"core_08_brain__{args.solver}__{args.resolution}"
        f"__{args.pair}__seed{args.seed}.json"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()

        t_load = time.perf_counter()
        sub_A = _load_subject(sub_a, args.resolution, args.hemi, cache_dir)
        sub_B = _load_subject(sub_b, args.resolution, args.hemi, cache_dir)
        rec["n_vertices"] = sub_A["n_v"]
        wall_load = time.perf_counter() - t_load

        # Inter-subject feature cost on train contrasts
        # (only dense path — fsaverage7 cost will OOM dense; flagged in spec §10)
        from scipy import sparse as _sp
        if _sp.issparse(sub_A["C_geo"]):
            # FUGW-only path: feed sparse C_geo to FUGW; other solvers OOM.
            if args.solver != "fugw-native":
                raise RuntimeError(
                    f"sparse geodesic at {args.resolution} only supported by "
                    f"fugw-native; {args.solver} requires dense (memory OOM expected)")
            C_a, C_b = sub_A["C_geo"], sub_B["C_geo"]
        else:
            C_a, C_b = sub_A["C_geo"], sub_B["C_geo"]
        C_lin = precompute.feature_cost_matrix(sub_A["F_train"], sub_B["F_train"])

        t_solve = time.perf_counter()
        out = solvers.fgw_pair(args.solver, C_a, C_b, C_lin,
                               epsilon=args.epsilon, fgw_alpha=args.fgw_alpha,
                               seed=args.seed, rho_a=args.rho, rho_b=args.rho)
        wall_solve = time.perf_counter() - t_solve
        T = out["T"]

        ev = eval_brain.eval_alignment(T, sub_A["F_test"], sub_B["F_test"])
        rec["metrics"] = {**ev, "fgw_objective": out["fgw_objective"]}
        rec["efficiency"] = {
            "wall_s_load":  float(wall_load),
            "wall_s_solve": float(wall_solve),
            "wall_s_total": float(wall_load + wall_solve),
            "gpu_peak_gb":  float(torch.cuda.max_memory_allocated() / 2**30)
                            if torch.cuda.is_available() else None,
            "cpu_peak_gb":  _peak_rss_gb(),
        }
    except Exception as e:
        rec["status"] = "fail"; rec["error"] = f"{type(e).__name__}: {e}"

    with open(out_file, "w") as fh:
        json.dump(rec, fh, indent=2, default=str)
    print(f"[c8] wrote {out_file} (status={rec['status']})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test on one pair at fsaverage5**

```bash
mkdir -p /tmp/c8_smoke
env -u PYTHONPATH micromamba run -n c8_brain python \
    tracks/core/08_brain_alignment/run.py \
    --resolution fsaverage5 --solver pot-entropic-fgw \
    --pair sub-01__sub-04 --seed 0 --out /tmp/c8_smoke
cat /tmp/c8_smoke/*.json | python3 -m json.tool | head -30
```

Expected: `"status": "ok"`, all metrics populated, sane wall times. Repeat
with `--solver torchgw-balanced` and `--solver torchgw-unbalanced` to
sanity-check torchgw side; finally with `--solver fugw-native` to verify
FUGW API call works (this is where the probe-driven solvers.py adjustment
shows up — fix per probe output if needed).

- [ ] **Step 3: Commit**

```bash
git add tracks/core/08_brain_alignment/run.py
git commit -m "feat(C8): run.py — one (resolution, solver, pair, seed) cell"
```

---

### Task B8: Bench sweep script

**Files:**
- Create: `scripts/run_c8_bench.sh`

- [ ] **Step 1: Write sweep script**

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT="$REPO_ROOT/results/c8_brain_alignment"
LOG="$REPO_ROOT/logs/c8_bench.log"
mkdir -p "$OUT" "$REPO_ROOT/logs"

RESOLUTIONS=(fsaverage5 fsaverage6 fsaverage7)
SOLVERS=(fugw-native pot-entropic-fgw torchgw-balanced torchgw-unbalanced)
SEEDS=(0 1 2)

# Parse subject IDs from manifest
mapfile -t SUBJECTS < <(awk -F'\t' '!/^#/ && !/^subject_id/ && NF>=3 {print $1}' \
                       "$REPO_ROOT/tracks/core/08_brain_alignment/manifest.txt")
N=${#SUBJECTS[@]}
echo "[c8] $N subjects, $((N*(N-1)/2)) pairs per (resolution, solver, seed) cell"

for resolution in "${RESOLUTIONS[@]}"; do
    for solver in "${SOLVERS[@]}"; do
        # Spec §8: pot-entropic-fgw OOM at fsaverage6+; skip cleanly
        if [[ "$solver" == "pot-entropic-fgw" && "$resolution" != "fsaverage5" ]]; then
            echo "[c8] skip $solver @ $resolution (spec §8 OOM)"; continue
        fi
        # torchgw-balanced expected to OOM at fsaverage7 per C1
        if [[ "$solver" == "torchgw-balanced" && "$resolution" == "fsaverage7" ]]; then
            echo "[c8] skip $solver @ $resolution (spec §8 OOM)"; continue
        fi
        for seed in "${SEEDS[@]}"; do
            for ((i=0; i<N; i++)); do
                for ((j=i+1; j<N; j++)); do
                    pair="${SUBJECTS[i]}__${SUBJECTS[j]}"
                    json="$OUT/core_08_brain__${solver}__${resolution}__${pair}__seed${seed}.json"
                    if [[ -s "$json" ]]; then
                        continue
                    fi
                    echo "[c8] $solver $resolution $pair seed$seed"
                    env -u PYTHONPATH micromamba run -n c8_brain python \
                        "$REPO_ROOT/tracks/core/08_brain_alignment/run.py" \
                        --resolution "$resolution" --solver "$solver" \
                        --pair "$pair" --seed "$seed" --out "$OUT" \
                        2>&1 | tee -a "$LOG"
                done
            done
        done
    done
done
echo "[c8] done."
```

- [ ] **Step 2: Make executable + dry-run validate**

```bash
chmod +x scripts/run_c8_bench.sh
bash -n scripts/run_c8_bench.sh && echo "syntax ok"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/run_c8_bench.sh
git commit -m "feat(C8): bench sweep script — 3 res × 4 solvers × 66 pairs × 3 seeds"
```

---

### Task B9: Plotting script

**Files:**
- Create: `scripts/experiments/make_c8_plots.py`

- [ ] **Step 1: Write plotting script**

```python
#!/usr/bin/env python
"""C8 plots: quality vs resolution, wall vs resolution, scatter func_corr
vs FGW objective, survival rate at fsaverage7."""
from __future__ import annotations
import argparse, json, pathlib
import matplotlib.pyplot as plt
import numpy as np

SOLVER_ORDER = ["fugw-native", "pot-entropic-fgw",
                "torchgw-balanced", "torchgw-unbalanced"]
SOLVER_COLOR = {
    "fugw-native":         "#444",
    "pot-entropic-fgw":    "#1f77b4",
    "torchgw-balanced":    "#9467bd",
    "torchgw-unbalanced":  "#d62728",
}
RES_ORDER = ["fsaverage5", "fsaverage6", "fsaverage7"]
RES_NV = {"fsaverage5": 10242, "fsaverage6": 40962, "fsaverage7": 163842}


def _load(results_dir: pathlib.Path) -> list[dict]:
    out = []
    for p in sorted(results_dir.glob("core_08_brain__*.json")):
        d = json.load(open(p))
        out.append(d)
    return out


def _by(records, key_path: tuple, only_ok: bool = True):
    bins: dict = {}
    for r in records:
        if only_ok and r.get("status") != "ok":
            continue
        v = r
        for k in key_path:
            v = v.get(k) if isinstance(v, dict) else None
        if v is None: continue
        bins.setdefault((r["solver"], r["resolution"]), []).append(float(v))
    return bins


def _plot_metric(records, key_path, ylabel, title, out_path, log_y=False):
    bins = _by(records, key_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    for solver in SOLVER_ORDER:
        xs, ys, errs = [], [], []
        for res in RES_ORDER:
            if (solver, res) not in bins: continue
            v = bins[(solver, res)]
            xs.append(RES_NV[res]); ys.append(np.mean(v)); errs.append(np.std(v))
        if xs:
            ax.errorbar(xs, ys, yerr=errs, marker="o", label=solver,
                        color=SOLVER_COLOR[solver])
    ax.set_xscale("log"); ax.set_xlabel("# vertices (per hemisphere)")
    if log_y: ax.set_yscale("log")
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def _plot_survival(records, out_path):
    """For each (solver, resolution) cell, fraction of pairs that ran ok."""
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.18
    x = np.arange(len(RES_ORDER))
    for i, solver in enumerate(SOLVER_ORDER):
        rates = []
        for res in RES_ORDER:
            cells = [r for r in records if r["solver"] == solver
                     and r["resolution"] == res]
            if not cells:
                rates.append(0.0); continue
            ok = sum(1 for r in cells if r.get("status") == "ok")
            rates.append(ok / len(cells))
        ax.bar(x + i * width - 1.5 * width, rates, width=width,
               label=solver, color=SOLVER_COLOR[solver])
    ax.set_xticks(x); ax.set_xticklabels(RES_ORDER); ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of pairs run successfully")
    ax.set_title("C8 — solver survival across resolutions")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main():
    repo = pathlib.Path(__file__).resolve().parents[2]
    rdir = repo / "results" / "c8_brain_alignment"
    figdir = repo / "docs" / "figures"; figdir.mkdir(exist_ok=True, parents=True)

    records = _load(rdir)
    if not records:
        raise SystemExit(f"no records in {rdir}")

    _plot_metric(records, ("metrics", "func_corr_holdout_mean"),
                 "Held-out functional Pearson r",
                 "C8 — alignment quality vs resolution",
                 figdir / "c8_quality.png")
    _plot_metric(records, ("metrics", "retrieval_top1"),
                 "Retrieval accuracy (top-1)",
                 "C8 — contrast retrieval vs resolution",
                 figdir / "c8_retrieval.png")
    _plot_metric(records, ("efficiency", "wall_s_total"),
                 "Wall time (s)",
                 "C8 — per-pair wall vs resolution",
                 figdir / "c8_wall.png", log_y=True)
    _plot_survival(records, figdir / "c8_survival.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it parses + --help**

```bash
python -c "import ast; ast.parse(open('scripts/experiments/make_c8_plots.py').read()); print('ok')"
env -u PYTHONPATH micromamba run -n c8_brain python \
    scripts/experiments/make_c8_plots.py --help 2>&1 | tail -5 || \
    echo "(--help may fail without records; that's fine pre-bench)"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/experiments/make_c8_plots.py
git commit -m "feat(C8): plotting — quality/retrieval/wall vs resolution + survival bars"
```

---

### Task B10: Writeup template + cross-track index entry

**Files:**
- Create: `docs/experiments/2026-04-26-c8-brain-alignment.md`
- Modify: `docs/experiments/README.md`

- [ ] **Step 1: Draft the writeup**

```markdown
# C8 — fMRI brain alignment vs FUGW (2026-04-26)

> **Status: scaffold — bench runs deferred. `<fill>` placeholders
> populated post-bench. See spec at
> `docs/superpowers/specs/2026-04-26-c8-brain-alignment-design.md` and
> plan at `docs/superpowers/plans/2026-04-26-c8-brain-alignment.md`.**

## Setup

Inter-subject cortical alignment on IBC (12 subjects × ~50 task contrasts).
Pipeline: per-subject fsaverage{5,6,7} surface + train-contrast feature
matrix → C_geo (mesh geodesic, sparse at fsaverage7) and C_lin (1 - cosine
on train features) per pair → FGW solver → apply plan to held-out test
contrasts → vertex-wise Pearson r + retrieval accuracy.

**FUGW backend (probe at install)**: `<fill from
tracks/core/08_brain_alignment/fugw_probe.txt>`.

## Solvers (4)
| Solver | Algorithm |
|---|---|
| fugw-native       | FUGW package, full unbalanced FGW (rho=1.0, eps=5e-3) |
| pot-entropic-fgw  | POT balanced entropic FGW |
| torchgw-balanced  | torchgw current main, balanced FGW |
| torchgw-unbalanced| torchgw + new (rho_a, rho_b) PR; symmetric rho=1.0 |

## Headline results

![quality](../figures/c8_quality.png)
![retrieval](../figures/c8_retrieval.png)
![wall](../figures/c8_wall.png)
![survival](../figures/c8_survival.png)

| solver | fsaverage5 r | fsaverage6 r | fsaverage7 r | wall (s) @ fs6 |
|---|---|---|---|---|
| fugw-native        | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| pot-entropic-fgw   | `<fill>` | OOM      | OOM      | `<fill>` |
| torchgw-balanced   | `<fill>` | `<fill>` | OOM (likely) | `<fill>` |
| torchgw-unbalanced | `<fill>` | `<fill>` | `<fill>` | `<fill>` |

## Take-home

1. **Algorithm gap closed?** `<fill from torchgw-unbalanced vs fugw-native
   func_corr difference at fsaverage5/6>`. If within ε_corr=0.02 → claim
   "the new Sinkhorn variant matches FUGW on real fMRI data."
2. **Scale ceiling lifted?** `<fill from fsaverage7 survival count>`.
   torchgw-unbalanced surviving fsaverage7 means the new path lifts the
   C1 30 k ceiling on real cortical meshes; not surviving means the
   ceiling is structural to the sampled-MC plan, not the Sinkhorn.
3. **Speed:** `<fill from wall ratio per resolution>`.

## Caveats

- torchgw-unbalanced is a NEW solver, not a FUGW reproduction. Inner
  Sinkhorn is correctly two-sided, but outer GW iteration uses torchgw's
  existing sampled-MC Lambda_gw — no Sejourne-style outer correction.
  Fixed point differs from FUGW package's by design.
- Triton fast-path falls back to PyTorch when `rho_a == rho_b != 1.0`;
  torchgw-unbalanced wall numbers are lower-bound estimates.
- fsaverage7 dense `C_geo` is 213 GB; sparse path used and only the
  fugw-native solver natively consumes sparse cost matrices.
- IBC release pinned via nilearn version in `env.yaml`; subject IDs
  pinned in `manifest.txt`.

## Reproducing

\`\`\`bash
micromamba activate c8_brain
bash tracks/core/08_brain_alignment/fetch.sh
bash scripts/run_c8_bench.sh
python scripts/experiments/make_c8_plots.py
\`\`\`
```

- [ ] **Step 2: Add C8 to cross-track table in `docs/experiments/README.md`**

Insert one column to the synthesis table (after C7 column):

```markdown
| C8 (fMRI alignment, large fused unbalanced GW)        |
| `<fill>`                                              |
| `<fill>`                                              |
| 12 subjects × 163k vertices/hemi (fsaverage7)         |
| mesh geodesic (dense fs5/6, sparse fs7) + cosine feat |
| 5e-3                                                  |
| 3N/4 capped 1000                                      |
| `<fill: pot OOM @ fs6, torchgw-bal OOM @ fs7?>`       |
| balanced + unbalanced both viable depending on PR     |
```

Plus a one-paragraph C8 section under existing track sections, link to
the writeup.

- [ ] **Step 3: Commit**

```bash
git add docs/experiments/2026-04-26-c8-brain-alignment.md docs/experiments/README.md
git commit -m "docs(C8): writeup template + cross-track index entry"
```

- [ ] **Step 4: Push**

```bash
git push origin main
```

---

## Self-review notes

- **Spec §3 PR pre-work**: Tasks A1–A4 implement the 5 file changes plus
  cross-validation test plus PR open.
- **Spec §4 dataset**: Task B2's manifest pins 12 subject IDs;
  fetch.sh wraps nilearn IBC + fsaverage downloads.
- **Spec §5 swap point invariant**: enforced by `solvers.fgw_pair`
  taking pre-built `(C_a, C_b, C_lin)` only — solvers cannot rebuild
  cost matrices.
- **Spec §6 4 solvers**: present in `solvers._DISPATCH`; each takes the
  shared `(epsilon, fgw_alpha, seed)` plus optional `(rho_a, rho_b)`.
- **Spec §7 metrics**: A/B/C/E all emitted by `eval_brain.eval_alignment`
  + `solvers.fgw_pair` + run.py efficiency dict.
- **Spec §8 skip rules**: bench script enforces pot-entropic-fgw skip
  beyond fsaverage5 and torchgw-balanced skip at fsaverage7.
- **Spec §10 caveats**: writeup template explicitly mentions all four:
  no Sejourne outer, fsaverage7 sparse, expected OOM, IBC pinning.
- **Spec §11 success criteria**: writeup template's "Take-home" section
  is explicitly the three claims from §11, with `<fill>` for the
  numbers.
- **Spec §12 non-goals**: no HCP fetch, no vertex_displacement metric, no
  Sejourne outer correction, no Triton extension. All confirmed absent.
- **Risk: nilearn fetch_ibc_contrasts API drift** — flagged in Task B3
  Step 3 note; recovery path is to grep the actual returned dict shape.
- **Risk: FUGW package API drift** — flagged in Task B5; probe runs
  before solvers.py implementation, output saved verbatim, solvers.py
  adjusts per probe.
- **Risk: gdist memory at fsaverage7** — sparse mode in Task B4 caps
  per-source distance computation at `max_dist=50` to keep memory
  bounded; tunable per resolution if needed.
