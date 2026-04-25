"""Print CAJAL's default pairwise-GW backend so the writeup can name it."""
import inspect
import importlib.metadata as md
import cajal.run_gw as run_gw

print(f"cajal version: {md.version('cajal')}")
print()
print("=== cajal.run_gw.gw (single pair) ===")
print(f"signature: {inspect.signature(run_gw.gw)}")
print(inspect.getsource(run_gw.gw))
print()
print("=== cajal.run_gw.gw_pairwise_parallel (batch) ===")
print(f"signature: {inspect.signature(run_gw.gw_pairwise_parallel)}")
src = inspect.getsource(run_gw.gw_pairwise_parallel)
for kw in ("entropic", "epsilon", "loss_fun", "ot.gromov", "gw_cython_core"):
    if kw in src:
        print(f"  mentions {kw!r}: yes")
