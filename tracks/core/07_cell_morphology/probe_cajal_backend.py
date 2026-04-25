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
