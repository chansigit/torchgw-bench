# C1 — Point-Cloud Scale Benchmark

This track is the **scalability proof** for torchgw: it demonstrates that
torchgw's Gromov-Wasserstein solver can process large 3-D point clouds (up to
~100 k vertices per shape) in settings where POT runs out of memory.  ModelNet40
meshes are loaded as raw vertex arrays via `io.py`, optionally downsampled with
farthest-point sampling, and then used as inputs for GW distance computation
across object classes.

See `docs/experiments/2026-04-19-c1-point-cloud-scale.md` for the full
experimental protocol, solver configurations, and results.
