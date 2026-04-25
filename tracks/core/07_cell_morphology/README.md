# C7 — Cell morphology vs CAJAL

Reuses CAJAL's intracell-geodesic preprocessing and swaps only the pairwise-GW
solver. Compares `cajal-native`, `pot-entropic-gpu`, `pot-exact-gpu`, and
`torchgw-precomputed` on two stages (NeuroMorpho hand-picked + Allen CTDB).
See `docs/superpowers/specs/2026-04-25-c7-cell-morphology-design.md` for the
design and `docs/experiments/2026-04-25-c7-cell-morphology.md` for results.
