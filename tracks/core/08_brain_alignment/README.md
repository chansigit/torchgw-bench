# C8 — fMRI brain alignment vs FUGW

4-solver shootout (fugw-native, pot-entropic-fgw, torchgw-balanced,
torchgw-unbalanced) on Brainomics Localizer inter-subject cortical alignment,
three freesurfer resolutions (fsaverage5/6/7). Tests whether torchgw's new
two-sided unbalanced Sinkhorn (upstream PR feat-unbalanced-fgw) closes
the quality gap to FUGW package, and whether sampled-MC scales past
the C1-found 30k vertex memory ceiling on real cortical meshes.
See `docs/superpowers/specs/2026-04-26-c8-brain-alignment-design.md`
for the design and `docs/experiments/2026-04-26-c8-brain-alignment.md`
for results.
