# C5 · Word Embedding Alignment

Reproduces the cross-lingual word-embedding alignment experiments from
Alvarez-Melis & Jaakkola (2018) *"Gromov-Wasserstein Alignment of Word Embedding Spaces"*.
GW distance is used to align fastText wiki vectors (en ↔ es, en ↔ fi) without any
cross-lingual supervision, and translation accuracy (P@1/P@5) is measured against
MUSE bilingual dictionaries.
See `docs/experiments/2026-04-18-c5-word-embedding.md` for the full writeup.
