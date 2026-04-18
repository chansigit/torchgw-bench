from __future__ import annotations

"""C5 word-embedding track — vector/dictionary IO + cost-matrix utilities."""

import numpy as np


def read_fasttext(path: str, N: int) -> tuple[list[str], np.ndarray]:
    """Read first N word vectors from a fastText .vec file.

    Parameters
    ----------
    path:
        Path to a fastText ``.vec`` file whose first line is ``vocab dim``.
    N:
        Number of word vectors to read.

    Returns
    -------
    words:
        List of N tokens (no lowercasing; fastText wiki vectors are already
        lowercase).
    V:
        Float32 array of shape ``(N, dim)``.
    """
    words: list[str] = []
    rows: list[np.ndarray] = []

    with open(path, "r", encoding="utf-8") as fh:
        # First line is the header: "<vocab> <dim>"
        _ = fh.readline()
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(" ")
            words.append(parts[0])
            rows.append(np.array(parts[1:], dtype=np.float32))
            if len(words) == N:
                break

    V = np.stack(rows, axis=0)  # (N, dim)
    return words, V


def read_muse_dict(path: str) -> dict[str, set[str]]:
    """Read a MUSE bilingual dictionary file.

    Each line is ``word_src word_tgt`` (whitespace-separated).  Returns a
    mapping from (lowercased) source word to the set of (lowercased) target
    translations.

    Parameters
    ----------
    path:
        Path to a MUSE ``.txt`` dictionary file.

    Returns
    -------
    d:
        ``dict[str, set[str]]`` — one-to-many mapping.
    """
    d: dict[str, set[str]] = {}

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            src, tgt = parts[0].lower(), parts[1].lower()
            d.setdefault(src, set()).add(tgt)

    return d


def cosine_cost(V: np.ndarray) -> np.ndarray:
    """Pairwise cosine-distance matrix for rows of V.

    Internally L2-normalises V (defensive copy — cheap, prevents user error).

    Parameters
    ----------
    V:
        Float array of shape ``(N, dim)``.

    Returns
    -------
    C:
        Float32 symmetric matrix of shape ``(N, N)`` with ``C[i, j] = 1 -
        cos(v_i, v_j)``.  Diagonal is 0; off-diagonal in ``[0, 2]``.
    """
    V = np.array(V, dtype=np.float32)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    # Avoid division by zero for zero vectors
    norms = np.where(norms == 0.0, 1.0, norms)
    V = V / norms
    C = (1.0 - V @ V.T).astype(np.float32)
    # Clip tiny floating-point negatives on the diagonal
    np.fill_diagonal(C, 0.0)
    return C


def range_normalize(C: np.ndarray) -> np.ndarray:
    """Rescale C so that min=0 and max=1.

    Critical for hard language pairs where intra-lingual cost-matrix scales
    differ across languages (paper Section 4).

    Parameters
    ----------
    C:
        Float array (any shape).

    Returns
    -------
    C_norm:
        Float32 array of the same shape with values in ``[0, 1]``.
    """
    C = np.array(C, dtype=np.float32)
    lo, hi = C.min(), C.max()
    if hi == lo:
        return np.zeros_like(C, dtype=np.float32)
    return ((C - lo) / (hi - lo)).astype(np.float32)
