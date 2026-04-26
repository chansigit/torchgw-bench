"""Brainomics Localizer + fsaverage surface loaders.

Naming: file is `io_brain.py` (not `io.py`) to avoid shadowing Python's
builtin `io` module — lesson from C7. Tests import as `import io_brain`.
"""
from __future__ import annotations
import pathlib
import numpy as np
import nibabel as nb

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data" / "core_08_brain_alignment"

# Canonical 32 Brainomics Localizer contrasts, sorted alphabetically for
# deterministic train/test splitting.
# Source: nilearn.datasets.func.CONTRAST_NAME_WRAPPER (nilearn >= 0.10).
ALL_LOCALIZER_CONTRASTS: list[str] = [
    "button press (auditory cue) vs sentence listening",
    "button press (visual cue) vs sentence reading",
    "button press vs calculation and sentence listening/reading",
    "calculation (auditory and visual cue)",
    "calculation (auditory cue)",
    "calculation (auditory cue) and sentence listening",
    "calculation (auditory cue) and sentence listening vs calculation (visual cue) and sentence reading",
    "calculation (auditory cue) vs sentence listening",
    "calculation (visual cue)",
    "calculation (visual cue) and sentence reading",
    "calculation (visual cue) and sentence reading vs calculation (auditory cue) and sentence listening",
    "calculation (visual cue) and sentence reading vs checkerboard",
    "calculation (visual cue) vs sentence reading",
    "calculation and sentence listening/reading vs button press",
    "calculation vs sentences",
    "checkerboard",
    "horizontal checkerboard",
    "horizontal vs vertical checkerboard",
    "left button press",
    "left button press (auditory cue)",
    "left button press (visual cue)",
    "left vs right button press",
    "right button press",
    "right button press (auditory cue)",
    "right button press (visual cue)",
    "right vs left button press",
    "sentence listening",
    "sentence listening and reading",
    "sentence reading",
    "sentence reading vs checkerboard",
    "vertical checkerboard",
    "vertical vs horizontal checkerboard",
]


# ── fsaverage surface loaders ────────────────────────────────────────

def load_fsaverage_mesh(resolution: str, hemi: str = "left"
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices [N, 3] float32, faces [F, 3] int64) for the requested
    fsaverage{5,6,7} resolution and hemisphere.

    Uses nilearn's bundled fsaverage5 + on-disk fsaverage6/7 download.
    """
    from nilearn import datasets, surface
    fs = datasets.fetch_surf_fsaverage(
        mesh=resolution, data_dir=str(DATA_ROOT / "fsaverage"))
    key = "pial_left" if hemi == "left" else "pial_right"
    verts, faces = surface.load_surf_mesh(fs[key])
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int64)


# ── Brainomics Localizer contrast loaders ────────────────────────────

def list_localizer_contrasts() -> list[str]:
    """Return all 32 Brainomics Localizer contrasts in alphabetical order.

    Pulled from nilearn's CONTRAST_NAME_WRAPPER (nilearn.datasets.func),
    sorted for deterministic train/test splitting. The list is also hardcoded
    as ALL_LOCALIZER_CONTRASTS for offline use.
    """
    return list(ALL_LOCALIZER_CONTRASTS)


def load_localizer_contrast_volume(subject_id: str, contrast: str) -> np.ndarray:
    """Return the 3D MNI152 t-map (or contrast map) for one (subject, contrast).

    The on-disk Brainomics Localizer cache stores one NIfTI per (subject,
    contrast). nilearn's fetcher in cache mode reads them without re-download.

    Parameters
    ----------
    subject_id : str
        Subject identifier as stored in the Brainomics dataset, e.g. ``'S01'``.
    contrast : str
        Contrast name, one of :func:`list_localizer_contrasts`.

    Returns
    -------
    np.ndarray
        3D float64 array of shape (X, Y, Z).
    """
    from nilearn import datasets
    # 12 subjects (S01–S12) on disk per manifest.txt; nilearn returns up to
    # n_subjects from cache without re-downloading.
    loc = datasets.fetch_localizer_contrasts(
        [contrast], n_subjects=12,
        data_dir=str(DATA_ROOT / "localizer"),
        get_anats=False, get_masks=False, verbose=0,
    )
    df = loc.ext_vars
    matches = df.index[df["participant_id"] == subject_id].tolist()
    if not matches:
        raise ValueError(
            f"subject {subject_id!r} not found in localizer cache "
            f"(available: {list(df['participant_id'])})"
        )
    idx = matches[0]
    img = nb.load(loc.cmaps[idx])
    return np.asarray(img.get_fdata())


def load_subject_contrasts_volume(subject_id: str
                                  ) -> tuple[np.ndarray, list[str]]:
    """Return (4D stacked contrast tensor [X, Y, Z, n_contrasts], contrast_names).

    Iterates list_localizer_contrasts() in alphabetical order so the contrast
    axis is deterministic (matches manifest train/test indices).
    """
    contrasts = list_localizer_contrasts()
    vols = [load_localizer_contrast_volume(subject_id, c) for c in contrasts]
    return np.stack(vols, axis=-1), contrasts
