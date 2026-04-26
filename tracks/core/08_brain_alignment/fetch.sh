#!/usr/bin/env bash
# C8 — download fsaverage meshes + Brainomics Localizer contrasts (12 subjects x 32 contrasts).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$REPO_ROOT/data/core_08_brain_alignment"
mkdir -p "$DATA_DIR"

env -u PYTHONPATH DATA_DIR="$DATA_DIR" micromamba run -n c8_brain python - <<'PY'
import os, pathlib
import nilearn.datasets as nd
data_dir = pathlib.Path(os.environ["DATA_DIR"])
data_dir.mkdir(parents=True, exist_ok=True)

print(f"[c8-fetch] downloading fsaverage5/6/7 meshes ...")
for res in ("fsaverage5", "fsaverage6", "fsaverage7"):
    fs = nd.fetch_surf_fsaverage(mesh=res, data_dir=str(data_dir / "fsaverage"))
    print(f"  {res}: {fs.pial_left}")

print(f"[c8-fetch] downloading Brainomics Localizer for 12 subjects x 32 contrasts ...")
# 32 human-readable contrast keys from nilearn's CONTRAST_NAME_WRAPPER,
# sorted alphabetically (matches manifest.txt indices 0-31).
ALL_CONTRASTS = [
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
assert len(ALL_CONTRASTS) == 32, f"Expected 32 contrasts, got {len(ALL_CONTRASTS)}"

loc = nd.fetch_localizer_contrasts(
    ALL_CONTRASTS, n_subjects=12, get_anats=False, get_masks=False,
    data_dir=str(data_dir / "localizer"),
)
n_subjects = len(set(loc.ext_vars["participant_id"]))
print(f"  Localizer loaded: {len(loc.cmaps)} contrast files for {n_subjects} subjects")
PY

echo "[c8-fetch] done."
