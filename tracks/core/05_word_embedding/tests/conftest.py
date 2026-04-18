"""Make the track's io.py importable from the tests in this directory."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

TRACK_DIR = Path(__file__).resolve().parent.parent
if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))

# Pre-load the module under a safe alias so tests can do:
#   from conftest import word_io
# or simply import it as 'word_io' after conftest runs.
# We load it here so the sys.path insertion above takes effect first.
if "word_io" not in sys.modules:
    spec = importlib.util.spec_from_file_location("word_io", TRACK_DIR / "io.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    sys.modules["word_io"] = mod
