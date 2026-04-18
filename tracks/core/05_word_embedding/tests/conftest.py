"""Make the track's io.py and eval.py importable from tests in this directory."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

TRACK_DIR = Path(__file__).resolve().parent.parent
if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))


def _load_alias(filename: str, alias: str) -> None:
    """Load *filename* from TRACK_DIR under *alias* in sys.modules."""
    if alias not in sys.modules:
        spec = importlib.util.spec_from_file_location(alias, TRACK_DIR / filename)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        sys.modules[alias] = mod


# Pre-load modules under safe aliases so tests can do:
#   from conftest import word_io
#   from conftest import word_eval
# We load them here so the sys.path insertion above takes effect first.
_load_alias("io.py", "word_io")
_load_alias("eval.py", "word_eval")

# Re-export for direct import in test files
word_io = sys.modules["word_io"]
word_eval = sys.modules["word_eval"]
