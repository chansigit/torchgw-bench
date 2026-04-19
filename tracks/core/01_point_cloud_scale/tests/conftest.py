"""Make the track's io.py importable from tests without shadowing stdlib io."""
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


_load_alias("io.py", "c1_io")
_load_alias("pair.py", "c1_pair")
_load_alias("eval.py", "c1_eval")

# Re-export for direct import in test files
c1_io   = sys.modules["c1_io"]
c1_pair = sys.modules["c1_pair"]
c1_eval = sys.modules["c1_eval"]
