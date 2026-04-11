"""Make `run.py` importable from the tests in this directory.

Tests in this directory import helpers defined at module level in `run.py`
(the track's self-contained CLI script). Because `run.py` is not packaged,
we insert its parent directory into sys.path so pytest can import it.
"""
import sys
from pathlib import Path

TRACK_DIR = Path(__file__).resolve().parent.parent
if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))
