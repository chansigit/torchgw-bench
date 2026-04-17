"""Make `run.py` importable from the tests in this directory."""
import sys
from pathlib import Path

TRACK_DIR = Path(__file__).resolve().parent.parent
if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))
