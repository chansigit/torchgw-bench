import sys, pathlib

TRACK = pathlib.Path(__file__).resolve().parents[1]
if str(TRACK) not in sys.path:
    sys.path.insert(0, str(TRACK))
