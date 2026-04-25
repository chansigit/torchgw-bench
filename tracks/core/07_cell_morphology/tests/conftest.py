import sys, pathlib
import importlib.util

TRACK = pathlib.Path(__file__).resolve().parents[1]

# Load our local io.py module and register it as 'io' in sys.modules
# This must happen before test module tries to import io
io_path = TRACK / "io.py"
spec = importlib.util.spec_from_file_location("io", io_path)
track_io = importlib.util.module_from_spec(spec)
# Register in sys.modules BEFORE executing to handle circular imports
sys.modules['io'] = track_io
spec.loader.exec_module(track_io)

sys.path.insert(0, str(TRACK))
