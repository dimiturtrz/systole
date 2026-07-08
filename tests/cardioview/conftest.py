"""Put the cardioview package dir on sys.path so its modules' bare intra-package imports
(`from common import ...`, `from geometry import ...`) resolve under pytest — that's how they
run as scripts (`python cardioview/render_volume.py`). Lets the logic tests import the modules
by bare name (`import render_volume`) exactly as production does, while the geometry tests keep
using the dotted `cardioview.geometry` form (both resolve)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cardioview"))
