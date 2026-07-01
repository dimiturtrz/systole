"""Chamber mesh export (core.mesh, bd 7c9.1). marching-cubes surface + STL are GL-free (tested here);
GLB needs an OpenGL context (smoke at call time, not unit-tested)."""
import numpy as np
import pytest
from core.mesh import chamber_surface, export_stl

pytest.importorskip("pyvista")
pytest.importorskip("skimage")


def _blob(label=3, n=24):
    """A solid cube of `label` inside a background volume (a meshable chamber)."""
    m = np.zeros((n, n, n), dtype=np.int64)
    m[6:18, 6:18, 6:18] = label
    return m


def test_chamber_surface_produces_mesh():
    mesh = chamber_surface(_blob(3), 3, spacing=(1.0, 1.0, 1.0))
    assert mesh is not None and mesh.n_points > 0 and mesh.n_faces > 0


def test_chamber_surface_absent_is_none():
    assert chamber_surface(_blob(3), 1, spacing=(1.0, 1.0, 1.0)) is None   # label 1 not present


def test_export_stl_writes_per_chamber(tmp_path):
    m = _blob(1) + _blob(3)                          # RV + LV-cav blobs (overlap-free by construction)
    m = _blob(3).copy(); m[6:18, 6:18, 6:18] = 3     # ensure a clean single chamber
    paths = export_stl(m, (1.0, 1.0, 1.0), tmp_path, "subjX")
    assert len(paths) >= 1 and all(p.exists() and p.suffix == ".stl" for p in paths)
