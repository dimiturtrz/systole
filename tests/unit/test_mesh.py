"""Chamber mesh export (core.mesh, bd 7c9.1). marching-cubes surface + STL are GL-free (tested here);
GLB needs an OpenGL context (smoke at call time, not unit-tested)."""
import numpy as np
import pytest

import core.mesh as meshmod
from core.mesh import chamber_surface, export_glb, export_meshes, export_stl

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


def test_chamber_surface_tiny_is_none():
    """Boundary: < _MIN_CHAMBER_VOXELS -> None (too small to mesh)."""
    m = np.zeros((16, 16, 16), np.int64)
    m[8, 8, 8] = 3                                    # single voxel
    assert chamber_surface(m, 3, spacing=(1.0, 1.0, 1.0)) is None


def test_chamber_surface_coarse_resample_none():
    """Boundary: a thin chamber whose coarse (iso>>voxel) resample drops below iso-level -> None."""
    m = np.zeros((16, 16, 16), np.int64)
    m[8, 6:10, 6:10] = 3                              # 1-voxel-thick sheet
    assert chamber_surface(m, 3, spacing=(1.0, 1.0, 1.0), iso=20.0) is None


def test_export_stl_writes_per_chamber(tmp_path):
    m = _blob(3).copy(); m[6:18, 6:18, 6:18] = 3     # a clean single chamber
    paths = export_stl(m, (1.0, 1.0, 1.0), tmp_path, "subjX")
    assert len(paths) >= 1 and all(p.exists() and p.suffix == ".stl" for p in paths)


def test_export_glb_writes_file(tmp_path):
    """export_glb: colored multi-chamber scene -> one .glb (needs off-screen GL, present in viz extra)."""
    m = _blob(3)
    out = export_glb(m, (1.0, 1.0, 1.0), tmp_path / "s.glb")
    assert out.exists() and out.suffix == ".glb"


def test_export_meshes_both_formats(tmp_path):
    """export_meshes: root override -> <root>/<subject>/ with GLB + STL written."""
    m = _blob(3)
    out = export_meshes(m, (1.0, 1.0, 1.0), "subjZ", root=tmp_path)
    assert out == tmp_path / "subjZ"
    assert (out / "subjZ.glb").exists()
    assert any(p.suffix == ".stl" for p in out.iterdir())


def test_export_meshes_glb_only(tmp_path):
    """export_meshes boundary: formats=('glb',) -> GLB written, no STL."""
    out = export_meshes(_blob(3), (1.0, 1.0, 1.0), "subjG", formats=("glb",), root=tmp_path)
    assert (out / "subjG.glb").exists()
    assert not any(p.suffix == ".stl" for p in out.iterdir())


def test_main_cli_reads_npz_and_exports(monkeypatch, tmp_path):
    """_main class: parses argv, loads the npz, dispatches export_meshes with the chosen frame/root."""
    npz = tmp_path / "case001.npz"
    np.savez(npz, ed_gt=_blob(3).astype(np.uint8),
             es_gt=_blob(3).astype(np.uint8), spacing=np.array([2.0, 1.0, 1.0], np.float32))
    seen = {}

    def _fake_export(mask, spacing, subject, formats, root=None):
        seen.update(mask=mask, spacing=spacing, subject=subject, formats=formats, root=root)
        return tmp_path / subject

    monkeypatch.setattr(meshmod, "setup", lambda: None)
    monkeypatch.setattr(meshmod, "export_meshes", _fake_export)
    monkeypatch.setattr("sys.argv",
                        ["mesh", "--npz", str(npz), "--frame", "es", "--formats", "stl"])
    meshmod._main()
    assert seen["subject"] == "case001"                  # default stem = npz filename
    assert seen["formats"] == ("stl",)
    assert seen["spacing"] == (2.0, 1.0, 1.0)            # spacing floats from npz
