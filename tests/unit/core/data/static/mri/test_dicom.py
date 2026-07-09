"""DICOM reader (core.data.static.mri.dicom). Round-trips a synthetic volume through a written DICOM
series and back — verifies the [D,H,W] shape, (z,y,x) spacing recovery (z from slice positions), and
tag extraction, with no external data. Closes the NIfTI-only gap (clinical data ships DICOM)."""
import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")
from core.data.static.mri.dicom import Dicom


def _write_series(dirpath, vol, spacing_xyz, vendor="Siemens"):
    """Write a [D,H,W] int16 volume as a DICOM series with z-spacing carried by slice positions."""
    import SimpleITK as sitk
    d, sx, sy, sz = vol.shape[0], *spacing_xyz
    series_uid = "1.2.826.0.1.3680043.2.1125.1.111111111111"
    w = sitk.ImageFileWriter(); w.KeepOriginalImageUIDOn()
    for i in range(d):
        sl = sitk.GetImageFromArray(vol[i][None].astype(np.int16))   # 1-slice 3D
        sl.SetSpacing((sx, sy, sz))
        for tag, val in [("0008|0060", "MR"), ("0020|000e", series_uid),
                         ("0020|0032", f"0\\0\\{i * sz}"),            # ImagePositionPatient (z steps)
                         ("0020|0013", str(i)),                       # InstanceNumber
                         ("0008|0070", vendor), ("0018|0080", "3.2"), ("0018|1314", "50")]:
            sl.SetMetaData(tag, val)
        w.SetFileName(str(dirpath / f"s{i:03d}.dcm")); w.Execute(sl)


def test_read_series_roundtrip(tmp_path):
    vol = (np.arange(5 * 8 * 8).reshape(5, 8, 8) % 97).astype(np.int16)   # [D,H,W]
    _write_series(tmp_path, vol, spacing_xyz=(1.5, 1.5, 8.0))
    assert len(Dicom.series_ids(tmp_path)) == 1
    arr, spacing, meta = Dicom.read_series(tmp_path)
    assert arr.shape == (5, 8, 8)                                   # [D,H,W] preserved
    assert np.allclose(spacing, (8.0, 1.5, 1.5), atol=1e-3)         # (z,y,x); z from slice positions
    assert np.array_equal(arr, vol)                                 # voxels intact
    assert meta.get("vendor") == "Siemens" and meta.get("flip_deg") == "50"


def test_no_series_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Dicom.read_series(tmp_path)
