"""DICOM I/O — read a DICOM series into the SAME (volume [D,H,W], spacing (z,y,x) mm) currency the
NIfTI adapters use (`base.load_nifti`). Closes the NIfTI-only gap: clinical cardiac data ships as
DICOM (Sunnybrook, Kaggle DSB, hospital PACS), and the whole pipeline downstream is format-agnostic
once a volume + spacing come out here.

SimpleITK's GDCM reader (already a dep via the `n4` extra) — no `pydicom`. A directory can hold several
series (SAX cine = many); `series_ids` lists them, `read_series` loads one (default: the largest, i.e.
the full stack). Slices are geometry-sorted by ImagePositionPatient. 4D cine grouping (slice × phase)
is DATASET-specific and lives in the per-dataset adapter, not here — this is the reusable 3D primitive.
"""
from __future__ import annotations

from pathlib import Path

from core.types import Spacing, Volume

# DICOM tags we surface for the reference/normalization store (mirrors what the NIfTI adapters carry).
_TAGS = {"0008|0070": "vendor", "0018|0087": "field_T", "0018|0080": "tr_ms",
         "0018|1314": "flip_deg", "0018|0050": "slice_mm", "0010|0040": "sex"}


def series_ids(dicom_dir: str | Path) -> list[str]:
    """The GDCM series UIDs present in `dicom_dir` (a SAX cine directory usually holds several)."""
    import SimpleITK as sitk
    return list(sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir)))


def read_series(dicom_dir: str | Path, series_id: str | None = None) -> tuple[Volume, Spacing, dict]:
    """Read ONE DICOM series from `dicom_dir` -> (array [D,H,W], spacing (z,y,x) mm, meta). With several
    series present, `series_id` picks one; default = the series with the most slices (the full stack).
    Slices are geometry-sorted. `meta` carries acquisition/demographic tags (vendor/field/TR/flip/…)
    when the header has them — the normalization hook, same as the NIfTI adapters' `meta()`."""
    import numpy as np
    import SimpleITK as sitk
    d = str(dicom_dir)
    ids = series_ids(d)
    if not ids:
        raise FileNotFoundError(f"no DICOM series found in {dicom_dir}")
    reader = sitk.ImageSeriesReader()
    if series_id is None:                                    # pick the fullest series (most files)
        series_id = max(ids, key=lambda s: len(reader.GetGDCMSeriesFileNames(d, s)))
    files = reader.GetGDCMSeriesFileNames(d, series_id)
    reader.SetFileNames(files)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    img = reader.Execute()
    arr = sitk.GetArrayFromImage(img)                        # SimpleITK -> [z, y, x] = [D, H, W]
    sx, sy, sz = img.GetSpacing()                            # SimpleITK spacing = (x, y, z)
    meta = {}
    for tag, name in _TAGS.items():                          # read from the first slice's header
        if reader.HasMetaDataKey(0, tag):
            meta[name] = reader.GetMetaData(0, tag).strip()
    return arr, (sz, sy, sx), meta
