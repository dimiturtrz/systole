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

import SimpleITK as sitk

from core.types import Spacing, Volume

# DICOM tags we surface for the reference/normalization store (mirrors what the NIfTI adapters carry).
_TAGS = {"0008|0070": "vendor", "0008|1090": "scanner", "0008|0080": "institution",
         "0018|0087": "field_T", "0018|0080": "tr_ms", "0018|0081": "te_ms",
         "0018|1314": "flip_deg", "0018|0050": "slice_mm", "0020|1041": "slice_loc",
         "0018|1060": "trigger_ms", "0008|103e": "series_desc", "0010|0040": "sex", "0010|1010": "age"}


class Dicom:
    """DICOM reader — the reusable 3D primitive (series/instance -> volume + spacing + meta), folded from
    the free functions into staticmethods. `read_image` (one instance) and `read_series` (one series) are
    the public entry points imported by the per-dataset adapters (SCD, Kaggle DSB)."""

    @staticmethod
    def series_ids(dicom_dir: str | Path) -> list[str]:
        """The GDCM series UIDs present in `dicom_dir` (a SAX cine directory usually holds several)."""
        return list(sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir)))

    @staticmethod
    def read_image(dcm_path: str | Path) -> tuple["object", tuple, dict]:
        """Read ONE DICOM file -> (array [H,W], (row_mm, col_mm) in-plane spacing, meta tags). For datasets
        keyed by individual instances (e.g. SCD contours reference a specific slice/phase image), not a stack."""
        r = sitk.ImageFileReader()
        r.SetFileName(str(dcm_path))
        r.LoadPrivateTagsOn()
        img = r.Execute()
        arr = sitk.GetArrayFromImage(img)[0]                     # [1,H,W] -> [H,W]
        sx, sy = img.GetSpacing()[:2]                            # (col, row) mm
        meta = {}
        for tag, name in _TAGS.items():
            try:
                if r.HasMetaDataKey(tag):
                    meta[name] = r.GetMetaData(tag).strip()
            except RuntimeError:   # sitk raises RuntimeError on an unreadable/odd-encoded tag — skip it
                pass
        return arr, (sy, sx), meta

    @staticmethod
    def read_series(dicom_dir: str | Path, series_id: str | None = None) -> tuple[Volume, Spacing, dict]:
        """Read ONE DICOM series from `dicom_dir` -> (array [D,H,W], spacing (z,y,x) mm, meta). With several
        series present, `series_id` picks one; default = the series with the most slices (the full stack).
        Slices are geometry-sorted. `meta` carries acquisition/demographic tags (vendor/field/TR/flip/…)
        when the header has them — the normalization hook, same as the NIfTI adapters' `meta()`."""
        d = str(dicom_dir)
        ids = Dicom.series_ids(d)
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
