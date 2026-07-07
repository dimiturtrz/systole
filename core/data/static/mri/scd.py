"""Sunnybrook Cardiac Data (SCD) adapter — the DICOM lane (CC0 1.0). LV-ONLY: endo/epi manual contours →
canonical {0 bg, 2 LV-myo, 3 LV-cav}; NO RV (SCD doesn't label it), so masks carry classes {0,2,3}.

Everything here was verified against the real data (per the label-encoding lesson — geometry, not guesses):
- **images**: `<raw>/sunnybrook/SCD00001XX/CINESAX_*/IM-*-NNNN.dcm` — the short-axis cine (the patient dir
  also holds CINELAX / PERF / SCOUT series we ignore; the SAX series is the one with SeriesDescription
  'CINESAX'). 1.37 mm in-plane, ~256².
- **contours**: `<raw>/sunnybrook/SCD_ManualContours/<origID>/contours-manual/IRCCI-expert/
  IM-0001-NNNN-{i,o}contour-manual.txt` — i=inner=endo, o=outer=epi; float `x y` pixel coords. The
  contour series prefix (0001) is the ORIGINAL series number; match by **instance NNNN** within the
  CINESAX series (CAP renumbered the series). Some slices have only the icontour (no epi → no myo there).
- **ED/ES**: two phases contoured per slice (same SliceLocation). ED = LARGER endo area, ES = smaller —
  assigned by geometry, never instance order.
- **ids**: `scd_patientdata.csv` maps CAP PatientID (SCD0000101) ↔ OriginalID (SC-HF-I-1) + Gender/Age/
  Pathology; the contour dir zero-pads the trailing number (SC-HF-I-01).
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from skimage.draw import polygon

from core.config import data_root
from core.data.static.mri.base import DatasetAdapter, PatientData, load_csv_info
from core.data.static.mri.dicom import read_image

_IRCCI = "contours-manual/IRCCI-expert"


def _root(root: str | Path | None = None) -> Path:
    return Path(root) if root else Path(data_root("raw")) / "sunnybrook"


def _patient_csv(root: Path) -> dict[str, dict[str, str]]:
    return load_csv_info(root / "scd_patientdata.csv", key_col="PatientID")


def _contour_dir_name(original_id: str) -> str:
    """CSV OriginalID 'SC-HF-I-1' -> contour-folder name 'SC-HF-I-01' (trailing number zero-padded to 2)."""
    return re.sub(r"(\d+)$", lambda m: m.group(1).zfill(2), original_id)


def scd_cases(root: str | Path | None = None) -> list[Path]:
    """Patient image dirs (SCD00001XX) that have a manual-contour folder — the labelled subset."""
    base = _root(root)
    csv = _patient_csv(base)
    out = []
    for pid, info in csv.items():
        img_dir = base / pid
        cdir = base / "SCD_ManualContours" / _contour_dir_name(info.get("OriginalID", "")) / _IRCCI
        if img_dir.is_dir() and cdir.is_dir():
            out.append(img_dir)
    return sorted(out)


def _sax_series_dir(patient_dir: Path) -> Path | None:
    """The short-axis cine series dir (SeriesDescription 'CINESAX'); dir name is 'CINESAX_<n>'."""
    for d in sorted(patient_dir.glob("CINESAX*")):
        if d.is_dir():
            return d
    return None


def _read_contour(path: Path) -> np.ndarray:
    return np.loadtxt(path)                                   # [N,2] float (x=col, y=row) pixel coords


def _fill(pts: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize a closed polygon (x=col, y=row pixel coords) to a boolean mask on `shape` [rows,cols].
    skimage.draw (a core dep) — matplotlib.path circular-imports in spawn workers on Windows."""
    m = np.zeros(shape, dtype=bool)
    rr, cc = polygon(pts[:, 1], pts[:, 0], shape)        # polygon(row=y, col=x)
    m[rr, cc] = True
    return m


def _rasterize(endo: np.ndarray | None, epi: np.ndarray | None, shape) -> np.ndarray:
    """endo/epi polygons -> canonical mask: 2 = LV myo (epi ring minus endo), 3 = LV cav (endo interior)."""
    m = np.zeros(shape, np.uint8)
    if epi is not None:
        m[_fill(epi, shape)] = 2
    if endo is not None:
        m[_fill(endo, shape)] = 3                             # cavity overrides the myo fill it sits inside
    return m


def load_ed_es(case: str | Path, root: str | Path | None = None) -> PatientData:
    """SCD patient -> ED/ES frames (img [D,H,W], canonical mask {0,2,3}). D = contoured SAX slices; each
    slice's ED/ES chosen by endo area (larger=ED). Spacing (z,y,x): z from SliceLocation steps, y/x from px."""
    case = Path(case)
    base = _root(root)
    info = _patient_csv(base).get(case.name, {})
    cdir = base / "SCD_ManualContours" / _contour_dir_name(info.get("OriginalID", "")) / _IRCCI
    sax = _sax_series_dir(case)
    out: PatientData = {"group": info.get("Pathology"), "spacing": None}
    if sax is None or not cdir.is_dir():
        return out

    # gather per contoured instance: (sliceLoc, endo_area, img, mask, px)
    by_slice: dict[float, list[dict]] = defaultdict(list)
    for ic in sorted(cdir.glob("IM-*-icontour-manual.txt")):
        inst = ic.name.split("-")[2]                          # IM-0001-NNNN-icontour...
        dcm = next(iter(sax.glob(f"*-{inst}.dcm")), None)
        if dcm is None:
            continue
        img, (sy, sx), meta = read_image(dcm)
        loc = round(float(meta.get("slice_loc", "0")), 1)    # SliceLocation groups phases of one slice
        endo = _read_contour(ic)
        ocf = ic.with_name(ic.name.replace("icontour", "ocontour"))
        epi = _read_contour(ocf) if ocf.exists() else None
        mask = _rasterize(endo, epi, img.shape)
        by_slice[loc].append({"loc": loc, "area": int((mask == 3).sum()), "img": img, "mask": mask, "px": (sy, sx)})  # noqa: PLR2004 (3 = LV-cav label id)

    if not by_slice:
        return out
    locs = sorted(by_slice)
    ed_i, es_i, px = [], [], None
    for loc in locs:
        recs = sorted(by_slice[loc], key=lambda r: r["area"])
        es_i.append(recs[0]); ed_i.append(recs[-1]); px = recs[0]["px"]   # 1 rec -> ED==ES for that slice
    dz = float(np.median(np.diff(locs))) if len(locs) > 1 else float(px[0])
    out["spacing"] = (abs(dz), px[0], px[1])
    for tag, recs in (("ED", ed_i), ("ES", es_i)):
        out[tag] = {"img": np.stack([r["img"] for r in recs]).astype(np.float32),
                    "gt": np.stack([r["mask"] for r in recs])}
    return out


def scd_meta(case: str | Path, root: str | Path | None = None) -> dict:
    """Demographics + pathology from the patient CSV — the stratified-eval / normalization hook."""
    case = Path(case)
    info = _patient_csv(_root(root)).get(case.name, {})
    m = {"group": info.get("Pathology"), "sex": info.get("Gender"), "age": info.get("Age"),
         "vendor": "GE", "original_id": info.get("OriginalID"),        # SCD = single-vendor (GE Signa)
         "centre": "Sunnybrook Health Sciences Centre", "country": "Canada"}   # single-centre Toronto
    sax = _sax_series_dir(case)                                       # real acquisition + scanner from DICOM —
    dcm = next(iter(sax.glob("*.dcm")), None) if sax else None        # the TR/TE/flip/model the NIfTI sets lack
    if dcm is not None:
        _, _, d = read_image(dcm)
        m.update(tr_ms=d.get("tr_ms"), te_ms=d.get("te_ms"), flip_deg=d.get("flip_deg"),
                 field_T=d.get("field_T"), scanner=d.get("scanner"), institution=d.get("institution"))
    return m


class ScdAdapter(DatasetAdapter):
    name = "scd"
    label_map: dict[int, int] = {}                            # masks built canonical from contours; no remap

    def cases(self) -> list[Path]:
        return scd_cases()

    def load_ed_es(self, case: Path) -> PatientData:
        return load_ed_es(case)

    def meta(self, case: Path) -> dict:
        return scd_meta(case)
