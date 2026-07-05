"""Kaggle Second Annual Data Science Bowl (2015) reader — DICOM cine + EF TARGET (regression). This is
NOT a seg adapter: the dataset has NO masks; ground truth = EDV/ESV/EF volumes only. So it doesn't join
the seg `DatasetAdapter` registry — it's a lighter, separate reader.

Use: EF-at-scale eval / a large real-DICOM testbed. Run our seg→EF pipeline on the SAX cine, compare the
predicted EF against the Kaggle ground-truth volumes.

Layout: `<split>/<split>/<case>/study/{sax_NN, 2ch_NN, 4ch_NN}/IM-*.dcm`. SAX cine = the `sax_*` series
(each a slice location, ~30 cardiac phases). Targets: `train.csv`/`validate.csv` (`Id, Systole=ESV,
Diastole=EDV` mL); test = `solution.csv` (`Id=<case>_Diastole/_Systole, Volume, Usage`). ~1140 cases,
NIH + Children's National, normal + diseased.
"""
from __future__ import annotations

import csv
from pathlib import Path


# Provenance is dataset knowledge -> it lives in the adapter (the PROCESS layer), not the raw data.
# DSB 2015 was compiled by NIH + Children's National, scanned in the Washington DC area.
CENTRE = "Children's National Medical Center / NIH"
COUNTRY = "USA"


def _base(root: str | Path | None = None) -> Path:
    from core.config import data_root
    return Path(root) if root else Path(data_root("raw")) / "kaggle_dsb2015"


def _split_dir(split: str, root: str | Path | None = None) -> Path:
    return _base(root) / split / split                       # nested <split>/<split>/


def _ef(edv: float, esv: float) -> dict:
    return {"edv": edv, "esv": esv, "ef": round(100.0 * (edv - esv) / edv, 2) if edv else None}


def kaggle_ef(split: str, root: str | Path | None = None) -> dict[str, dict]:
    """{case_id: {edv, esv, ef}} in mL / %. train/validate from `<split>.csv` (Diastole=EDV, Systole=ESV);
    test from `solution.csv` (`<case>_Diastole` / `<case>_Systole` rows)."""
    base = _base(root)
    out: dict[str, dict] = {}
    if split in ("train", "validate"):
        with (base / f"{split}.csv").open(newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                out[r["Id"]] = _ef(float(r["Diastole"]), float(r["Systole"]))
    else:                                                     # test — solution.csv, two rows per case
        vols: dict[str, dict] = {}
        with (base / "solution.csv").open(newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                cid, phase = r["Id"].rsplit("_", 1)
                vols.setdefault(cid, {})[phase] = float(r["Volume"])
        out = {cid: _ef(v["Diastole"], v["Systole"]) for cid, v in vols.items()
               if "Diastole" in v and "Systole" in v}
    return out


def kaggle_cases(split: str, root: str | Path | None = None) -> list[Path]:
    """Case dirs for a split, numerically sorted."""
    d = _split_dir(split, root)
    return sorted((p for p in d.glob("*") if p.is_dir()), key=lambda p: int(p.name))


def load_sax(case: str | Path) -> list[tuple]:
    """SAX cine of one case: list of (volume [phases,H,W], spacing (z,y,x), meta) — one entry per `sax_*`
    series (slice location), sorted apex→base by SliceLocation. Each series is the ~30-phase cine loop at
    that slice. Uses `dicom.read_series`; broken/odd series skipped."""
    from core.data.static.mri.dicom import read_series
    rows = []
    for sd in sorted((Path(case) / "study").glob("sax_*")):
        try:
            vol, sp, meta = read_series(sd)
            rows.append((vol, sp, meta, float(meta.get("slice_loc") or 0)))
        except Exception:
            continue
    rows.sort(key=lambda t: t[3])
    return [(v, s, m) for v, s, m, _ in rows]


def kaggle_meta(case: str | Path, ef_targets: dict | None = None) -> dict:
    """Per-case metadata: location constants (adapter = process layer) + real vendor/scanner/acquisition
    from a sample SAX DICOM + the EF target. NB Kaggle is MIXED sequences (segmented cine/GRE) -> its TR
    is NOT the ~3ms per-frame bSSFP TR; captured as-recorded, but the bSSFP acquisition reference filters
    it (store.fit_acquisition_reference)."""
    from core.data.static.mri.dicom import read_image
    from core.data.static.store import _norm_vendor
    case = Path(case)
    m = {"centre": CENTRE, "country": COUNTRY}
    sd = next(iter((case / "study").glob("sax_*")), None)
    dcm = next(iter(sd.glob("*.dcm")), None) if sd else None
    if dcm is not None:
        _, _, d = read_image(dcm)
        m.update(vendor=_norm_vendor(d.get("vendor")), scanner=d.get("scanner"), field_T=d.get("field_T"),
                 tr_ms=d.get("tr_ms"), te_ms=d.get("te_ms"), flip_deg=d.get("flip_deg"),
                 institution=d.get("institution"))
    if ef_targets and (t := ef_targets.get(case.name)):
        m.update(t)                                          # edv, esv, ef
    return m


def build_kaggle_meta(split: str, root: str | Path | None = None) -> Path:
    """Extract Kaggle's per-case metadata to processed/kaggle/<split>/meta.csv (location + vendor/scanner/
    acquisition + EF target). NO image npz — Kaggle is EF-at-scale, images stay pristine in raw and are
    read on demand (load_sax). This meta.csv makes Kaggle's location/vendor/acquisition part of the data
    cloud (fit_acquisition_reference reads the tr_ms column, bSSFP-filtered)."""
    import polars as pl
    from core.config import data_root
    ef = kaggle_ef(split, root)
    rows = [{"subject_id": c.name, "dataset": "kaggle", **kaggle_meta(c, ef)} for c in kaggle_cases(split, root)]
    out = Path(data_root("processed")) / "kaggle" / split
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, strict=False).write_csv(out / "meta.csv")
    return out / "meta.csv"
