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
