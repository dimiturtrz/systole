"""Consolidated data store: scans + metadata as one homogeneous thing, per dataset.

Mirrors raw/ into processed/, but consolidated to a common format:

    processed/<dataset>/<paramkey>/
        data/<subject>.npz      # ed_img, ed_gt, es_img, es_gt (each [D,H,W]), spacing, group
        meta.csv                # one row per subject, common schema (read with polars)

`paramkey` = the preprocessing params (inplane resample, N4) so two recipes never collide
(`processed/acdc/inplane1p5/` vs `.../inplane1p5_n4/`). Each processed dataset is self-contained:
its meta.csv carries the full unified schema, so concatenating the meta.csv of the datasets you
ask for *is* the data cloud — no separate inventory.

The adapters (data/mri/*) stay raw->canonical readers. This is the layer that turns them into a
homogeneous, query-able store: `load(names)` ensures each is processed (builds if the folder is
missing), then returns one polars frame over all requested datasets. Splits are queries over it
(see data/splits.py).
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import BaseModel, Field

from core.config import DEFAULT_INPLANE, DEFAULT_SIZE, KNOWN_DATASETS, _VALIDATE, data_root
from core.preprocessing.n4 import N4Cfg
from core.data.static.mri.pathology import harmonize
from core.data.static.mri.registry import get_adapter
from core.preprocessing.preprocess import TARGET_INPLANE, preprocess_case


class DataCfg(BaseModel):
    """The data + the split, as criteria over the cloud (no named splits). Load `sources`; hold out
    everything matching `test_datasets` (whole dataset) OR `test_vendors` (by vendor); train/val =
    the rest, labelled. The criteria ARE the split — serialized to config.json. Defaults = the
    generalization split (ACDC centre-shift + Canon unseen-vendor)."""
    model_config = _VALIDATE
    sources: tuple[str, ...] = KNOWN_DATASETS
    # Split = criteria over the cloud. TEST = unseen vendors (Canon + GE) held out entirely, plus
    # cmrxmotion as a whole (single-vendor Siemens motion-robustness set — must be held out by
    # dataset, else it'd silently join Siemens train). VAL = ACDC (a held-out centre/protocol) — a
    # real domain-shift tuning signal that is NOT test, so aug/calibration are tuned without peeking
    # at test. TRAIN = the rest (Siemens + Philips).
    test_datasets: tuple[str, ...] = ("cmrxmotion",)
    test_vendors: tuple[str, ...] = ("Canon", "GE")
    val_datasets: tuple[str, ...] = ("acdc",)        # held-out domain for val (empty -> random val_frac)
    val_vendors: tuple[str, ...] = ()
    inplane: float = Field(DEFAULT_INPLANE, gt=0)
    n4: bool = False
    n4_params: N4Cfg = Field(default_factory=N4Cfg)   # only applied when n4=True; recorded regardless
    val_frac: float = Field(0.2, gt=0, lt=1)
    size: int = Field(DEFAULT_SIZE, ge=32)

# The real raw datasets that get consolidated, one processed/<name>/ each. NOT the same as the
# adapter registry: "canon" is a registered adapter but it's a vendor SLICE of mnms1 (a split query,
# vendor=="Canon"), never its own processed folder — else its subjects double-count in the cloud.
SOURCE_DATASETS = ["acdc", "mnm2", "mnms1", "cmrxmotion"]

# common meta schema — the unified cloud columns. `file` points at the npz in data/; `raw_path` is
# the original scan dir (the scan's "filename"); `labelled` flags usable masks (M&Ms-1 + CMRxMotion
# withhold some). `motion_grade` (CMRxMotion respiratory-motion severity 1-3) is the schema growing
# to hold a genuinely new stratification axis — null on datasets that don't carry it.
META_FIELDS = ["subject_id", "dataset", "file", "raw_path", "vendor", "scanner", "pathology",
               "pathology_raw", "field_T", "centre", "country", "age", "age_band", "sex", "height",
               "weight", "bsa", "motion_grade", "labelled"]


def param_key(inplane: float = TARGET_INPLANE, n4: bool = False, n4_params: N4Cfg | None = None) -> str:
    """Processed-cache key. n4=False -> 'inplaneXpY' (unchanged). n4=True -> encodes the N4 params
    too, so different N4 settings never collide on one cache dir."""
    key = f"inplane{str(inplane).replace('.', 'p')}"
    if n4:
        p = n4_params or N4Cfg()
        key += f"_n4-s{p.shrink}-i{'x'.join(map(str, p.iters))}-f{str(p.fwhm).replace('.', 'p')}"
    return key


def dataset_dir(dataset: str, inplane: float = TARGET_INPLANE, n4: bool = False,
                n4_params: N4Cfg | None = None) -> Path:
    return Path(data_root("processed")) / dataset / param_key(inplane, n4, n4_params)


def _bsa(height, weight):
    """Body surface area (m^2, Mosteller) from height(cm)+weight(kg). None if either missing."""
    try:
        h, w = float(height), float(weight)
        return round((h * w / 3600.0) ** 0.5, 2) if h > 0 and w > 0 else None
    except (TypeError, ValueError):
        return None


def _age_band(age):
    try:
        a = float(age)
    except (TypeError, ValueError):
        return None
    return "<45" if a < 45 else "45-60" if a < 60 else "60-75" if a < 75 else "75+"


def _norm_vendor(v):
    if not v:
        return None
    s = str(v).upper()
    for key, short in (("SIEMENS", "Siemens"), ("PHILIPS", "Philips"), ("GE", "GE"), ("CANON", "Canon")):
        if key in s:
            return short
    return str(v)


def _is_labelled(arrays: dict) -> bool:
    """Usable masks = both ED and ES present with non-empty GT (M&Ms-1 zero-fills withheld GT)."""
    ok = []
    for tag in ("ed", "es"):
        gt = arrays.get(f"{tag}_gt")
        ok.append(gt is not None and bool((gt > 0).any()))
    return all(ok)


def _meta_row(name: str, case: Path, arrays: dict, meta: dict, file: str) -> dict:
    f = meta.get("field_T")
    return {
        "subject_id": case.name, "dataset": name, "file": file, "raw_path": str(case),
        "vendor": _norm_vendor(meta.get("vendor")), "scanner": meta.get("scanner"),
        "pathology": harmonize(meta.get("group")), "pathology_raw": meta.get("group"),
        "field_T": "/".join(map(str, f)) if isinstance(f, list) else f,
        "centre": meta.get("centre"), "country": meta.get("country"),
        "age": meta.get("age"), "age_band": _age_band(meta.get("age")),
        "sex": meta.get("sex"), "height": meta.get("height"), "weight": meta.get("weight"),
        "bsa": _bsa(meta.get("height"), meta.get("weight")),
        "motion_grade": meta.get("motion_grade"), "labelled": _is_labelled(arrays),
    }


def build(name: str, inplane: float = TARGET_INPLANE, n4: bool = False,
          n4_params: N4Cfg | None = None, workers: int | None = None, rebuild: bool = False) -> Path:
    """Consolidate one dataset into processed/<name>/<paramkey>/ (data/*.npz + meta.csv).

    Process-if-missing: skips subjects already written; re-emits meta.csv each call. Parallel
    (ThreadPool — resample/N4 release the GIL). Returns the processed dir.
    """
    out = dataset_dir(name, inplane, n4, n4_params)
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    adapter = get_adapter(name)
    cases = adapter.cases()
    todo = cases if rebuild else [c for c in cases if not (data_dir / f"{c.name}.npz").exists()]

    def _one(case: Path):
        arrays = preprocess_case(case, target_inplane=inplane, loader=adapter.load_ed_es,
                                 n4=n4, n4_params=n4_params)
        npz = {k: v for k, v in arrays.items() if k != "patient"}
        np.savez_compressed(data_dir / f"{case.name}.npz", **npz)

    if todo:
        from core.obs import progress
        import logging
        log = logging.getLogger("cardioseg.store")
        workers = workers or max(1, (os.cpu_count() or 4) - 2)
        log.info("consolidating %s: %d subjects -> %s (%d threads, n4=%s)", name, len(todo), out, workers, n4)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for _ in progress(ex.map(_one, todo), f"consolidate {name}", total=len(todo)):
                pass

    # (re)write meta.csv over ALL written subjects (cheap: meta() is sidecar parsing, no image load)
    rows = []
    for case in cases:
        f = f"{case.name}.npz"
        if not (data_dir / f).exists():
            continue
        try:
            meta = adapter.meta(case)
        except Exception:
            meta = {}
        arrays = dict(np.load(data_dir / f, allow_pickle=True))
        rows.append(_meta_row(name, case, arrays, meta, f))
    def _dtype(k):
        if k in ("age", "height", "weight", "bsa"):
            return pl.Float64
        return pl.Boolean if k == "labelled" else pl.Utf8
    pl.DataFrame(rows, schema={k: _dtype(k) for k in META_FIELDS}, strict=False).write_csv(out / "meta.csv")
    return out


def load(names: list[str] | str | None = None, inplane: float = TARGET_INPLANE,
         n4: bool = False, n4_params: N4Cfg | None = None, workers: int | None = None) -> pl.DataFrame:
    """Ensure each requested dataset is consolidated, then return ONE polars frame over all of them
    (the data cloud, for these params). Adds an absolute `path` column to each npz. names=None -> all."""
    names = SOURCE_DATASETS if names is None else ([names] if isinstance(names, str) else list(names))
    frames = []
    for name in names:
        out = dataset_dir(name, inplane, n4, n4_params)
        if not (out / "meta.csv").exists():
            build(name, inplane, n4, n4_params=n4_params, workers=workers)
        # Pin `labelled` to Boolean — don't rely on polars schema inference (newer polars reads the
        # "true"/"false" column as String, breaking the `pl.col('labelled')` filter cross-platform).
        df = pl.read_csv(out / "meta.csv", infer_schema_length=10000,
                         schema_overrides={"labelled": pl.Boolean})
        df = df.with_columns((pl.lit(str(out / "data")) + "/" + pl.col("file")).alias("path"))
        frames.append(df)
    cloud = pl.concat(frames, how="vertical_relaxed")
    # continent is DERIVED from country (SSOT in data/geo) — queryable column, never hand-stored.
    from core.data.static.geo import COUNTRY_CONTINENT
    return cloud.with_columns(
        pl.col("country").replace_strict(COUNTRY_CONTINENT, default=None).alias("continent"))


def load_arrays(path: str | Path) -> dict:
    """Load one consolidated subject npz -> dict (ed_img/ed_gt/es_img/es_gt/spacing/group)."""
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    import argparse
    from core.obs import setup
    setup()
    ap = argparse.ArgumentParser(description="consolidate datasets into processed/<ds>/<paramkey>/")
    ap.add_argument("--names", nargs="*", default=None, help="datasets (default: all)")
    ap.add_argument("--inplane", type=float, default=TARGET_INPLANE)
    ap.add_argument("--n4", action="store_true")
    args = ap.parse_args()
    df = load(args.names, inplane=args.inplane, n4=args.n4)
    print(f"\n=== data cloud: {len(df)} subjects ===")
    print(df.group_by("dataset").agg(pl.len().alias("n"),
          pl.col("labelled").sum().alias("labelled")).sort("dataset"))
    print(df.group_by("vendor").agg(pl.len().alias("n")).sort("n", descending=True))
