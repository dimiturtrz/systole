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
    """The data + the split. Load `sources`; TEST = frozen manifests (`test_manifests`, preferred —
    comparable across store growth) or live criteria (`test_datasets`/`test_vendors`); train/val =
    the labelled rest. Serialized to config.json (the run self-documents its split). Named presets
    live in splits.SPLITS (`--split xvendor`); defaults = the generalization split (ACDC centre-shift
    VAL + Canon/GE unseen-vendor + cmrxmotion TEST)."""
    model_config = _VALIDATE
    sources: tuple[str, ...] = KNOWN_DATASETS
    # Split = criteria over the cloud. TEST = unseen vendors (Canon + GE) held out entirely, plus
    # cmrxmotion as a whole (single-vendor Siemens motion-robustness set — must be held out by
    # dataset, else it'd silently join Siemens train). VAL = ACDC (a held-out centre/protocol) — a
    # real domain-shift tuning signal that is NOT test, so aug/calibration are tuned without peeking
    # at test. TRAIN = the rest (Siemens + Philips).
    # TEST source, two mutually-exclusive modes. test_manifests (preferred) = FROZEN manifests by name
    # (core.data.static.manifest): the test set is pinned + comparable across store growth. When empty,
    # fall back to LIVE criteria (test_datasets / test_vendors), recomputed over the current cloud — the
    # original behaviour, kept for ad-hoc runs. Named splits (splits.SPLITS) set test_manifests.
    # NEW-STYLE split: a coded-filter family@version (core.data.splits). When set, it OWNS the
    # train/val/test partition (via core.data.split.resolve) and the criteria below are ignored — the
    # split is code, not criteria. Recorded to config.json for lineage. Empty -> old criteria path.
    split: str = ""
    test_manifests: tuple[str, ...] = ()
    test_datasets: tuple[str, ...] = ("cmrxmotion",)
    test_vendors: tuple[str, ...] = ("Canon", "GE")
    val_datasets: tuple[str, ...] = ("acdc",)        # held-out domain for val (empty -> random val_frac)
    val_vendors: tuple[str, ...] = ()
    train_vendors: tuple[str, ...] = ()              # if set: restrict TRAIN to these vendors only (the
    #                                                scarce/single-vendor regime; val/test intact — bd 5r7n)
    inplane: float = Field(DEFAULT_INPLANE, gt=0)
    n4: bool = False
    n4_params: N4Cfg = Field(default_factory=N4Cfg)   # only applied when n4=True; recorded regardless
    nyul: bool = False                                # Nyúl histogram standardization (harmonization,
    #                                                qfz): map the intensity distribution to a cohort-fit
    #                                                standard before z-score. The standard = reference data.
    anatomy_pool: str = ""                            # if set: path to a synthetic-anatomy label-map pool
    #                                                (core.data.dynamic.anatomy.build_pool from Rodero SSM
    #                                                meshes). TRAIN masks come from it (synth anatomy, zero
    #                                                real data); val/test stay REAL held-out. (bd 1vl/bwp)
    anatomy_mode: str = "replace"                     # how the anatomy_pool enters training: "replace" =
    #                                                synth anatomy ONLY (zero real, the generation probe);
    #                                                "mix" = REAL train + synth anatomy UNION, synth rows
    #                                                painted per-batch, real rows kept (augmentation, bd pwih).
    norm: str = "zscore"                              # intensity normalization: 'zscore' (default) or
    #                                                'blood' = two-point affine (air->0, blood->1),
    #                                                composition-robust harmonization (bd h8k). ORACLE
    #                                                (uses GT blood) — cache is separate (_blood suffix).
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
               "pathology_raw", "field_T", "tr_ms", "te_ms", "flip_deg", "centre", "country", "region",
               "institution", "age", "age_band", "sex", "height", "weight", "bsa", "motion_grade",
               "labelled"]

# country -> region: the coarse population-shift axis (genetic Europe/Asia, lifestyle Europe/America)
# at the granularity our thin per-country data supports. DERIVED from real country (never fabricated) —
# null country stays null region; extend the map as new countries land.
_REGION = {"France": "Europe", "Spain": "Europe", "Germany": "Europe", "Italy": "Europe", "UK": "Europe",
           "China": "Asia", "Japan": "Asia", "Korea": "Asia",
           "USA": "North America", "Canada": "North America"}


def _region_of(country):
    return _REGION.get(country) if country else None


def param_key(inplane: float = TARGET_INPLANE, n4: bool = False, n4_params: N4Cfg | None = None,
              nyul: bool = False, norm: str = "zscore") -> str:
    """Processed-cache key. n4=False -> 'inplaneXpY' (unchanged). n4=True -> encodes the N4 params
    too, so different N4 settings never collide on one cache dir. nyul -> '_nyul' suffix (harmonized
    cache is separate). norm='blood' -> '_blood' suffix (blood-anchored normalization, bd h8k)."""
    key = f"inplane{str(inplane).replace('.', 'p')}"
    if n4:
        p = n4_params or N4Cfg()
        key += f"_n4-s{p.shrink}-i{'x'.join(map(str, p.iters))}-f{str(p.fwhm).replace('.', 'p')}"
    if nyul:
        key += "_nyul"
    if norm != "zscore":
        key += f"_{norm}"
    return key


def dataset_dir(dataset: str, inplane: float = TARGET_INPLANE, n4: bool = False,
                n4_params: N4Cfg | None = None, nyul: bool = False, norm: str = "zscore") -> Path:
    return Path(data_root("processed")) / dataset / param_key(inplane, n4, n4_params, nyul, norm)


def _nyul_ref_path() -> Path:
    from core.data.static.reference import reference_dir
    return reference_dir() / "nyul.yaml"


def fit_nyul_standard(names: list[str] | None = None, inplane: float = TARGET_INPLANE,
                      per_dataset: int = 40) -> "np.ndarray":
    """Fit the Nyúl standard landmark scale from the cohort (resampled, pre-z-score images) and write
    it to reference/nyul.yaml with provenance. Samples up to per_dataset subjects/dataset (landmarks are
    stable). The standard is a normalization axis -> reference data, fit once, applied in preprocess."""
    from core.preprocessing.preprocess import resample_inplane
    from core.preprocessing.nyul import image_landmarks, fit_standard, LANDMARKS
    from omegaconf import OmegaConf
    names = SOURCE_DATASETS if names is None else names
    rows = []
    for name in names:
        adapter = get_adapter(name)
        for case in adapter.cases()[:per_dataset]:
            d = adapter.load_ed_es(case)
            if "ED" not in d:
                continue
            img, _ = resample_inplane(d["ED"]["img"], d["spacing"], inplane, is_mask=False)
            rows.append(image_landmarks(img))
    std = fit_standard(np.stack(rows))
    p = _nyul_ref_path(); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# Nyúl standard landmark scale (harmonization qfz) — fit by store.fit_nyul_standard\n"
                 + OmegaConf.to_yaml(OmegaConf.create({"nyul": {"standard": {
                     "value": [round(float(v), 5) for v in std], "landmarks": list(LANDMARKS),
                     "source": "computed", "based_on": f"resampled ED, {names}, per<={per_dataset}, n={len(rows)}",
                     "extracted_by": "computed", "verified": True}}})))
    return std


def load_nyul_standard() -> "np.ndarray | None":
    """The fitted Nyúl standard from reference/nyul.yaml, or None if absent (then nyul can't run)."""
    from core.data.static.reference import Reference
    v = Reference().get("nyul", "standard")
    return np.asarray(v, dtype=np.float64) if v is not None else None


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


def fit_acquisition_reference(root: str | Path | None = None) -> dict:
    """Aggregate REAL DICOM acquisition (TR/TE/flip) from the built stores into per-(vendor,field)
    reference values -> reference/acquisition.yaml (verified: DICOM-measured). This is the DAG COMPOSE:
    `mri_physics.acquisition_for` OVERRIDES its physics derivation with these where present, and the
    derivation stays the backbone for the null majority — real refines, derived covers. Only rows with
    real acquisition contribute (DICOM datasets, e.g. SCD=GE); NIfTI datasets have nulls and are skipped,
    so the whole domain-randomization sweep survives for everything we lack real values for."""
    import polars as pl
    from omegaconf import OmegaConf
    from core.data.static.reference import reference_dir
    base = Path(data_root("processed"))
    metas = [pl.read_csv(str(f), infer_schema_length=0) for f in base.glob("*/*/meta.csv")]
    if not metas:
        return {}
    df = pl.concat(metas, how="diagonal")
    if "tr_ms" not in df.columns:
        return {}
    # bSSFP-TR sanity gate: cine bSSFP TR is ~2.7-6 ms (mriquestions). Mixed-sequence DICOM (Kaggle:
    # segmented cine / GRE, TR ~39 ms) must NOT feed the bSSFP-derivation override — filter it out.
    tr = pl.col("tr_ms").cast(pl.Float64, strict=False)
    real = df.filter(pl.col("tr_ms").is_not_null() & pl.col("vendor").is_not_null()
                     & (tr >= 2.0) & (tr <= 6.0))
    # normalize field to a float so "1.5" and "1.500000" (DICOM formatting) are ONE group, not two
    real = real.with_columns(pl.col("field_T").cast(pl.Float64, strict=False).round(1).alias("_field"))
    acq: dict[str, dict] = {}
    for (vendor, field), g in real.group_by(["vendor", "_field"]):
        med = lambda c: round(float(g[c].cast(pl.Float64).median()), 3)
        based = f"{g.height} DICOM subjects @{field}T"
        leaf = lambda v: {"value": v, "source": "DICOM-measured", "based_on": based,
                          "extracted_by": "computed", "verified": True}      # per-leaf provenance schema
        e = acq.setdefault(str(vendor), {})
        e["tr_ms"], e["te_ms"] = leaf(med("tr_ms")), leaf(med("te_ms"))
        near15 = abs(float(field) - 1.5) < 0.6 if field else True
        e["flip_deg_1p5t" if near15 else "flip_deg_3t"] = leaf(med("flip_deg"))
    out = reference_dir() / "acquisition.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# Real per-(vendor,field) acquisition, DICOM-mined by store.fit_acquisition_reference.\n"
                   "# acquisition_for OVERRIDES the physics derivation with these where present (DAG compose).\n"
                   + OmegaConf.to_yaml(OmegaConf.create({"acquisition": acq})))
    return acq


def _meta_row(name: str, case: Path, arrays: dict, meta: dict, file: str) -> dict:
    f = meta.get("field_T")
    return {
        "subject_id": case.name, "dataset": name, "file": file, "raw_path": str(case),
        "vendor": _norm_vendor(meta.get("vendor")), "scanner": meta.get("scanner"),
        "pathology": harmonize(meta.get("group")), "pathology_raw": meta.get("group"),
        "field_T": "/".join(map(str, f)) if isinstance(f, list) else f,
        # real per-image ACQUISITION — only DICOM carries these (TR/TE/flip); NIfTI datasets stripped the
        # headers so they stay null. The ground truth our synth/normalization thread otherwise *derives*.
        "tr_ms": meta.get("tr_ms"), "te_ms": meta.get("te_ms"), "flip_deg": meta.get("flip_deg"),
        "centre": meta.get("centre"), "country": meta.get("country"),
        "region": _region_of(meta.get("country")), "institution": meta.get("institution"),
        "age": meta.get("age"), "age_band": _age_band(meta.get("age")),
        "sex": meta.get("sex"), "height": meta.get("height"), "weight": meta.get("weight"),
        "bsa": _bsa(meta.get("height"), meta.get("weight")),
        "motion_grade": meta.get("motion_grade"), "labelled": _is_labelled(arrays),
    }


def _write_meta(name: str, adapter, data_dir: Path, out: Path) -> Path:
    """(re)write out/meta.csv over every written subject via adapter.meta() (sidecar parse, no image
    reload) — the ONE place the meta schema is materialized. Shared by build() and migrate_meta()."""
    rows = []
    for case in adapter.cases():
        f = f"{case.name}.npz"
        if not (data_dir / f).exists():
            continue
        try:
            meta = adapter.meta(case)
        except Exception:
            meta = {}
        arrays = dict(np.load(data_dir / f, allow_pickle=True))
        rows.append(_meta_row(name, case, arrays, meta, f))
    _dt = lambda k: pl.Float64 if k in ("age", "height", "weight", "bsa") else (
        pl.Boolean if k == "labelled" else pl.Utf8)
    pl.DataFrame(rows, schema={k: _dt(k) for k in META_FIELDS}, strict=False).write_csv(out / "meta.csv")
    return out / "meta.csv"


def migrate_meta(names: list[str] | None = None) -> list[Path]:
    """MIGRATION: re-emit meta.csv for already-built stores with the CURRENT META_FIELDS + adapter.meta()
    — NO image reload (sidecar parse only). Run after adding a metadata column or fixing an adapter's
    meta() (e.g. SCD now carrying centre/country/scanner). Every processed/<name>/<paramkey>/ of a
    REGISTERED adapter is refreshed; unregistered dirs (anatomy pools etc.) skipped. `names` filters."""
    from core.data.static.mri.registry import get_adapter
    base = Path(data_root("processed"))
    out: list[Path] = []
    for meta_csv in sorted(base.glob("*/*/meta.csv")):
        param_dir = meta_csv.parent
        name = param_dir.parent.name
        if names and name not in names:
            continue
        try:
            adapter = get_adapter(name)
        except KeyError:
            continue                                         # not a registered dataset (e.g. mrxcat pools)
        out.append(_write_meta(name, adapter, param_dir / "data", param_dir))
    return out


def build(name: str, inplane: float = TARGET_INPLANE, n4: bool = False,
          n4_params: N4Cfg | None = None, workers: int | None = None, rebuild: bool = False,
          nyul: bool = False, nyul_standard=None, norm: str = "zscore") -> Path:
    """Consolidate one dataset into processed/<name>/<paramkey>/ (data/*.npz + meta.csv).

    Process-if-missing: skips subjects already written; re-emits meta.csv each call. Parallel
    (ThreadPool — resample/N4 release the GIL). `nyul`+`nyul_standard` apply Nyúl harmonization
    (qfz) to a separate _nyul cache. Returns the processed dir.
    """
    out = dataset_dir(name, inplane, n4, n4_params, nyul, norm)
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    adapter = get_adapter(name)
    cases = adapter.cases()
    todo = cases if rebuild else [c for c in cases if not (data_dir / f"{c.name}.npz").exists()]

    def _one(case: Path):
        arrays = preprocess_case(case, target_inplane=inplane, loader=adapter.load_ed_es,
                                 n4=n4, n4_params=n4_params,
                                 nyul_standard=nyul_standard if nyul else None, norm=norm)
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

    _write_meta(name, adapter, data_dir, out)                # (re)emit meta.csv (sidecar parse, no reload)
    return out


def load(names: list[str] | str | None = None, inplane: float = TARGET_INPLANE,
         n4: bool = False, n4_params: N4Cfg | None = None, workers: int | None = None,
         nyul: bool = False, norm: str = "zscore") -> pl.DataFrame:
    """Ensure each requested dataset is consolidated, then return ONE polars frame over all of them
    (the data cloud, for these params). Adds an absolute `path` column to each npz. names=None -> all.
    nyul=True harmonizes to the cohort standard (reference/nyul.yaml; fit it first with
    `python -m core.data.static.store --fit-nyul`)."""
    names = SOURCE_DATASETS if names is None else ([names] if isinstance(names, str) else list(names))
    std = load_nyul_standard() if nyul else None
    if nyul and std is None:
        raise RuntimeError("nyul=True but no reference/nyul.yaml — fit it first: "
                           "python -m core.data.static.store --fit-nyul")
    frames = []
    for name in names:
        out = dataset_dir(name, inplane, n4, n4_params, nyul, norm)
        if not (out / "meta.csv").exists():
            build(name, inplane, n4, n4_params=n4_params, workers=workers, nyul=nyul,
                  nyul_standard=std, norm=norm)
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
    ap.add_argument("--nyul", action="store_true", help="harmonize to the Nyúl standard (fit it first)")
    ap.add_argument("--fit-nyul", action="store_true", dest="fit_nyul",
                    help="fit the Nyúl standard from the cohort -> reference/nyul.yaml, then exit")
    ap.add_argument("--migrate-meta", action="store_true", dest="migrate_meta",
                    help="re-emit meta.csv for built stores with the current schema/adapter.meta() "
                         "(no image reload) — run after adding metadata fields, then exit")
    ap.add_argument("--fit-acquisition", action="store_true", dest="fit_acq",
                    help="mine real per-(vendor,field) TR/TE/flip from built DICOM stores -> "
                         "reference/acquisition.yaml (acquisition_for then overrides the derivation), then exit")
    args = ap.parse_args()
    if args.fit_nyul:
        std = fit_nyul_standard(args.names, inplane=args.inplane)
        print(f"fit Nyúl standard -> {_nyul_ref_path()}\n  {[round(float(v), 3) for v in std]}")
        raise SystemExit
    if args.fit_acq:
        acq = fit_acquisition_reference()
        print(f"fit acquisition reference -> reference/acquisition.yaml\n  {list(acq)}")
        raise SystemExit
    if args.migrate_meta:
        paths = migrate_meta(args.names)
        print(f"migrated meta.csv (current schema, no image reload) for {len(paths)} store(s):")
        for p in paths:
            print(f"  {p}")
        raise SystemExit
    df = load(args.names, inplane=args.inplane, n4=args.n4, nyul=args.nyul)
    print(f"\n=== data cloud: {len(df)} subjects ===")
    print(df.group_by("dataset").agg(pl.len().alias("n"),
          pl.col("labelled").sum().alias("labelled")).sort("dataset"))
    print(df.group_by("vendor").agg(pl.len().alias("n")).sort("n", descending=True))
