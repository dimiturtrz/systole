"""The store's READ + metadata surface — preprocessing-free, so the many read-only consumers
(`load_arrays`, `DataCfg`, the meta helpers) never drag the N4/Nyúl/resample pipeline. The build/
normalize engine that DOES need preprocessing lives in the sibling `build`/`normalize` modules; this
module is the light half everything else imports.

Cache layout (materialized by `build`):

    processed/<dataset>/<paramkey>/
        data/<subject>.npz      # ed_img, ed_gt, es_img, es_gt (each [D,H,W]), spacing, group
        meta.csv                # one row per subject, common schema (read with polars)

`paramkey` (`param_key`) encodes the preprocessing recipe (inplane resample, N4, Nyúl, norm) so two
recipes never collide. Each processed dataset is self-contained: concatenating the meta.csv of the
datasets you ask for *is* the data cloud — no separate inventory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import OmegaConf
from pydantic import BaseModel, Field

from core.config import _VALIDATE, DEFAULT_INPLANE, DEFAULT_SIZE, KNOWN_DATASETS, Config
from core.data.static.mri.pathology import Pathology
from core.data.static.mri.registry import AdapterRegistry
from core.data.static.reference import Reference
from core.preprocessing.n4 import N4Cfg


class Recipe(BaseModel):
    """The preprocessing recipe that addresses a processed-cache dir (the paramkey): inplane resample, N4
    bias field, Nyúl harmonization, intensity norm. One value object threaded through Store/Build/
    Normalizer instead of five loose params travelling together (Fowler Introduce Parameter Object, bd
    cardiac-seg-h7vy.4.1). The fitted Nyúl STANDARD is a separate artifact (data, not a knob) — it stays
    out of the recipe and rides alongside as `nyul_standard`."""

    inplane: float = Field(DEFAULT_INPLANE, gt=0)
    n4: bool = False
    n4_params: N4Cfg = Field(default_factory=N4Cfg)
    nyul: bool = False
    norm: str = "zscore"


class DataCfg(BaseModel):
    """The data + the LEGACY criteria split. Prefer a coded split family (`split`, core.data.ingest.splits).
    Legacy path: load `sources`; TEST = live criteria (`test_datasets`/`test_vendors`); train/val =
    the labelled rest. Serialized to config.json (the run self-documents its split). Prefer a coded
    split family (`split` field, core.data.ingest.splits); these criteria defaults = the generalization
    split (ACDC centre-shift VAL + Canon/GE unseen-vendor + cmrxmotion TEST)."""
    model_config = _VALIDATE
    sources: tuple[str, ...] = KNOWN_DATASETS
    # Split = criteria over the cloud. TEST = unseen vendors (Canon + GE) held out entirely, plus
    # cmrxmotion as a whole (single-vendor Siemens motion-robustness set — must be held out by
    # dataset, else it'd silently join Siemens train). VAL = ACDC (a held-out centre/protocol) — a
    # real domain-shift tuning signal that is NOT test, so aug/calibration are tuned without peeking
    # at test. TRAIN = the rest (Siemens + Philips).
    # NEW-STYLE split: a coded-filter family@version (core.data.ingest.splits). When set, it OWNS the
    # train/val/test partition (via core.data.ingest.split.resolve) and the criteria below are ignored — the
    # split is code, not criteria. Recorded to config.json for lineage. Empty -> legacy criteria path.
    split: str = ""
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

    @property
    def recipe(self) -> Recipe:
        """The preprocessing recipe this cfg selects — the bridge from the flat serialized fields (kept
        for config.json back-compat) to the Recipe value object Store/Build/Normalizer consume."""
        return Recipe(inplane=self.inplane, n4=self.n4, n4_params=self.n4_params,
                      nyul=self.nyul, norm=self.norm)


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

# age-band cut points (years) for demographic stratification
_AGE_45, _AGE_60, _AGE_75 = 45, 60, 75

# cine bSSFP TR sanity gate (ms): reject mixed-sequence DICOM (GRE/segmented cine, TR ~39 ms)
_MIN_BSSFP_TR_MS, _MAX_BSSFP_TR_MS = 2.0, 6.0
_FIELD_1P5T = 1.5           # tesla; nominal low-field
_FIELD_1P5T_TOL = 0.6       # |field - 1.5T| below this -> treat as the 1.5T flip bucket


class Store:
    """The store's cache-addressing read surface, bound to ONE preprocessing recipe (inplane resample,
    N4, Nyúl, norm): construct once with the recipe, then `param_key()` / `dataset_dir(dataset)` address
    the cache for any dataset under it. The recipe is the fixed session — it's exactly what a paramkey
    encodes — while the dataset name is the per-call data. `load_arrays` is recipe-free (a pure npz read),
    so it stays a staticmethod."""

    def __init__(self, recipe: Recipe | None = None):
        self.recipe = recipe or Recipe()

    def param_key(self) -> str:
        """Processed-cache key for this recipe. n4=False -> 'inplaneXpY' (unchanged). n4=True -> encodes
        the N4 params too, so different N4 settings never collide on one cache dir. nyul -> '_nyul' suffix
        (harmonized cache is separate). norm='blood' -> '_blood' suffix (blood-anchored norm, bd h8k)."""
        r = self.recipe
        key = f"inplane{str(r.inplane).replace('.', 'p')}"
        if r.n4:
            p = r.n4_params or N4Cfg()
            key += f"_n4-s{p.shrink}-i{'x'.join(map(str, p.iters))}-f{str(p.fwhm).replace('.', 'p')}"
        if r.nyul:
            key += "_nyul"
        if r.norm != "zscore":
            key += f"_{r.norm}"
        return key

    def dataset_dir(self, dataset: str) -> Path:
        return Path(Config.data_root("processed")) / dataset / self.param_key()

    @staticmethod
    def load_arrays(path: str | Path) -> dict:
        """Load one consolidated subject npz -> dict (ed_img/ed_gt/es_img/es_gt/spacing/group)."""
        z = np.load(path, allow_pickle=True)
        return {k: (z[k].item() if k == "group" else z[k]) for k in z.files}   # group -> plain py scalar (0-d npz)


class MetaBuilder:
    """Materializes the ONE unified meta.csv schema for a dataset — the read-side counterpart to the
    npz build. Holds the (dataset name, adapter) it writes for; `write` (re)emits meta.csv over the
    written npz via `adapter.meta()` (sidecar parse, no image reload). `migrate` refreshes every built
    store to the current schema (run after adding a column / fixing an adapter's meta())."""

    def __init__(self, name: str, adapter):
        self.name, self.adapter = name, adapter

    @staticmethod
    def _region_of(country):
        return _REGION.get(country) if country else None

    @staticmethod
    def _bsa(height, weight):
        """Body surface area (m^2, Mosteller) from height(cm)+weight(kg). None if either missing."""
        try:
            h, w = float(height), float(weight)
            return round((h * w / 3600.0) ** 0.5, 2) if h > 0 and w > 0 else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _age_band(age):
        try:
            a = float(age)
        except (TypeError, ValueError):
            return None
        return ("<45" if a < _AGE_45 else "45-60" if a < _AGE_60
                else "60-75" if a < _AGE_75 else "75+")

    @staticmethod
    def _norm_vendor(v):
        if not v:
            return None
        s = str(v).upper()
        for key, short in (("SIEMENS", "Siemens"), ("PHILIPS", "Philips"), ("GE", "GE"), ("CANON", "Canon")):
            if key in s:
                return short
        return str(v)

    @staticmethod
    def _is_labelled(arrays: dict) -> bool:
        """Usable masks = both ED and ES present with non-empty GT (M&Ms-1 zero-fills withheld GT)."""
        ok = []
        for tag in ("ed", "es"):
            gt = arrays.get(f"{tag}_gt")
            ok.append(gt is not None and bool((gt > 0).any()))
        return all(ok)

    def _row(self, case: Path, arrays: dict, meta: dict, file: str) -> dict:
        f = meta.get("field_T")
        return {
            "subject_id": case.name, "dataset": self.name, "file": file, "raw_path": str(case),
            "vendor": self._norm_vendor(meta.get("vendor")), "scanner": meta.get("scanner"),
            "pathology": Pathology.harmonize(meta.get("group")), "pathology_raw": meta.get("group"),
            "field_T": "/".join(map(str, f)) if isinstance(f, list) else f,
            # real per-image ACQUISITION — only DICOM carries these (TR/TE/flip); NIfTI datasets stripped the
            # headers so they stay null. The ground truth our synth/normalization thread otherwise *derives*.
            "tr_ms": meta.get("tr_ms"), "te_ms": meta.get("te_ms"), "flip_deg": meta.get("flip_deg"),
            "centre": meta.get("centre"), "country": meta.get("country"),
            "region": self._region_of(meta.get("country")), "institution": meta.get("institution"),
            "age": meta.get("age"), "age_band": self._age_band(meta.get("age")),
            "sex": meta.get("sex"), "height": meta.get("height"), "weight": meta.get("weight"),
            "bsa": self._bsa(meta.get("height"), meta.get("weight")),
            "motion_grade": meta.get("motion_grade"), "labelled": self._is_labelled(arrays),
        }

    def write(self, data_dir: Path, out: Path) -> Path:
        """(re)write out/meta.csv over every written subject — the ONE place the schema is materialized."""
        rows = []
        for case in self.adapter.cases():
            f = f"{case.name}.npz"
            if not (data_dir / f).exists():
                continue
            try:
                meta = self.adapter.meta(case)
            except (KeyError, ValueError, OSError, AttributeError):   # missing/odd metadata field -> no meta, not fatal
                meta = {}
            arrays = dict(np.load(data_dir / f, allow_pickle=True))
            rows.append(self._row(case, arrays, meta, f))
        def _dt(k):
            if k in ("age", "height", "weight", "bsa"):
                return pl.Float64
            return pl.Boolean if k == "labelled" else pl.Utf8
        pl.DataFrame(rows, schema={k: _dt(k) for k in META_FIELDS}, strict=False).write_csv(out / "meta.csv")
        return out / "meta.csv"

    @staticmethod
    def migrate(names: list[str] | None = None) -> list[Path]:
        """Re-emit meta.csv for already-built stores with the CURRENT META_FIELDS + adapter.meta() — NO
        image reload (sidecar parse only). Every processed/<name>/<paramkey>/ of a REGISTERED adapter is
        refreshed; unregistered dirs (anatomy pools etc.) skipped. `names` filters."""
        base = Path(Config.data_root("processed"))
        out: list[Path] = []
        for meta_csv in sorted(base.glob("*/*/meta.csv")):
            param_dir = meta_csv.parent
            name = param_dir.parent.name
            if names and name not in names:
                continue
            try:
                adapter = AdapterRegistry.get_adapter(name)
            except KeyError:
                continue                                         # not a registered dataset (e.g. mrxcat pools)
            out.append(MetaBuilder(name, adapter).write(param_dir / "data", param_dir))
        return out


class AcqReference:
    """Mines REAL per-(vendor,field) acquisition (TR/TE/flip) from the built stores' meta.csv into
    reference/acquisition.yaml — the DAG compose where `mri_physics.acquisition_for` OVERRIDES its
    physics derivation with measured values where present, derivation covering the null majority."""

    @staticmethod
    def from_frame(df: "pl.DataFrame") -> dict:
        """PURE core: a cloud meta frame -> per-(vendor,field) TR/TE/flip medians with provenance leaves.
        bSSFP-TR sanity gate rejects mixed-sequence DICOM (TR outside 2-6 ms) and null vendor/TR; field
        normalized to 1 dp so '1.5'/'1.500000' fold into ONE group. {} if no tr_ms column / no rows survive."""
        if "tr_ms" not in df.columns:
            return {}
        # bSSFP-TR sanity gate: cine bSSFP TR is ~2.7-6 ms (mriquestions). Mixed-sequence DICOM (Kaggle:
        # segmented cine / GRE, TR ~39 ms) must NOT feed the bSSFP-derivation override — filter it out.
        tr = pl.col("tr_ms").cast(pl.Float64, strict=False)
        real = df.filter(pl.col("tr_ms").is_not_null() & pl.col("vendor").is_not_null()
                         & (tr >= _MIN_BSSFP_TR_MS) & (tr <= _MAX_BSSFP_TR_MS))
        # normalize field to a float so "1.5" and "1.500000" (DICOM formatting) are ONE group, not two
        real = real.with_columns(pl.col("field_T").cast(pl.Float64, strict=False).round(1).alias("_field"))
        acq: dict[str, dict] = {}
        for (vendor, field), g in real.group_by(["vendor", "_field"]):
            def med(col, frame=g):
                return round(float(frame[col].cast(pl.Float64).median()), 3)
            based = f"{g.height} DICOM subjects @{field}T"

            def leaf(value, provenance=based):
                return {"value": value, "source": "DICOM-measured", "based_on": provenance,
                        "extracted_by": "computed", "verified": True}      # per-leaf provenance schema
            e = acq.setdefault(str(vendor), {})
            e["tr_ms"], e["te_ms"] = leaf(med("tr_ms")), leaf(med("te_ms"))
            near15 = abs(float(field) - _FIELD_1P5T) < _FIELD_1P5T_TOL if field else True
            e["flip_deg_1p5t" if near15 else "flip_deg_3t"] = leaf(med("flip_deg"))
        return acq

    @staticmethod
    def fit(root: str | Path | None = None) -> dict:  # pragma: no cover  globs built meta.csv from disk + writes reference/acquisition.yaml (pure core = from_frame, tested)
        """Aggregate REAL DICOM acquisition from the built stores -> reference/acquisition.yaml. Only rows
        with real acquisition contribute (DICOM datasets, e.g. SCD=GE); NIfTI datasets have nulls and are
        skipped, so the domain-randomization sweep survives for everything we lack real values for."""
        base = Path(Config.data_root("processed"))
        metas = [pl.read_csv(str(f), infer_schema_length=0) for f in base.glob("*/*/meta.csv")]
        if not metas:
            return {}
        acq = AcqReference.from_frame(pl.concat(metas, how="diagonal"))
        if not acq:
            return {}
        out = Reference.reference_dir() / "acquisition.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("# Real per-(vendor,field) acquisition, DICOM-mined by AcqReference.fit.\n"
                       "# acquisition_for OVERRIDES the physics derivation with these where present (DAG compose).\n"
                       + OmegaConf.to_yaml(OmegaConf.create({"acquisition": acq})))
        return acq
