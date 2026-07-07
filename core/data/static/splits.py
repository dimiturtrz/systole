"""Train/val/test splits as criteria over the consolidated meta frame (data/store.py).

A split isn't a named thing — it's the data cloud filtered on criteria. Hold out everything matching
`test_datasets` (whole dataset) or `test_vendors` (by vendor) as test; train/val = the rest, labelled.
The criteria live on DataCfg (serialized to config.json), so a run self-documents what it held out.

    meta = store.load(cfg.generator.data.sources)
    train, val, test = make_split(meta, cfg.generator.data.test_datasets, cfg.generator.data.test_vendors,
                                  cfg.generator.data.val_frac, cfg.seed)

Change the criteria → change the split. No registry, no name, no flag.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from core.data.ingest.splits import resolve_cfg
from core.data.static import store


def eval_set(name: str) -> pl.DataFrame:
    """Resolve a named EVAL set to its labelled subject rows, EXPRESSED AS A SPLIT — it routes through
    make_split's criteria and returns the TEST partition, so eval-set knowledge lives in the one split
    mechanism (test_vendors/test_datasets) rather than a bespoke filter. 'canon' = the unseen-vendor
    slice (test_vendors=['Canon'] over M&Ms-1); any other name = that whole dataset held out
    (test_datasets=[name]). Single home for what was copy-pasted in distribution.py + uncertainty.py."""
    if name == "canon":
        return make_split(store.load(["mnms1"]), test_vendors=("Canon",))[2]
    return make_split(store.load([name]), test_datasets=(name,))[2]


def split_patients(cases: list[Path], val_frac: float = 0.2, seed: int = 0
                   ) -> tuple[list[Path], list[Path]]:
    """Deterministic patient-level train/val split over raw case dirs -> (train_dirs, val_dirs).
    The path-list counterpart to patient_val (which splits the meta frame); used where a caller has
    case directories rather than the consolidated store (e.g. the viewer's held-out check)."""
    cases = list(cases)
    idx = np.random.default_rng(seed).permutation(len(cases))
    n_val = max(1, int(round(len(cases) * val_frac)))
    val_names = {cases[i].name for i in idx[:n_val]}
    train = [c for c in cases if c.name not in val_names]
    val = [c for c in cases if c.name in val_names]
    return train, val


def patient_val(train: pl.DataFrame, val_frac: float = 0.2, seed: int = 0
                ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Carve a deterministic val set out of train (subject-level; rows are one-per-subject)."""
    shuffled = train.sample(fraction=1.0, shuffle=True, seed=seed)
    n_val = max(1, round(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]


def make_split(meta: pl.DataFrame, test_datasets=(), test_vendors=(), val_frac: float = 0.2,  # noqa: PLR0913  low-level split primitive; config-object path is split_from_cfg(DataCfg)
               seed: int = 0, val_datasets=(), val_vendors=(), train_vendors=()
               ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(train, val, test) from criteria. test = rows whose dataset ∈ test_datasets OR vendor ∈
    test_vendors (+ labelled). VAL: if val_datasets/val_vendors given, val = rows matching those
    (a held-out *domain* for tuning that isn't test) — otherwise a random `val_frac` carved from
    train (in-domain). train = everything labelled that's neither test nor val. `train_vendors`
    (if given) restricts TRAIN to only those vendors — the scarce/single-vendor regime (bd 5r7n);
    val/test unaffected."""
    test_expr = (pl.col("dataset").is_in(list(test_datasets))
                 | pl.col("vendor").is_in(list(test_vendors))) & pl.col("labelled")
    test = meta.filter(test_expr)
    rest = meta.filter(pl.col("labelled") & ~test_expr)
    if val_datasets or val_vendors:
        val_expr = pl.col("dataset").is_in(list(val_datasets)) | pl.col("vendor").is_in(list(val_vendors))
        train, val = rest.filter(~val_expr), rest.filter(val_expr)
    else:
        train, val = patient_val(rest, val_frac, seed)
    if train_vendors:                                           # restrict TRAIN only (val/test intact)
        train = train.filter(pl.col("vendor").is_in(list(train_vendors)))
    return train, val, test


def paths(df: pl.DataFrame) -> list[str]:
    """The npz paths for a split (what the torch dataset consumes)."""
    return df.get_column("path").to_list()


def model_val(d, meta: pl.DataFrame) -> pl.DataFrame:
    """The val subject frame a model (DataCfg `d`) held out — a coded split's resolved val when
    `d.split` is set, else the DataCfg-criteria val. Analysis tools that want 'the model's held-out
    real slices' MUST use this, not raw make_split: make_split reads only the criteria and silently
    ignores a coded split (today it gives the right val only by the criteria defaults coinciding with
    the coded splits' val — a trap this removes)."""
    if getattr(d, "split", ""):
        return resolve_cfg(d, meta).val.frame
    return make_split(meta, d.test_datasets, d.test_vendors, d.val_frac, 0,
                      d.val_datasets, d.val_vendors, d.train_vendors)[1]


def split_from_cfg(d, meta: pl.DataFrame, seed: int = 0
                   ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(train, val, test) from a DataCfg's CRITERIA (test_datasets/test_vendors, val criteria). The
    LEGACY path — kept so a run without a coded split, and the matrix reconstructing an OLD model's
    train set from its saved DataCfg, still work. New splits are coded families (core.data.ingest.splits)."""
    return make_split(meta, d.test_datasets, d.test_vendors, d.val_frac, seed,
                      d.val_datasets, d.val_vendors, d.train_vendors)
