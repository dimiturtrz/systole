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


def make_split(meta: pl.DataFrame, test_datasets=(), test_vendors=(), val_frac: float = 0.2,
               seed: int = 0, val_datasets=(), val_vendors=()
               ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(train, val, test) from criteria. test = rows whose dataset ∈ test_datasets OR vendor ∈
    test_vendors (+ labelled). VAL: if val_datasets/val_vendors given, val = rows matching those
    (a held-out *domain* for tuning that isn't test) — otherwise a random `val_frac` carved from
    train (in-domain). train = everything labelled that's neither test nor val."""
    test_expr = (pl.col("dataset").is_in(list(test_datasets))
                 | pl.col("vendor").is_in(list(test_vendors))) & pl.col("labelled")
    test = meta.filter(test_expr)
    rest = meta.filter(pl.col("labelled") & ~test_expr)
    if val_datasets or val_vendors:
        val_expr = pl.col("dataset").is_in(list(val_datasets)) | pl.col("vendor").is_in(list(val_vendors))
        return rest.filter(~val_expr), rest.filter(val_expr), test   # train, val (held-out domain), test
    train, val = patient_val(rest, val_frac, seed)
    return train, val, test


def paths(df: pl.DataFrame) -> list[str]:
    """The npz paths for a split (what the torch dataset consumes)."""
    return df.get_column("path").to_list()
