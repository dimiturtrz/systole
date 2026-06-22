"""Train/val/test splits as criteria over the consolidated meta frame (data/store.py).

A split isn't a named thing — it's the data cloud filtered on criteria. Hold out everything matching
`test_datasets` (whole dataset) or `test_vendors` (by vendor) as test; train/val = the rest, labelled.
The criteria live on DataCfg (serialized to config.json), so a run self-documents what it held out.

    meta = store.load(cfg.data.sources)
    train, val, test = make_split(meta, cfg.data.test_datasets, cfg.data.test_vendors,
                                  cfg.data.val_frac, cfg.seed)

Change the criteria → change the split. No registry, no name, no flag.
"""
from __future__ import annotations

import polars as pl


def patient_val(train: pl.DataFrame, val_frac: float = 0.2, seed: int = 0
                ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Carve a deterministic val set out of train (subject-level; rows are one-per-subject)."""
    shuffled = train.sample(fraction=1.0, shuffle=True, seed=seed)
    n_val = max(1, round(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]


def make_split(meta: pl.DataFrame, test_datasets=(), test_vendors=(), val_frac: float = 0.2,
               seed: int = 0) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(train, val, test) from criteria. test = rows whose dataset ∈ test_datasets OR vendor ∈
    test_vendors (+ labelled); train+val = everything else labelled, val carved at the subject level."""
    test_expr = (pl.col("dataset").is_in(list(test_datasets))
                 | pl.col("vendor").is_in(list(test_vendors))) & pl.col("labelled")
    test = meta.filter(test_expr)
    train_all = meta.filter(pl.col("labelled") & ~test_expr)
    train, val = patient_val(train_all, val_frac, seed)
    return train, val, test


def paths(df: pl.DataFrame) -> list[str]:
    """The npz paths for a split (what the torch dataset consumes)."""
    return df.get_column("path").to_list()
