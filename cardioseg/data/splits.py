"""Train/val/test splits as queries over the consolidated meta frame (data/store.py).

The roles aren't baked into the datasets — a split is a *rule* over metadata. The flagship
generalization battery is one such rule: hold out a clean centre-shift set (ACDC) AND an unseen
vendor (Canon), train on everything else that has usable masks.

    meta = store.load()                 # the whole cloud
    train, val, test = battery(meta)    # declarative split

Swap the predicate to ask a different question (leave-one-vendor-out, pathology holdout, …) without
touching the data layer.
"""
from __future__ import annotations

import polars as pl


def split(meta: pl.DataFrame, test_expr: pl.Expr, train_expr: pl.Expr | None = None
          ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(train, test) by polars predicates. Default train = everything labelled and not in test."""
    test = meta.filter(test_expr)
    if train_expr is None:
        train_expr = pl.col("labelled") & ~test_expr
    train = meta.filter(train_expr)
    return train, test


def patient_val(train: pl.DataFrame, val_frac: float = 0.2, seed: int = 0
                ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Carve a deterministic val set out of train at the subject level (no slice leakage —
    rows are already one-per-subject, but this keeps the contract explicit)."""
    shuffled = train.sample(fraction=1.0, shuffle=True, seed=seed)
    n_val = max(1, round(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]


def battery(meta: pl.DataFrame, val_frac: float = 0.2, seed: int = 0
            ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """The generalization battery: test = ACDC (centre shift) ∪ Canon (unseen vendor); train+val =
    everything else with usable masks. Returns (train, val, test)."""
    test_expr = (pl.col("dataset") == "acdc") | ((pl.col("vendor") == "Canon") & pl.col("labelled"))
    train_all, test = split(meta, test_expr)
    train, val = patient_val(train_all, val_frac, seed)
    return train, val, test


def paths(df: pl.DataFrame) -> list[str]:
    """The npz paths for a split (what the torch dataset consumes)."""
    return df.get_column("path").to_list()
