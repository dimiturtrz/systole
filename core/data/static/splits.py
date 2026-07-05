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


# ── Named splits ────────────────────────────────────────────────────────────────────────────────
# A split recipe IS a DataCfg (store.DataCfg): sources + train/val/test criteria + the synth knob
# (anatomy_pool/anatomy_mode). Named presets live here as override dicts — the codebase's config is
# typed-code, so a dict of DataCfg overrides (not a YAML folder) keeps one home, DRY, type-checked.
# TEST points at FROZEN manifests (comparable); train/val stay live criteria (the pool may grow).
# `named_split(name)` -> a DataCfg; train.py takes `--split <name>` and tags the model with it.
SPLITS: dict[str, dict] = {
    # The generalization split (was the DataCfg default): unseen vendors + a motion cohort held out
    # as the frozen test; ACDC = the domain-shift VAL (tune without peeking at test); train = the rest.
    "xvendor": dict(
        test_manifests=("vendor_canon", "vendor_ge", "dataset_cmrxmotion"),
        test_datasets=(), test_vendors=(),
        val_datasets=("acdc",),
    ),
    # Magnum opus — zero real data in training: TRAIN (and val) from the synthetic anatomy pool, ALL
    # real data is the TEST set. The purest domain-generalization claim. NB val is still real here
    # until the val-synth plumbing lands (a follow-up); the claim upgrades to "zero real in train".
    "synth_to_real": dict(
        anatomy_pool="", anatomy_mode="replace",          # set anatomy_pool at launch (--set) to a built pool
        test_manifests=("all_real",),
        test_datasets=(), test_vendors=(), val_datasets=("acdc",),
    ),
}


def named_split(name: str):
    """Build a DataCfg from a named preset in SPLITS (lazily imports DataCfg to avoid a cycle)."""
    from core.data.static.store import DataCfg
    if name not in SPLITS:
        raise KeyError(f"unknown split {name!r}; have {sorted(SPLITS)}")
    return DataCfg(**SPLITS[name])


def _keyed(meta: pl.DataFrame) -> pl.DataFrame:
    return meta.with_columns(
        (pl.col("dataset") + "\t" + pl.col("subject_id").cast(pl.Utf8)).alias("_k"))


def split_from_cfg(d, meta: pl.DataFrame, seed: int = 0
                   ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, list[list]]:
    """(train, val, test, missing) from a DataCfg-like `d`, honouring FROZEN test manifests.

    If `d.test_manifests` is set: TEST = the union of those frozen manifests' subjects (resolved to
    current rows); those exact subjects are carved OUT of train (no leak); val = val_datasets/
    val_vendors criteria else a random frac; `missing` = frozen ids absent from the store (drift, the
    caller must surface). Else: fall back to criteria `make_split` (missing = [])."""
    tm = tuple(getattr(d, "test_manifests", ()) or ())
    if not tm:
        tr, val, test = make_split(meta, d.test_datasets, d.test_vendors, d.val_frac, seed,
                                   d.val_datasets, d.val_vendors, d.train_vendors)
        return tr, val, test, []
    from core.data.static import manifest
    want: set[str] = set()
    for name in tm:
        want |= {f"{a}\t{b}" for a, b in manifest.load(name)["subjects"]}
    k = _keyed(meta).filter(pl.col("labelled"))
    in_test = pl.col("_k").is_in(want)
    test = k.filter(in_test).drop("_k")
    rest = k.filter(~in_test)
    missing = [x.split("\t") for x in sorted(want - set(k.filter(in_test)["_k"]))]
    if d.val_datasets or d.val_vendors:
        vexpr = pl.col("dataset").is_in(list(d.val_datasets)) | pl.col("vendor").is_in(list(d.val_vendors))
        train, val = rest.filter(~vexpr).drop("_k"), rest.filter(vexpr).drop("_k")
    else:
        train, val = patient_val(rest.drop("_k"), d.val_frac, seed)
    if d.train_vendors:
        train = train.filter(pl.col("vendor").is_in(list(d.train_vendors)))
    return train, val, test, missing
