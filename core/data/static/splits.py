"""Train/val/test splits as criteria over the consolidated meta frame (data/store.py).

A split isn't a named thing — it's the data cloud filtered on criteria. Hold out everything matching
`test_datasets` (whole dataset) or `test_vendors` (by vendor) as test; train/val = the rest, labelled.
The criteria live on DataCfg (serialized to config.json), so a run self-documents what it held out.

    meta = store.load(cfg.generator.data.sources)
    train, val, test = Splits.make_split(meta, cfg.generator.data.test_datasets, cfg.generator.data.test_vendors,
                                         cfg.generator.data.val_frac, cfg.seed)

Change the criteria → change the split. No registry, no name, no flag.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from core.data.ingest.source import SubjectIds
from core.data.ingest.splits import Splits as SplitRegistry
from core.data.static.store.build import Build as store


class Splits:
    """The split-derivation free helpers, folded in as staticmethods. A split isn't a named thing —
    it's the data cloud filtered on criteria (see module docstring)."""

    @staticmethod
    def eval_set(name: str, *, holdout: bool = False, seed: int = 0) -> pl.DataFrame:
        """Labelled rows for a named eval set. `name` matches (case-insensitive) the dataset OR the vendor
        column, so 'acdc' (dataset) and 'canon'/'ge' (vendor) resolve through one path — no special-case.
        `holdout` carves the seed-0 0.2 val slice (in-domain runs)."""
        key = name.lower()
        df = store.load(None).filter(
            pl.col("labelled")
            & ((pl.col("dataset").str.to_lowercase() == key) | (pl.col("vendor").str.to_lowercase() == key)))
        return Splits.patient_val(df, 0.2, seed)[1] if holdout else df

    @staticmethod
    def split_patients(cases: list[Path], val_frac: float = 0.2, seed: int = 0
                       ) -> tuple[list[Path], list[Path]]:
        """Deterministic patient-level train/val split over raw case dirs -> (train_dirs, val_dirs).
        The path-list counterpart to patient_val (which splits the meta frame); used where a caller has
        case directories rather than the consolidated store (e.g. the viewer's held-out check)."""
        cases = list(cases)
        idx = np.random.default_rng(seed).permutation(len(cases))
        n_val = max(1, round(len(cases) * val_frac))
        val_names = {cases[i].name for i in idx[:n_val]}
        train = [c for c in cases if c.name not in val_names]
        val = [c for c in cases if c.name in val_names]
        return train, val

    @staticmethod
    def patient_val(train: pl.DataFrame, val_frac: float = 0.2, seed: int = 0
                    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Carve a deterministic val set out of train (subject-level; rows are one-per-subject)."""
        shuffled = train.sample(fraction=1.0, shuffle=True, seed=seed)
        n_val = max(1, round(len(shuffled) * val_frac))
        return shuffled[n_val:], shuffled[:n_val]

    @staticmethod
    def cap_subjects(train: pl.DataFrame, k: int, seed: int = 0) -> pl.DataFrame:
        """Cap TRAIN to K subjects, seeded (the data-scarcity knob, bd wqmh). Split rows are one-per-
        subject, so a seeded K-row sample is subject-disjoint and size-K by construction — never by-slice
        (which would leak a subject across the cap). k<=0 or k>=len(train) -> unchanged (0 = use all)."""
        if k <= 0 or k >= len(train):
            return train
        return train.sample(n=k, shuffle=True, seed=seed)

    @staticmethod
    def make_split(meta: pl.DataFrame, test_datasets: Any = (), test_vendors: Any = (), val_frac: float = 0.2,  # noqa: PLR0913  low-level split primitive; config-object path is split_from_cfg(DataCfg)
                   seed: int = 0, val_datasets: Any = (), val_vendors: Any = (), train_vendors: Any = (), train_subjects: int = 0
                   ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """(train, val, test) from criteria. test = rows whose dataset ∈ test_datasets OR vendor ∈
        test_vendors (+ labelled). VAL: if val_datasets/val_vendors given, val = rows matching those
        (a held-out *domain* for tuning that isn't test) — otherwise a random `val_frac` carved from
        train (in-domain). train = everything labelled that's neither test nor val. `train_vendors`
        (if given) restricts TRAIN to only those vendors — the scarce/single-vendor regime (bd 5r7n);
        val/test unaffected. `train_subjects` (if >0) then caps TRAIN to K subjects (bd wqmh scarcity
        sweep) — applied last so it composes with the vendor restriction; val/test unaffected."""
        test_expr = (pl.col("dataset").is_in(list(test_datasets))
                     | pl.col("vendor").is_in(list(test_vendors))) & pl.col("labelled")
        test = meta.filter(test_expr)
        rest = meta.filter(pl.col("labelled") & ~test_expr)
        if val_datasets or val_vendors:
            val_expr = pl.col("dataset").is_in(list(val_datasets)) | pl.col("vendor").is_in(list(val_vendors))
            train, val = rest.filter(~val_expr), rest.filter(val_expr)
        else:
            train, val = Splits.patient_val(rest, val_frac, seed)
        if train_vendors:                                           # restrict TRAIN only (val/test intact)
            train = train.filter(pl.col("vendor").is_in(list(train_vendors)))
        return Splits.cap_subjects(train, train_subjects, seed), val, test

    @staticmethod
    def paths(df: pl.DataFrame) -> list[str]:
        """The npz paths for a split (what the torch dataset consumes)."""
        return df.get_column("path").to_list()


class ModelSplit:
    """The split a trained model's DataCfg induces over a meta frame: construct with `(d, meta)`, then
    query `.val` / `.test` / `.seen_keys()` / `.train_keys()` / `.split()`. `d` (the run's DataCfg =
    criteria or a coded split) and `meta` (the store frame this query runs over) are the fixed session;
    each query derives one partition off them. Analysis/eval tools that want 'the model's held-out real
    slices' MUST go through this, not raw `Splits.make_split`: make_split reads only the criteria and
    silently ignores a coded split (a trap this removes)."""

    def __init__(self, d: Any, meta: pl.DataFrame) -> None:
        self.d, self.meta = d, meta

    def _criteria_split(self, meta: pl.DataFrame, seed: int = 0
                        ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """(train, val, test) from the DataCfg's CRITERIA over an arbitrary meta frame (train_keys runs it
        on a sources-filtered frame, so meta stays a param, not self.meta)."""
        d = self.d
        return Splits.make_split(meta, d.test_datasets, d.test_vendors, d.val_frac, seed,
                                 d.val_datasets, d.val_vendors, d.train_vendors, d.train_subjects)

    @property
    def val(self) -> pl.DataFrame:
        """The held-out val subject frame — the coded split's resolved val when `d.split` is set, else the
        DataCfg-criteria val."""
        if self.d.split:
            return SplitRegistry.resolve_cfg(self.d, self.meta).val.frame
        return self._criteria_split(self.meta)[1]

    @property
    def test(self) -> pl.DataFrame:
        """The frozen test subject frame — the coded split's test when `d.split` is set, else the criteria
        test. Per-axis (e.g. per-vendor) eval filters this rather than re-deriving from a literal."""
        if self.d.split:
            return SplitRegistry.resolve_cfg(self.d, self.meta).test.frame
        return self._criteria_split(self.meta)[2]

    def split(self, seed: int = 0) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """(train, val, test) from the DataCfg's CRITERIA over this model's meta. The LEGACY path — kept so
        a run without a coded split, and the matrix reconstructing an OLD model's train set from its saved
        DataCfg, still work. New splits are coded families (core.data.ingest.splits)."""
        return self._criteria_split(self.meta, seed)

    def seen_keys(self) -> set[str]:
        """The real subjects the model actually SAW = train ∪ val, restricted to its own `sources` — the
        honest basis for an OOD/leak check (a val subject IS seen; early-stopping touched it). Coded split:
        resolved val ∪ (static) train — a DYNAMIC/synth train contributes no real subjects. Old criteria:
        labelled ∩ sources − test. ANY tool that scores a model on a set it chose by name (not via the
        model's split) must exclude these to stay leak-free (bd cardiac-seg-h9bz). The `dataset\\tsubject`
        key format matches `subject_keys`, so callers can set-difference directly."""
        in_sources = self.meta.filter(pl.col("dataset").is_in(list(self.d.sources)))
        if self.d.split:
            r = SplitRegistry.resolve_cfg(self.d, in_sources)
            seen = set(r.val.subjects())                            # val is always a real StaticSource
            if r.train.kind == "static":
                seen |= set(r.train.subjects())                     # dynamic/synth train = no real subjects
            return {f"{a}\t{b}" for a, b in seen}
        test = (pl.col("dataset").is_in(list(self.d.test_datasets))
                | pl.col("vendor").is_in(list(self.d.test_vendors)))
        return SubjectIds.subject_keys(in_sources.filter(pl.col("labelled") & ~test))

    def train_keys(self) -> set[str]:
        """The real subjects the model actually TRAINED on (gradient) — like `seen_keys` but EXCLUDING val.
        For tools that may legitimately show the held-out VAL (qualitative overlays, a val-centre
        distribution) yet must never score on TRAIN: exclude these, keep val. Coded split: the static train
        subjects (a dynamic/synth train = none). Old criteria: the train partition of `split`."""
        in_sources = self.meta.filter(pl.col("dataset").is_in(list(self.d.sources)))
        if self.d.split:
            r = SplitRegistry.resolve_cfg(self.d, in_sources)
            if r.train.kind != "static":
                return set()                                        # dynamic/synth train touches no real subject
            return {f"{a}\t{b}" for a, b in r.train.subjects()}
        return SubjectIds.subject_keys(self._criteria_split(in_sources)[0])   # [0] = train (val carved off)
