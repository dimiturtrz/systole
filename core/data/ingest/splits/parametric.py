"""Parametric — the criteria knobs (`test_vendors`/`test_datasets`/`train_vendors`) expressed as a CODED
split family (bd cardiac-seg-gz19).

The frozen families (StaticMain, …) pin ONE test set by name+lock. The DataCfg criteria path is the
FLEXIBLE counterpart — arbitrary holdouts (hold out GE, or restrict train to Siemens, bd 5r7n) that no
frozen family expresses. This family closes that gap: the same knobs become constructor params, and the
resolved test is still content-hashed per combo (`Resolution.test_hash`), so an ad-hoc holdout is a
first-class coded split, not a criteria branch. Defaults reproduce the criteria defaults (Canon+GE +
the motion cohort held out, ACDC as the val centre-shift), so `Parametric()` == the canonical xvendor
split, and `Parametric(test_vendors=("GE",))` is a coded single-vendor holdout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import polars as pl

from core.data.ingest.source import StaticSource
from core.data.ingest.split import SplitDef
from core.data.ingest.testsets import SEG
from core.data.static.mri.base import Dataset, Vendor

V = pl.col


@dataclass(frozen=True)
class Parametric:
    name: ClassVar[str] = "parametric"
    sources: ClassVar[tuple[str, ...]] = ()
    test_vendors: tuple[str, ...] = (Vendor.CANON, Vendor.GE)     # vendors held out of train entirely (the OOD test)
    test_datasets: tuple[str, ...] = (Dataset.CMRXMOTION,)    # whole datasets held out (e.g. the motion cohort)
    train_vendors: tuple[str, ...] = ()                 # if set: restrict TRAIN to these vendors only (bd 5r7n)
    val_dataset: str = Dataset.ACDC                           # the domain-shift tuning centre

    def _test(self, cloud: pl.DataFrame) -> StaticSource:
        held = V("vendor").is_in(list(self.test_vendors)) | V("dataset").is_in(list(self.test_datasets))
        frame = cloud.filter(V("labelled") & V("dataset").is_in(SEG) & held)
        return StaticSource(frame, f"held-out vendors {self.test_vendors} + datasets {self.test_datasets}")

    def _val(self, cloud: pl.DataFrame) -> StaticSource:
        return StaticSource(cloud.filter(V("labelled") & (V("dataset") == self.val_dataset)),
                            f"{self.val_dataset} centre-shift")

    def _train(self, cloud: pl.DataFrame) -> StaticSource:
        keep = (V("labelled") & V("vendor").is_in(list(self.train_vendors))
                & ~V("vendor").is_in(list(self.test_vendors))
                & ~V("dataset").is_in(list(self.test_datasets)) & (V("dataset") != self.val_dataset))
        return StaticSource(cloud.filter(keep), f"train vendors {self.train_vendors}")

    @property
    def versions(self) -> dict[str, SplitDef]:
        # train=None -> the labelled complement (already excludes the held-out test+val); an explicit
        # train appears only when train_vendors restricts it to a vendor subset.
        train = self._train if self.train_vendors else None
        return {"1.0.0": SplitDef(test=self._test, val=self._val, train=train)}
