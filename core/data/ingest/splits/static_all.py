"""StaticAll — static_main + SCD in training via partial-label (RV masked).

Same frozen TEST + ACDC val as static_main (so the headline number is comparable), but the loaded
cloud (`sources` adds scd) means the complement TRAIN also includes SCD. SCD is LV-only {0,2,3}; its
slices join training with RV + bg masked out (labels.valid_row -> PartialLabelDiceCE), so they teach
the LV positive signal (a new GE/Canada centre) and never "RV -> bg". Tests whether a real new centre
in training lifts held-vendor generalization without hurting RV.
"""
from __future__ import annotations

from typing import ClassVar

import polars as pl

from core.data.ingest.source import StaticSource
from core.data.ingest.split import SplitDef
from core.data.ingest.testsets import STATIC_MAIN_TEST
from core.data.static.mri.base import Dataset
from core.data.static.mri.registry import SEG_DATASETS

V = pl.col


class StaticAll:
    name = "static_all"
    sources = (*SEG_DATASETS, Dataset.SCD)     # load SCD too (not in the default seg cloud)
    versions: ClassVar[dict[str, SplitDef]] = {
        "1.0.0": SplitDef(
            # == static_main's frozen test (147); lambda defers the global lookup (testset swap)
            test=lambda c: STATIC_MAIN_TEST.source(c),  # noqa: PLW0108
            val=lambda c: StaticSource(c.filter(V("labelled") & (V("dataset") == Dataset.ACDC)), "ACDC centre-shift"),
            # train = labelled complement: Siemens+Philips seg + ALL SCD (RV masked via partial-label)
        ),
    }
