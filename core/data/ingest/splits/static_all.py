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

V = pl.col


class StaticAll:
    name = "static_all"
    sources = ("acdc", "mnm2", "mnms1", "cmrxmotion", "scd")     # load SCD too (not in the default seg cloud)
    versions: ClassVar[dict[str, SplitDef]] = {
        "1.0.0": SplitDef(
            # == static_main's frozen test (147); lambda defers the global lookup (testset swap)
            test=lambda c: STATIC_MAIN_TEST.source(c),  # noqa: PLW0108
            val=lambda c: StaticSource(c.filter(V("labelled") & (V("dataset") == "acdc")), "ACDC centre-shift"),
            # train = labelled complement: Siemens+Philips seg + ALL SCD (RV masked via partial-label)
        ),
    }
