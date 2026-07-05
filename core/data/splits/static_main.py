"""StaticMain — the all-real generalization split (translation of the old `xvendor`).

Coded filters, no named manifests: test = unseen vendors (GE, Canon) OR the motion cohort
(cmrxmotion); val = the ACDC centre (domain-shift tuning); train = the labelled complement
(Siemens + Philips). `test_lock` pins the resolved test ids by content hash — frozen + comparable,
anchored by checksum not by name.
"""
from __future__ import annotations

import polars as pl

from core.data.source import StaticSource
from core.data.split import SplitDef
from core.data.testsets import STATIC_MAIN_TEST

V = pl.col


class StaticMain:
    name = "static_main"
    versions = {
        "1.0.0": SplitDef(
            test=lambda c: STATIC_MAIN_TEST.source(c),           # unseen vendors + motion cohort (147, locked)
            val=lambda c: StaticSource(
                c.filter(V("labelled") & (V("dataset") == "acdc")), "ACDC centre-shift"),
            # train = labelled complement (Siemens + Philips)
        ),
    }
