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

V = pl.col


class StaticMain:
    name = "static_main"
    versions = {
        "1.0.0": SplitDef(
            test=lambda c: StaticSource(
                c.filter(V("labelled") & (V("vendor").is_in(["GE", "Canon"]) | (V("dataset") == "cmrxmotion"))),
                "unseen vendors (GE, Canon) + motion cohort (cmrxmotion)"),
            val=lambda c: StaticSource(
                c.filter(V("labelled") & (V("dataset") == "acdc")), "ACDC centre-shift"),
            # train = labelled complement (Siemens + Philips)
            test_lock="sha256:5f8f0a98e56065e6eca42ff51caf7d06d0be4ff98b8d9bf142dda77e8953faa5",  # 147 subjects; == old xvendor frozen manifests
        ),
    }
