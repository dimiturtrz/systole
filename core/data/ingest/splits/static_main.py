"""StaticMain — the all-real generalization split (translation of the old `xvendor`).

Coded filters, no named manifests: test = unseen vendors (GE, Canon) OR the motion cohort
(cmrxmotion); val = the ACDC centre (domain-shift tuning); train = the labelled complement
(Siemens + Philips). `test_lock` pins the resolved test ids by content hash — frozen + comparable,
anchored by checksum not by name.
"""
from __future__ import annotations

from typing import ClassVar

import polars as pl

from core.data.ingest.source import StaticSource
from core.data.ingest.split import SplitDef
from core.data.ingest.testsets import STATIC_MAIN_TEST
from core.data.static.mri.base import Dataset

V = pl.col


class StaticMain:
    name = "static_main"
    sources = ()                        # default seg cloud (= DataCfg.sources); no extra
    versions: ClassVar[dict[str, SplitDef]] = {
        "1.0.0": SplitDef(
            # unseen vendors + motion cohort (147, locked); lambda defers the global lookup (testset swap)
            test=lambda c: STATIC_MAIN_TEST.source(c),  # noqa: PLW0108
            val=lambda c: StaticSource(
                c.filter(V("labelled") & (V("dataset") == Dataset.ACDC)), "ACDC centre-shift"),
            # train = labelled complement (Siemens + Philips)
        ),
    }
