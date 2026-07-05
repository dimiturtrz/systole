"""Reusable frozen test sets — coded filters + a content-hash lock, referenced by SYMBOL not string.

A TestSet is the eval-target counterpart of a split: a coded predicate over the cloud (no named-manifest
string) plus a `lock` (sha256 of the resolved subject ids) that freezes it for comparability. Both the
split families (a split's `test` is a TestSet) and the generalization matrix (scores models on a list of
TestSets) consume these — one freeze mechanism, one home. Drift (the store grew the set) = lock mismatch
= hard error, never a silent change.

`task` says which metrics are valid: seg4 (all classes) or seg_lv (SCD, no RV → myo+cav only).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from core.data.source import StaticSource

V = pl.col


# The standard segmentation cohort (canonical 4-class datasets). Vendor test sets scope to it so a
# predicate like vendor=="GE" is SOURCE-INDEPENDENT: without the scope it would also match SCD's GE
# subjects when SCD is loaded, making the set (and its lock) depend on what else is in the cloud. The
# scope is part of the test-set's definition — coded data-values, not an opaque name.
SEG = ["acdc", "mnm2", "mnms1", "cmrxmotion"]
_IN_SEG = V("dataset").is_in(SEG)


@dataclass(frozen=True)
class TestSet:
    __test__ = False                         # not a pytest test class (name starts with "Test")
    name: str
    task: str
    predicate: pl.Expr                       # coded filter (labelled AND'd in by source())
    lock: str = ""                           # sha256 of resolved ids; "" until frozen

    def source(self, cloud: pl.DataFrame) -> StaticSource:
        s = StaticSource(cloud.filter(V("labelled") & self.predicate), self.name)
        if self.lock and s.ids_hash() != self.lock:
            raise ValueError(f"testset {self.name!r} drifted (lock {self.lock[:19]}… != "
                             f"{s.ids_hash()[:19]}…) — the store changed the set; freeze a new lock")
        return s


# ── granular eval targets (the matrix scores on these), scoped to the seg cohort ─────────────────
CANON = TestSet("canon", "seg4", _IN_SEG & (V("vendor") == "Canon"),
                "sha256:25de0c1c725e8e4692f0df46770317e9964fcd0f2cc2b6dd879c03965ccaf6c1")
GE = TestSet("ge", "seg4", _IN_SEG & (V("vendor") == "GE"),
             "sha256:972fa8d357139393e343874aaea7517e6d85e38876833ab48569d0b3e6e6fad5")
CMRXMOTION = TestSet("cmrxmotion", "seg4", V("dataset") == "cmrxmotion",
                     "sha256:1dde3ba03d3ed703bcc789e679530cb9c39826bab32bd96f98aa4c84bad193ca")
ACDC = TestSet("acdc", "seg4", V("dataset") == "acdc",
               "sha256:59abff2997890dd2d49da3d83e100d77dfc99ab211dfcc451d4383ca9de3fde8")
MNM2 = TestSet("mnm2", "seg4", V("dataset") == "mnm2",
               "sha256:b5e7e8f24f1a44c4faa30085354e5e8ddeede7914740c90e9083b5995038e4b2")
MNMS1 = TestSet("mnms1", "seg4", V("dataset") == "mnms1",
                "sha256:6d459b3934e706e2faa5aef14a05442f2d5b71d659939abe3bbe63d1eec1ed68")
SCD_LV = TestSet("scd_lv", "seg_lv", V("dataset") == "scd",
                 "sha256:1219e59aae642f34cdeb6a320c8cfcf2e4ee360a3dbd445e2304c9b5ddfc447b")

# ── composites (a split's `test`) ────────────────────────────────────────────────────────────────
# static_main: unseen vendors (GE, Canon) + the motion cohort. == the old xvendor frozen test (147).
STATIC_MAIN_TEST = TestSet(
    "static_main_test", "seg4",
    _IN_SEG & (V("vendor").is_in(["GE", "Canon"]) | (V("dataset") == "cmrxmotion")),
    "sha256:5f8f0a98e56065e6eca42ff51caf7d06d0be4ff98b8d9bf142dda77e8953faa5")
# synth_main: all seg real EXCEPT the ACDC val (642) — the near-all-real test for the synth arm.
SYNTH_MAIN_TEST = TestSet("synth_main_test", "seg4", _IN_SEG & (V("dataset") != "acdc"),
                          "sha256:ecd60aad00b5a19d49fc3b4bced9cc5768d10f931d4e94107d0c1dd031dfe8ed")

# the matrix's default granular battery (over-held models score OOD on the ones they didn't train)
MATRIX_TESTSETS: list[TestSet] = [CANON, GE, CMRXMOTION, ACDC, MNM2, MNMS1, SCD_LV]

_ALL: list[TestSet] = MATRIX_TESTSETS + [STATIC_MAIN_TEST, SYNTH_MAIN_TEST]
TESTSETS: dict[str, TestSet] = {ts.name: ts for ts in _ALL}     # lookup by name (CLI / reporting)

# datasets any TestSet can reference (incl SCD, not in the default seg SOURCE_DATASETS) — the eval cloud
EVAL_SOURCES = ["acdc", "mnm2", "mnms1", "cmrxmotion", "scd"]


def freeze_all(cloud: pl.DataFrame) -> dict[str, str]:
    """Recompute every TestSet's lock over `cloud` — run to fill/refresh the PLACEHOLDER locks."""
    out = {}
    for ts in _ALL:
        h = StaticSource(cloud.filter(V("labelled") & ts.predicate)).ids_hash()
        out[ts.name] = h
    return out
