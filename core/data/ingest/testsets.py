"""Reusable frozen test sets — coded filters + a content-hash lock, referenced by SYMBOL not string.

A TestSet is the eval-target counterpart of a split: a coded predicate over the cloud (no named-manifest
string) plus a `lock` (sha256 of the resolved subject ids) that freezes it for comparability. Both the
split families (a split's `test` is a TestSet) and the generalization matrix (scores models on a list of
TestSets) consume these — one freeze mechanism, one home. Drift (the store grew the set) = lock mismatch
= hard error, never a silent change.

`task` says which metrics are valid: seg4 (all classes) or seg_lv (SCD, no RV → myo+cav only).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import polars as pl

from core.data.ingest.source import StaticSource
from core.data.static.mri.base import Vendor
from core.data.static.mri.registry import SEG_DATASETS, Dataset
from core.data.static.store.build import Build as store

log = logging.getLogger("cardioseg.testsets")

V = pl.col


class Task(StrEnum):
    """What a TestSet's metrics mean: seg4 (all 4 classes) or seg_lv (SCD, no RV -> myo+cav only). The
    frozen-manifest task tag lives here, beside TestSet. (An `ef` regression task joins when the EF lane
    builds an EF testset — bd ax4a; not added speculatively.)"""
    SEG4 = "seg4"
    SEG_LV = "seg_lv"

# Locks live in a committed lockfile (DATA, derived) — not hand-pasted into the source (predicates are
# CODE). `python -m core.data lock-testsets --freeze` writes it; `--check` fails on drift (CI). Absent file
# -> empty locks -> no guard (bootstrap: run --freeze once).
_LOCKFILE = Path(__file__).resolve().parent / "testsets.lock.json"
_LOCKS: dict[str, str] = json.loads(_LOCKFILE.read_text()) if _LOCKFILE.exists() else {}


# The standard segmentation cohort (canonical 4-class datasets). Vendor test sets scope to it so a
# predicate like vendor=="GE" is SOURCE-INDEPENDENT: without the scope it would also match SCD's GE
# subjects when SCD is loaded, making the set (and its lock) depend on what else is in the cloud. The
# scope is part of the test-set's definition — coded data-values, not an opaque name.
SEG = list(SEG_DATASETS)
_IN_SEG = V("dataset").is_in(SEG)


@dataclass(frozen=True)
class TestSet:
    __test__ = False                         # not a pytest test class (name starts with "Test")
    name: str
    task: Task
    predicate: pl.Expr                       # coded filter (labelled AND'd in by source())
    lock: str = ""                           # sha256 of resolved ids; "" until frozen

    def source(self, cloud: pl.DataFrame) -> StaticSource:
        s = StaticSource(cloud.filter(V("labelled") & self.predicate), self.name)
        if self.lock and s.ids_hash() != self.lock:
            raise ValueError(f"testset {self.name!r} drifted (lock {self.lock[:19]}… != "
                             f"{s.ids_hash()[:19]}…) — the store changed the set; freeze a new lock")
        return s


class TestSets:
    """Factory + drift-computation helpers for the module's frozen TestSet constants (siblings
    `TestSets.ts(...)`). Holds no state — the TestSet instances live at module level."""

    @staticmethod
    def ts(name: str, task: Task, predicate: pl.Expr) -> TestSet:
        """A TestSet with its lock pulled from the committed lockfile (empty until --freeze)."""
        return TestSet(name, task, predicate, _LOCKS.get(name, ""))

    @staticmethod
    def compute_locks(cloud: pl.DataFrame) -> dict[str, str]:
        """Recompute every TestSet's lock (content hash) over `cloud`."""
        return {ts.name: StaticSource(cloud.filter(V("labelled") & ts.predicate)).ids_hash() for ts in _ALL}

    @staticmethod
    def add_args(ap):
        ap.add_argument("--freeze", action="store_true", help="recompute + WRITE testsets.lock.json")
        ap.add_argument("--check", action="store_true", help="recompute + compare to lockfile; exit 1 on drift")

    @staticmethod
    def run(args):
        fresh = TestSets.compute_locks(store.load(EVAL_SOURCES))
        if args.freeze:
            _LOCKFILE.write_text(json.dumps(fresh, indent=2) + "\n")
            log.info(f"froze {len(fresh)} locks -> {_LOCKFILE.name}")
        else:                                                        # --check (default)
            drift = {n: (h, _LOCKS.get(n, "")) for n, h in fresh.items() if h != _LOCKS.get(n, "")}
            if drift:
                for n, (now, was) in drift.items():
                    log.warning(f"DRIFT {n}: lockfile {was[:19]}… != store {now[:19]}…")
                raise SystemExit(1)
            log.info(f"OK — {len(fresh)} TestSet locks match the store")


# ── granular eval targets (the matrix scores on these), scoped to the seg cohort ─────────────────
CANON = TestSets.ts("canon", Task.SEG4, _IN_SEG & (V("vendor") == Vendor.CANON))
GE = TestSets.ts("ge", Task.SEG4, _IN_SEG & (V("vendor") == Vendor.GE))
CMRXMOTION = TestSets.ts("cmrxmotion", Task.SEG4, V("dataset") == Dataset.CMRXMOTION)
ACDC = TestSets.ts("acdc", Task.SEG4, V("dataset") == Dataset.ACDC)
MNM2 = TestSets.ts("mnm2", Task.SEG4, V("dataset") == Dataset.MNM2)
MNMS1 = TestSets.ts("mnms1", Task.SEG4, V("dataset") == Dataset.MNMS1)
SCD_LV = TestSets.ts("scd_lv", Task.SEG_LV, V("dataset") == Dataset.SCD)

# ── composites (a split's `test`) ────────────────────────────────────────────────────────────────
# static_main: unseen vendors (GE, Canon) + the motion cohort. == the old xvendor frozen test (147).
STATIC_MAIN_TEST = TestSets.ts("static_main_test", Task.SEG4,
                                _IN_SEG & (V("vendor").is_in([Vendor.GE, Vendor.CANON])
                                           | (V("dataset") == Dataset.CMRXMOTION)))
# synth_main: all seg real EXCEPT the ACDC val (642) — the near-all-real test for the synth arm.
SYNTH_MAIN_TEST = TestSets.ts("synth_main_test", Task.SEG4, _IN_SEG & (V("dataset") != Dataset.ACDC))

# the matrix's default granular battery (over-held models score OOD on the ones they didn't train)
MATRIX_TESTSETS: list[TestSet] = [CANON, GE, CMRXMOTION, ACDC, MNM2, MNMS1, SCD_LV]

_ALL: list[TestSet] = [*MATRIX_TESTSETS, STATIC_MAIN_TEST, SYNTH_MAIN_TEST]
TESTSETS: dict[str, TestSet] = {ts.name: ts for ts in _ALL}     # lookup by name (CLI / reporting)

# datasets any TestSet can reference (incl SCD, not in the default seg SOURCE_DATASETS) — the eval cloud
EVAL_SOURCES = [*SEG_DATASETS, Dataset.SCD]
