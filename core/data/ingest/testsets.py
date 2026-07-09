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
from pathlib import Path

import polars as pl

from core.data.ingest.source import StaticSource

log = logging.getLogger("cardioseg.testsets")

V = pl.col

# Locks live in a committed lockfile (DATA, derived) — not hand-pasted into the source (predicates are
# CODE). `python -m core.data.ingest.testsets --freeze` writes it; `--check` fails on drift (CI). Absent file
# -> empty locks -> no guard (bootstrap: run --freeze once).
_LOCKFILE = Path(__file__).resolve().parent / "testsets.lock.json"
_LOCKS: dict[str, str] = json.loads(_LOCKFILE.read_text()) if _LOCKFILE.exists() else {}


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


def _ts(name: str, task: str, predicate: pl.Expr) -> TestSet:
    """A TestSet with its lock pulled from the committed lockfile (empty until --freeze)."""
    return TestSet(name, task, predicate, _LOCKS.get(name, ""))


# ── granular eval targets (the matrix scores on these), scoped to the seg cohort ─────────────────
CANON = _ts("canon", "seg4", _IN_SEG & (V("vendor") == "Canon"))
GE = _ts("ge", "seg4", _IN_SEG & (V("vendor") == "GE"))
CMRXMOTION = _ts("cmrxmotion", "seg4", V("dataset") == "cmrxmotion")
ACDC = _ts("acdc", "seg4", V("dataset") == "acdc")
MNM2 = _ts("mnm2", "seg4", V("dataset") == "mnm2")
MNMS1 = _ts("mnms1", "seg4", V("dataset") == "mnms1")
SCD_LV = _ts("scd_lv", "seg_lv", V("dataset") == "scd")

# ── composites (a split's `test`) ────────────────────────────────────────────────────────────────
# static_main: unseen vendors (GE, Canon) + the motion cohort. == the old xvendor frozen test (147).
STATIC_MAIN_TEST = _ts("static_main_test", "seg4",
                       _IN_SEG & (V("vendor").is_in(["GE", "Canon"]) | (V("dataset") == "cmrxmotion")))
# synth_main: all seg real EXCEPT the ACDC val (642) — the near-all-real test for the synth arm.
SYNTH_MAIN_TEST = _ts("synth_main_test", "seg4", _IN_SEG & (V("dataset") != "acdc"))

# the matrix's default granular battery (over-held models score OOD on the ones they didn't train)
MATRIX_TESTSETS: list[TestSet] = [CANON, GE, CMRXMOTION, ACDC, MNM2, MNMS1, SCD_LV]

_ALL: list[TestSet] = MATRIX_TESTSETS + [STATIC_MAIN_TEST, SYNTH_MAIN_TEST]
TESTSETS: dict[str, TestSet] = {ts.name: ts for ts in _ALL}     # lookup by name (CLI / reporting)

# datasets any TestSet can reference (incl SCD, not in the default seg SOURCE_DATASETS) — the eval cloud
EVAL_SOURCES = ["acdc", "mnm2", "mnms1", "cmrxmotion", "scd"]


def compute_locks(cloud: pl.DataFrame) -> dict[str, str]:
    """Recompute every TestSet's lock (content hash) over `cloud`."""
    return {ts.name: StaticSource(cloud.filter(V("labelled") & ts.predicate)).ids_hash() for ts in _ALL}


if __name__ == "__main__":
    import argparse

    from core.data.static.store import build as store
    from core.obs import setup
    setup()
    ap = argparse.ArgumentParser(description="TestSet locks: --freeze writes the lockfile, --check verifies")
    ap.add_argument("--freeze", action="store_true", help="recompute + WRITE testsets.lock.json")
    ap.add_argument("--check", action="store_true", help="recompute + compare to lockfile; exit 1 on drift")
    args = ap.parse_args()
    fresh = compute_locks(store.load(EVAL_SOURCES))
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
