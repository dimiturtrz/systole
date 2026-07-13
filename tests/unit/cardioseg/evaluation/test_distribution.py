"""Pure error-distribution aggregation (distribution.py) — pooling per-case rows, grouping by a
stratify key, the per-group strata table, and the LoA string. The GPU inference (collect) + matplotlib
render (plot_*) + file writes are the shell (pragma'd / not unit-tested); the STATS these figures
draw are pure dict/array logic, tested here with synthetic rows.
"""
import numpy as np

from cardioseg.evaluation.distribution import Distribution
from core.data.static.labels import CLASSES
from core.measure import LOA_Z

RV, MYO, CAV = tuple(CLASSES)   # 1, 2, 3


def _row(dice=0.9, ef=(50.0, 48.0), vendor="A", **extra):
    """A minimal per-case row like collect() emits: sd + dice per class, ef gt/pred, stratify keys."""
    r = {"sd": {c: np.array([1.0, 2.0]) for c in CLASSES},
         "dice": dict.fromkeys(CLASSES, dice),
         "ef_gt": ef[0], "ef_pred": ef[1], "vendor": vendor}
    r.update(extra)
    return r


# --- _pooled: concat class-wise, split ef pairs ---
def test_pooled_concatenates_and_splits_ef():
    """Normal class: dists/dice collected per class, ef unzipped into aligned gt/pred arrays."""
    rows = [_row(dice=0.8, ef=(50.0, 45.0)), _row(dice=0.6, ef=(60.0, 66.0))]
    dists, dice_acc, ef_gt, ef_pred = Distribution.pooled(rows)
    assert dice_acc[RV] == [0.8, 0.6]
    assert list(ef_gt) == [50.0, 60.0] and list(ef_pred) == [45.0, 66.0]
    assert len(dists[CAV]) == 2


def test_pooled_skips_rows_missing_keys():
    """Missing class: a row without 'sd'/'dice'/'ef_gt' is dropped from that pool (no KeyError)."""
    rows = [_row(), {"vendor": "B"}]          # 2nd row has no sd/dice/ef
    dists, dice_acc, ef_gt, ef_pred = Distribution.pooled(rows)
    assert len(dice_acc[RV]) == 1 and len(ef_gt) == 1


def test_pooled_empty_is_empty_arrays():
    """Boundary: no rows -> empty ef arrays, empty per-class lists (never a crash)."""
    dists, dice_acc, ef_gt, ef_pred = Distribution.pooled([])
    assert ef_gt.size == 0 and ef_pred.size == 0 and dice_acc[RV] == []


# --- _groups: bucket by key, drop missing key/ef, sort by size desc ---
def test_groups_buckets_and_sorts_largest_first():
    """Sort class: groups ordered largest-first; only rows with the key AND ef counted."""
    rows = [_row(vendor="A"), _row(vendor="A"), _row(vendor="B")]
    g = Distribution._groups(rows, "vendor")
    assert list(g) == ["A", "B"] and len(g["A"]) == 2 and len(g["B"]) == 1


def test_groups_drops_missing_key_or_ef():
    """Drop class: a row with key=None or no ef_gt is excluded from every bucket."""
    rows = [_row(vendor="A"), _row(vendor=None), {"vendor": "C"}]  # 2nd: None key; 3rd: no ef
    g = Distribution._groups(rows, "vendor")
    assert set(g) == {"A"}


# --- strata_table: per-group dice/EF dict ---
def test_strata_table_empty_when_no_groups():
    """Empty class: no groupable rows -> {} (nothing to stratify)."""
    assert Distribution.strata_table([], "vendor") == {}


def test_strata_table_shape_and_values():
    """Normal class: per-group dict has n, per-class + mean dice, ef mae/bias, gt_ef mean."""
    rows = [_row(dice=1.0, ef=(50.0, 50.0), vendor="A"),
            _row(dice=0.5, ef=(60.0, 54.0), vendor="A")]
    out = Distribution.strata_table(rows, "vendor")
    a = out["A"]
    assert a["n"] == 2
    assert abs(a["dice_mean"] - 0.75) < 1e-9         # mean of 1.0 and 0.5
    assert set(a["dice"]) == {"RV", "LV-myo", "LV-cav"}
    assert abs(a["ef_bias"] - (-3.0)) < 1e-6         # mean of (0, -6)
    assert abs(a["gt_ef_mean"] - 55.0) < 1e-6


# --- lo_hi: the LoA-band string ---
def test_lo_hi_formats_signed_band():
    """LoA string = bias ± LOA_Z·sd, each signed to 1 dp; sd=0 collapses the band to the bias."""
    assert Distribution.lo_hi(0.0, 0.0) == "+0.0, +0.0"
    assert Distribution.lo_hi(2.0, 1.0) == f"{2.0 - LOA_Z:+.1f}, {2.0 + LOA_Z:+.1f}"
