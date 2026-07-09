"""Pure axis-record assembly (results.axis_dict) — the published-JSON shape + fixed rounding that the
docs read, extracted from `_axis` (whose `collect` is the GPU shell). Equivalence classes over the
pooled per-class boundary/dice inputs + the EF stats dict; synthetic arrays, no model, no store.
"""
import numpy as np

from cardioseg.evaluation.results import axis_dict
from core.data.static.labels import CLASSES

RV, MYO, CAV = tuple(CLASSES)   # 1, 2, 3


def _stats(mae=4.2, bias=-3.1, loa=(-12.0, 5.8)):
    return {"mae": mae, "bias": bias, "loa": list(loa)}


def _dists(sd=1.0):
    """One non-empty boundary-distance array per class (so surface_metrics runs)."""
    return {c: [np.array([sd, sd * 2, sd * 3])] for c in CLASSES}


def _dice(vals=(0.9, 0.8, 0.95)):
    return {RV: [vals[0]], MYO: [vals[1]], CAV: [vals[2]]}


# --- shape + keys ---
def test_axis_dict_has_published_shape():
    """Structure class: the axis dict carries n, per-class+mean dice, hd95, assd, ef mae/bias/loa."""
    out = axis_dict(150, _dists(), _dice(), _stats())
    assert out["n"] == 150
    assert set(out["dice"]) == {"RV", "LV-myo", "LV-cav", "mean"}
    assert set(out) == {"n", "dice", "hd95", "assd", "ef_mae", "ef_bias", "ef_loa"}


# --- rounding: each field to its fixed precision (docs depend on this) ---
def test_axis_dict_rounds_per_class_dice_3dp():
    """Dice class: per-class dice = mean of the dice list, rounded to 3 dp."""
    out = axis_dict(2, _dists(), _dice(vals=(0.86412, 0.8, 0.9)), _stats())
    assert out["dice"]["RV"] == 0.864


def test_axis_dict_mean_dice_over_classes():
    """Mean class: dice.mean = round(mean of the 3 per-class dice, 3dp) — not of the raw lists."""
    out = axis_dict(2, _dists(), _dice(vals=(0.9, 0.8, 0.7)), _stats())
    assert out["dice"]["mean"] == round((0.9 + 0.8 + 0.7) / 3, 3)


def test_axis_dict_ef_fields_rounded_1dp():
    """EF class: mae/bias to 1dp, loa a 2-list each to 1dp."""
    out = axis_dict(2, _dists(), _dice(), _stats(mae=4.27, bias=-3.14, loa=(-12.05, 5.83)))
    assert out["ef_mae"] == 4.3 and out["ef_bias"] == -3.1
    assert out["ef_loa"] == [-12.1, 5.8]


# --- boundary-metric boundary: empty pooled dists -> NaN hd95/assd (no crash) ---
def test_axis_dict_empty_dists_gives_nan_surface():
    """Empty class: a class with only empty boundary arrays -> hd95/assd NaN, dice still computed."""
    dists = {c: [np.array([])] for c in CLASSES}
    out = axis_dict(1, dists, _dice(), _stats())
    assert np.isnan(out["hd95"]["RV"]) and np.isnan(out["assd"]["RV"])
    assert out["dice"]["RV"] == 0.9      # dice unaffected by empty boundary
