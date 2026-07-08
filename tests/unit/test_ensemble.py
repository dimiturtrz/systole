"""Deep-ensemble BALD + the pure scoring folds behind ensemble_score/_headroom (the I/O shell that
loads npz + runs models is pragma'd).

Covers: `ensemble_decompose` (total = aleatoric + epistemic), `_dice_fold`/`_score_summary` (per-class
Dice accumulation + finalization), `reducible_frac` (epistemic fraction with the zero guard).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import cardioseg.evaluation.ensemble as E
from cardioseg.evaluation.ensemble import (
    _dice_fold,
    _headroom,
    _score_summary,
    ensemble_decompose,
    ensemble_score,
    reducible_frac,
)
from core.data.static.labels import FOREGROUND
from core.model import build_unet

SIZE = 32


def test_ensemble_decomposition():
    torch.manual_seed(0); m1 = build_unet().eval()
    torch.manual_seed(1); m2 = build_unet().eval()        # different weights -> a real ensemble
    vol = np.random.RandomState(0).randn(2, SIZE, SIZE).astype(np.float32)
    pred, total, ale, epi = ensemble_decompose([m1, m2], vol, SIZE, "cpu")
    assert pred.shape == (2, SIZE, SIZE)
    assert (epi >= -1e-6).all()                            # BALD >= 0
    assert np.allclose(total, ale + epi, atol=1e-5)        # total = aleatoric + epistemic
    assert epi.max() > 0                                   # distinct models disagree -> epistemic > 0


def _acc():
    return {c: 0.0 for c in FOREGROUND}, {c: 0.0 for c in FOREGROUND}


def test_dice_fold_perfect():
    """Perfect class: pred == gt -> per-class inter/den give Dice 1.0 on the finalize."""
    inter, den = _acc()
    gt = np.array([[1, 2, 3, 0]], np.uint8)
    _dice_fold(gt.copy(), gt, inter, den)
    out = _score_summary(inter, den, [])
    assert abs(out["dice_mean"] - 1.0) < 1e-9


def test_dice_fold_disjoint():
    """Disjoint class: no overlap on any label -> inter 0, den>0 -> Dice 0 each -> mean 0."""
    inter, den = _acc()
    pred = np.array([[1, 2, 3]], np.uint8)
    gt = np.array([[2, 3, 1]], np.uint8)     # every voxel a different label than pred
    _dice_fold(pred, gt, inter, den)
    assert _score_summary(inter, den, [])["dice_mean"] == 0.0


def test_dice_fold_accumulates_two_cases():
    """Accumulator: two folds pool inter/den (pooled Dice, not per-case averaged)."""
    inter, den = _acc()
    a = np.array([[1, 1, 0, 0]], np.uint8)                       # label-1 perfect (2 vox)
    _dice_fold(a.copy(), a, inter, den)
    b_gt = np.array([[1, 1, 0, 0]], np.uint8); b_pred = np.zeros((1, 4), np.uint8)  # all-miss
    _dice_fold(b_pred, b_gt, inter, den)
    # label 1: inter=2*2=4, den=(2+2)+(0+2)=6 ; labels 2,3 absent both -> NaN, dropped by nanmean
    assert abs(np.nanmean([4 / 6, np.nan, np.nan]) - (4 / 6)) < 1e-9   # sanity of the expected
    assert abs(_score_summary(inter, den, [])["dice_mean"] - round(4 / 6, 3)) < 1e-3


def test_score_summary_empty_absent_class_nan():
    """Empty-both across all classes -> every den 0 -> all Dice NaN -> nanmean NaN, EF MAE NaN."""
    inter, den = _acc()
    out = _score_summary(inter, den, [])
    assert np.isnan(out["dice_mean"]) and np.isnan(out["ef_mae"])


def test_score_summary_ef_mae():
    """EF MAE = mean|diff| over collected ED/ES diffs, rounded to 0.1."""
    inter, den = _acc()
    out = _score_summary(inter, den, [4.0, -2.0])
    assert out["ef_mae"] == 3.0


def test_reducible_frac_all_epistemic():
    """All-reducible class: aleatoric 0, epistemic > 0 -> fraction 1.0."""
    assert abs(reducible_frac([np.zeros(4)], [np.full(4, 0.5)]) - 1.0) < 1e-9


def test_reducible_frac_none_epistemic():
    """Irreducible class: epistemic 0 -> fraction 0.0."""
    assert reducible_frac([np.full(4, 0.5)], [np.zeros(4)]) == 0.0


def test_reducible_frac_half():
    """Balanced class: equal aleatoric and epistemic means -> fraction 0.5."""
    assert abs(reducible_frac([np.full(4, 0.3)], [np.full(4, 0.3)]) - 0.5) < 1e-9


def test_reducible_frac_zero_guard():
    """Degenerate class: both terms 0 -> guarded denom (1e-9), no div-by-zero -> 0.0."""
    assert reducible_frac([np.zeros(4)], [np.zeros(4)]) == 0.0


# --------- ensemble_score / _headroom orchestration: real CPU models, store.load_arrays stubbed ---------

class _DF:
    """Minimal stand-in for the polars eval frame: only `.iter_rows(named=True)` is used by the
    orchestration (one dict per case with a 'path' key)."""

    def __init__(self, rows):
        self._rows = rows

    def iter_rows(self, named=True):
        return iter(self._rows)


def _fake_case(seed=0):
    """Synthetic ED+ES subject dict in the `store.load_arrays` shape (concentric FG mask)."""
    rng = np.random.RandomState(seed)
    img = rng.randn(2, SIZE, SIZE).astype(np.float32)
    gt = np.zeros((2, SIZE, SIZE), np.uint8)
    gt[:, 8:24, 8:24] = 2; gt[:, 12:20, 12:20] = 3; gt[:, 8:12, 8:12] = 1
    return {"ed_img": img, "ed_gt": gt, "es_img": img.copy(), "es_gt": gt.copy(),
            "spacing": np.array([8.0, 1.5, 1.5])}


def _models(k=2):
    ms = []
    for s in range(k):
        torch.manual_seed(s); ms.append(build_unet().eval())
    return ms


def test_ensemble_score_end_to_end(monkeypatch):
    """Orchestration: ensemble_score over one case -> {dice_mean, ef_mae}, both finite floats."""
    monkeypatch.setattr(E.store, "load_arrays", lambda p: _fake_case())
    out = ensemble_score(_models(2), _DF([{"path": "c.npz"}]), SIZE, "cpu")
    assert set(out) == {"dice_mean", "ef_mae"}
    assert 0.0 <= out["dice_mean"] <= 1.0


def test_ensemble_score_es_only_no_ef(monkeypatch):
    """Missing-ED class: only ES present -> scored, no ED/ES pair -> EF MAE NaN (empty diffs)."""
    case = _fake_case(); del case["ed_img"], case["ed_gt"]
    monkeypatch.setattr(E.store, "load_arrays", lambda p: case)
    out = ensemble_score(_models(1), _DF([{"path": "c.npz"}]), SIZE, "cpu")
    assert np.isnan(out["ef_mae"])


def test_headroom_returns_two_fractions(monkeypatch):
    """_headroom: ensemble reducible-frac + single-model (TTA) reducible-frac, both in [0,1]."""
    monkeypatch.setattr(E.store, "load_arrays", lambda p: _fake_case())
    ef_red, tf_red = _headroom(_models(2), _DF([{"path": "c.npz"}]), SIZE, "cpu")
    assert 0.0 <= ef_red <= 1.0 and 0.0 <= tf_red <= 1.0


def test_headroom_skips_missing_frame(monkeypatch):
    """Missing-frame class: an ES-only case makes _headroom skip the absent ED tag (the continue)."""
    case = _fake_case(); del case["ed_img"], case["ed_gt"]
    monkeypatch.setattr(E.store, "load_arrays", lambda p: case)
    ef_red, tf_red = _headroom(_models(1), _DF([{"path": "c.npz"}]), SIZE, "cpu")
    assert 0.0 <= ef_red <= 1.0 and 0.0 <= tf_red <= 1.0
