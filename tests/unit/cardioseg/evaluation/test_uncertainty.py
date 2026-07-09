"""Uncertainty cores — the pure entropy/calibration math behind the TTA decomposition (the I/O shell
that loads npz + saves plots is pragma'd).

Covers: `ece` (calibration-error binning), `_boundary` (foreground boundary band), `tta_uncertainty`
(entropy decomposition on a real CPU U-net), `foreground_uncertainty` + `boundary_interior_uncertainty`
(the per-case folds). Equivalence classes over confidence-vs-correctness and over mask geometry.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import cardioseg.evaluation.uncertainty as U
from cardioseg.evaluation.uncertainty import Uncertainty
from core.model import Model
from core.preprocessing.preprocess import SIZE as PSIZE

SIZE = 32


def test_ece_perfect_calibration_is_zero():
    """Perfectly-calibrated class: confidence == accuracy in each occupied bin -> ECE 0.
    (Bins are half-open (lo, hi] so conf exactly 0 falls in no bin — use a low bin > 0.)"""
    conf = np.array([1.0, 1.0, 0.1, 0.1])
    correct = np.array([1.0, 1.0, 0.1, 0.1])   # bin means: acc 0.1 == conf 0.1, acc 1 == conf 1
    e, bins = Uncertainty.ece(conf, correct, n_bins=15)
    assert abs(e) < 1e-6
    assert len(bins) == 2      # two occupied bins, both zero-gap


def test_ece_maximally_miscalibrated():
    """Worst class: confidence 1.0 but always wrong -> per-bin gap |acc-conf|=1, weight 1 -> ECE 1."""
    conf = np.full(10, 1.0)
    correct = np.zeros(10)
    e, _ = Uncertainty.ece(conf, correct)
    assert abs(e - 1.0) < 1e-6


def test_ece_partial_gap_is_weighted():
    """Half at conf 1.0/acc 0.5 (gap .5), half at conf 0.0/acc 0.0 (gap 0) -> ECE = .5*.5 = .25."""
    conf = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    correct = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    e, _ = Uncertainty.ece(conf, correct)
    assert abs(e - 0.25) < 1e-6


def test_ece_empty_bins_skipped():
    """Sparse class: samples in one bin only -> other 14 bins skipped (no NaN from empty-bin mean)."""
    conf = np.full(5, 0.5)
    correct = np.ones(5)
    e, bins = Uncertainty.ece(conf, correct)
    assert len(bins) == 1 and not np.isnan(e)


def test_boundary_band_is_ring():
    """A solid square's boundary band = the square minus its erosion (a 1-voxel ring), interior removed."""
    m = np.zeros((8, 8), np.uint8); m[2:6, 2:6] = 1
    b = Uncertainty._boundary(m)
    assert b[2, 2] and not b[3, 3]         # corner on band, centre off
    assert b.sum() < (m > 0).sum()          # ring smaller than the filled region


def test_boundary_empty_mask():
    """Empty-mask class: no foreground -> empty boundary band."""
    assert Uncertainty._boundary(np.zeros((5, 5), np.uint8)).sum() == 0


def test_tta_uncertainty_decomposition():
    """total = aleatoric + epistemic, epistemic >= 0 (BALD/Jensen), conf in [0,1], shapes match input.
    Run on a real CPU U-net over a 2-slice synthetic volume."""
    torch.manual_seed(0)
    model = Model.build_unet().eval()
    vol = np.random.RandomState(0).randn(2, SIZE, SIZE).astype(np.float32)
    pred, total, conf, ale, epi = Uncertainty.tta_uncertainty(model, vol, SIZE, "cpu")
    assert pred.shape == (2, SIZE, SIZE)
    assert np.allclose(total, ale + epi, atol=1e-5)
    assert (epi >= -1e-6).all()
    assert conf.min() >= 0.0 and conf.max() <= 1.0 + 1e-6
    assert (total >= -1e-6).all() and (total <= 1.0 + 1e-6).all()   # normalized by log C


def test_foreground_uncertainty_selects_union_and_scores():
    """fg = (pred>0) OR (gt>0). One matching fg voxel (pred==gt) + one bg-bg voxel dropped ->
    1 sample kept, correct=True, score = mean fg entropy."""
    pred = np.array([[1, 0]], np.uint8)
    gt = np.array([[1, 0]], np.uint8)
    ent = np.array([[0.4, 0.9]], np.float32)
    conf = np.array([[0.8, 0.3]], np.float32)
    cf, ok, en, al, ep, score = Uncertainty.foreground_uncertainty(pred, gt, (ent, conf, ent, ent))
    assert len(en) == 1 and ok[0] and abs(en[0] - 0.4) < 1e-6
    assert abs(score - 0.4) < 1e-6


def test_foreground_uncertainty_empty_score_zero():
    """Empty-foreground class: pred and gt all background -> no samples, per-case score defined 0.0."""
    z = np.zeros((1, 2), np.uint8); f = np.zeros((1, 2), np.float32)
    cf, ok, en, al, ep, score = Uncertainty.foreground_uncertainty(z, z, (f, f, f, f))
    assert len(en) == 0 and score == 0.0


def test_boundary_vs_interior_split():
    """A filled square: boundary voxels get high entropy, interior low -> boundary mean > interior mean
    (the sanity check that uncertainty concentrates on edges)."""
    pred = np.zeros((1, 8, 8), np.uint8); pred[0, 2:6, 2:6] = 1
    ent = np.full((1, 8, 8), 0.1, np.float32)
    ent[0][Uncertainty._boundary(pred[0])] = 0.9
    bnd, inte = Uncertainty.boundary_interior_uncertainty(pred, ent)
    assert len(bnd) == 1 and len(inte) == 1
    assert bnd[0] > inte[0]


def test_boundary_vs_interior_no_interior():
    """Thin (all-boundary) mask: a 1-voxel-wide region is all boundary -> interior list stays empty."""
    pred = np.zeros((1, 5, 5), np.uint8); pred[0, 2, :] = 1   # a line: erosion empties it
    ent = np.ones((1, 5, 5), np.float32)
    bnd, inte = Uncertainty.boundary_interior_uncertainty(pred, ent)
    assert len(bnd) == 1 and len(inte) == 0


# --------- _collect_uncertainty orchestration: real CPU U-net, store.load_arrays stubbed ---------

class _DF:
    """Minimal polars-frame stand-in: only `.iter_rows(named=True)` (one dict per case) is used."""

    def __init__(self, rows):
        self._rows = rows

    def iter_rows(self, named=True):
        return iter(self._rows)


def _fake_case(seed=0):
    """Synthetic ED+ES subject dict (concentric FG mask) in the `store.load_arrays` shape."""
    rng = np.random.RandomState(seed)
    img = rng.randn(2, PSIZE, PSIZE).astype(np.float32)
    gt = np.zeros((2, PSIZE, PSIZE), np.uint8)
    gt[:, 8:24, 8:24] = 2; gt[:, 12:20, 12:20] = 3; gt[:, 8:12, 8:12] = 1
    return {"ed_img": img, "ed_gt": gt, "es_img": img.copy(), "es_gt": gt.copy(),
            "spacing": np.array([8.0, 1.5, 1.5])}


def test_collect_uncertainty_pools_samples(monkeypatch, tmp_path):
    """Orchestration (non-acdc, no overlay PNG): _collect_uncertainty folds TTA entropy per ED/ES frame
    -> pooled sample lists + one case dict per frame. eval_name='canon' skips the overlay branch."""
    monkeypatch.setattr(U.store, "load_arrays", lambda p: _fake_case())
    torch.manual_seed(0); model = Model.build_unet().eval()
    df = _DF([{"path": "subj0.npz"}])
    confs, corrects, ents, ales, epis, bnd_u, int_u, cases = Uncertainty._collect_uncertainty(
        model, df, "cpu", tmp_path, "canon")
    assert len(confs) == 2 and len(cases) == 2          # ED + ES frames
    assert cases[0]["case"].endswith("_ED") and cases[1]["case"].endswith("_ES")
    assert all(c["uncertainty"] >= 0.0 for c in cases)
    assert not (tmp_path / "uncertainty_map.png").exists()   # non-acdc -> no overlay written


def test_collect_uncertainty_acdc_saves_overlay(monkeypatch, tmp_path):
    """acdc/ED branch: exactly one overlay PNG is written (highest-fg slice), once (saved latch)."""
    monkeypatch.setattr(U.store, "load_arrays", lambda p: _fake_case())
    torch.manual_seed(0); model = Model.build_unet().eval()
    df = _DF([{"path": "subj0.npz"}, {"path": "subj1.npz"}])
    Uncertainty._collect_uncertainty(model, df, "cpu", tmp_path, "acdc")
    assert (tmp_path / "uncertainty_map.png").exists()


def test_collect_uncertainty_skips_missing_frame(monkeypatch, tmp_path):
    """Missing-frame class: an ES-only case makes _collect_uncertainty skip the absent ED tag."""
    case = _fake_case(); del case["ed_img"], case["ed_gt"]
    monkeypatch.setattr(U.store, "load_arrays", lambda p: case)
    torch.manual_seed(0); model = Model.build_unet().eval()
    out = Uncertainty._collect_uncertainty(model, _DF([{"path": "s.npz"}]), "cpu", tmp_path, "canon")
    assert len(out[-1]) == 1 and out[-1][0]["case"].endswith("_ES")   # only ES frame collected
