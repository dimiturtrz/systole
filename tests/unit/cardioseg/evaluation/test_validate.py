"""Validation scoring cores — the pure metric accumulation behind Evaluator (the I/O shell that loads
npz + runs the model is pragma'd; these test the math it folds into).

Covers: `_ClassScores` (one-hot Dice inter/denom over synthetic pred/gt masks + boundary distances),
`_foreground_samples` (calibration voxel selection/subsample), `summarize` (the report dict).
Equivalence classes over pred/gt overlap: perfect, disjoint, partial, empty-both, empty-one.
"""
import logging

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import cardioseg.evaluation.validate as V  # noqa: E402
from cardioseg.evaluation.validate import (  # noqa: E402
    CLASS_NAMES,
    EvalCfg,
    Evaluator,
    _ClassScores,
)
from core.model import Model  # noqa: E402

SP = (1.0, 1.0, 1.0)   # isotropic spacing (mm); Dice is spacing-free, surface uses it
SIZE = 32


def _vol(labels_2d):
    """Wrap a [H,W] label map as a 1-slice [1,H,W] volume (surface distances need 3D)."""
    return np.array(labels_2d, dtype=np.uint8)[None]


def test_dice_perfect_overlap():
    """Perfect class: pred == gt for every foreground label -> Dice 1.0 each."""
    gt = _vol([[1, 2], [3, 0]])
    s = _ClassScores(); s.add(gt.copy(), gt, SP)
    d = s.dice()
    assert all(abs(d[c] - 1.0) < 1e-9 for c in CLASS_NAMES)


def test_boundary_off_keeps_dice_skips_surface():
    """boundary=False: Dice identical to boundary=True, but no surface distances computed (the sweep
    speedup — HD95/ASSD EDT is the eval's heaviest step)."""
    pred = _vol([[1, 1], [2, 0]])
    gt = _vol([[1, 0], [2, 2]])
    on = _ClassScores(); on.add(pred.copy(), gt, SP)
    off = _ClassScores(boundary=False); off.add(pred.copy(), gt, SP)
    don, doff = on.dice(), off.dice()
    assert all((np.isnan(don[c]) and np.isnan(doff[c])) or don[c] == doff[c] for c in don)  # Dice unchanged
    assert all(not off.surf[c]["hd95"] and not off.surf[c]["assd"] for c in off.surf)  # no boundary work


def test_dice_disjoint_is_zero():
    """Disjoint class: pred and gt share no voxels of label 1 -> Dice 0 (both present, no overlap)."""
    pred = _vol([[1, 0], [0, 0]])
    gt = _vol([[0, 0], [0, 1]])
    s = _ClassScores(); s.add(pred, gt, SP)
    assert s.dice()[1] == 0.0


def test_dice_partial_overlap():
    """Partial class: pred label-1 = {(0,0),(0,1)}, gt = {(0,1),(1,1)} -> 1 shared, |P|=|G|=2 ->
    Dice = 2*1/(2+2) = 0.5."""
    pred = _vol([[1, 1], [0, 0]])
    gt = _vol([[0, 1], [0, 1]])
    s = _ClassScores(); s.add(pred, gt, SP)
    assert abs(s.dice()[1] - 0.5) < 1e-9


def test_dice_empty_both_is_nan():
    """Empty-both class: label 2 absent from pred AND gt -> denom 0 -> NaN (defined, no div-by-zero)."""
    pred = _vol([[1, 0], [0, 3]])
    gt = _vol([[1, 0], [0, 3]])   # label 2 (myo) never appears
    s = _ClassScores(); s.add(pred, gt, SP)
    assert np.isnan(s.dice()[2])


def test_dice_empty_one_is_zero():
    """Empty-one class: gt has label 1, pred has none -> denom>0, inter 0 -> Dice 0 (a miss, not NaN)."""
    pred = _vol([[0, 0], [0, 0]])
    gt = _vol([[1, 1], [0, 0]])
    s = _ClassScores(); s.add(pred, gt, SP)
    assert s.dice()[1] == 0.0


def test_dice_pools_over_volumes():
    """Accumulator class: two .add calls pool inter/denom (Dice is over the union, not averaged).
    Vol A perfect on label 1 (2 vox), vol B all-miss on label 1 (gt 2 vox, pred 0) ->
    pooled inter=2*2=4, denom=(2+2)+(0+2)=6 -> 4/6."""
    a = _vol([[1, 1], [0, 0]])
    b_gt = _vol([[1, 1], [0, 0]]); b_pred = _vol([[0, 0], [0, 0]])
    s = _ClassScores()
    s.add(a.copy(), a, SP)
    s.add(b_pred, b_gt, SP)
    assert abs(s.dice()[1] - (4 / 6)) < 1e-9


def test_surface_empty_when_label_absent():
    """No boundary samples for an absent label -> surface() median is NaN (nothing to summarize)."""
    pred = _vol([[1, 0], [0, 0]]); gt = _vol([[1, 0], [0, 0]])
    s = _ClassScores(); s.add(pred, gt, SP)
    surf = s.surface()
    assert np.isnan(surf[2]["hd95"]) and np.isnan(surf[2]["assd"])   # label 2 absent


def test_surface_zero_on_perfect_overlap():
    """Perfect overlap -> every boundary distance is 0 -> HD95 and ASSD both 0."""
    blob = np.zeros((1, 8, 8), np.uint8); blob[0, 2:6, 2:6] = 1
    s = _ClassScores(); s.add(blob.copy(), blob, SP)
    surf = s.surface()
    assert surf[1]["hd95"] == 0.0 and surf[1]["assd"] == 0.0


def test_foreground_samples_selects_fg_union():
    """Foreground = (gt>0) OR (argmax>0). Voxel 0: gt bg + pred bg -> dropped. Voxel 1: gt fg -> kept.
    Voxel 2: pred fg (argmax) though gt bg -> kept."""
    logits = np.array([[9.0, 0, 0, 0],    # argmax 0 (bg)
                       [9.0, 0, 0, 0],    # argmax 0 but gt fg
                       [0, 9.0, 0, 0]],   # argmax 1 (fg), gt bg
                      dtype=np.float32)
    y = np.array([0, 1, 0])
    rng = np.random.RandomState(0)
    lg, yy = Evaluator._foreground_samples(logits, y, per_vol=100, rng=rng)
    assert len(yy) == 2 and set(yy.tolist()) == {1, 0}


def test_foreground_samples_subsamples_to_cap():
    """Subsample class: more foreground voxels than per_vol -> exactly per_vol kept (bounded memory)."""
    n = 50
    logits = np.zeros((n, 4), np.float32); logits[:, 1] = 9.0   # all argmax 1 -> all foreground
    y = np.ones(n, np.int64)
    lg, yy = Evaluator._foreground_samples(logits, y, per_vol=10, rng=np.random.RandomState(1))
    assert len(yy) == 10 and lg.shape == (10, 4)


def test_foreground_samples_empty_when_all_background():
    """All-background class: no gt fg and every argmax is bg -> zero samples returned."""
    logits = np.zeros((5, 4), np.float32); logits[:, 0] = 9.0
    y = np.zeros(5, np.int64)
    lg, yy = Evaluator._foreground_samples(logits, y, per_vol=100, rng=np.random.RandomState(0))
    assert len(yy) == 0


def test_summarize_dict_and_mae(caplog):
    """summarize returns the JSON-able metrics dict: per-class Dice by name, mean Dice, EF MAE over
    rows, and passes boundary through. EF MAE = mean|ef_gt-ef_pred|."""
    dice = {1: 0.9, 2: 0.8, 3: 1.0}
    ef_rows = [{"patient": "p1", "group": "A", "ef_gt": 50.0, "ef_pred": 54.0, "edv_gt": 1, "edv_pred": 1},
               {"patient": "p2", "group": "B", "ef_gt": 60.0, "ef_pred": 58.0, "edv_gt": 1, "edv_pred": 1}]
    surf = {c: {"hd95": 3.0, "assd": 1.0} for c in CLASS_NAMES}
    with caplog.at_level(logging.INFO, logger="cardioseg.validate"):
        out = Evaluator.summarize(dice, ef_rows, surf)
    assert abs(out["dice_mean"] - (0.9 + 0.8 + 1.0) / 3) < 1e-9
    assert abs(out["ef_mae"] - 3.0) < 1e-9        # mean(|+4|,|-2|)
    assert out["dice"]["RV"] == 0.9
    assert out["boundary"]["LV-cav"]["hd95"] == 3.0


def test_summarize_no_ef_rows_is_nan():
    """No EF pairs (single-frame cases) -> EF MAE NaN, boundary None when omitted."""
    out = Evaluator.summarize({1: 0.5, 2: 0.5, 3: 0.5}, [])
    assert np.isnan(out["ef_mae"])
    assert out["boundary"] is None


# --------- Evaluator orchestration: real CPU U-net, load_arrays stubbed in-memory (no disk/GPU) ---------

def _fake_case(seed=0):
    """A synthetic consolidated-subject dict in the shape `load_arrays` returns (ED+ES img/gt,
    spacing, group) — a concentric RV/myo/LV-cav mask so every foreground label is present."""
    rng = np.random.RandomState(seed)
    img = rng.randn(2, SIZE, SIZE).astype(np.float32)
    gt = np.zeros((2, SIZE, SIZE), np.uint8)
    gt[:, 8:24, 8:24] = 2; gt[:, 12:20, 12:20] = 3; gt[:, 8:12, 8:12] = 1
    return {"ed_img": img, "ed_gt": gt, "es_img": img.copy(), "es_gt": gt.copy(),
            "spacing": np.array([8.0, 1.5, 1.5]), "group": "DCM"}


@pytest.fixture
def _model():
    torch.manual_seed(0)
    return Model.build_unet().eval()


def test_evaluator_validate_end_to_end(monkeypatch, _model):
    """Full case: validate() folds predict+Dice+EF -> Dice per FG class, one EF row, boundary dict."""
    monkeypatch.setattr(V, "load_arrays", lambda p: _fake_case())
    ev = Evaluator(_model, "cpu", EvalCfg(size=SIZE, postproc=True, tta=True))
    dice, ef_rows, surf = ev.validate(["case0.npz"])
    assert set(dice) == set(CLASS_NAMES) and set(surf) == set(CLASS_NAMES)
    assert len(ef_rows) == 1 and ef_rows[0]["group"] == "DCM"
    assert ef_rows[0]["patient"] == "case0"


def test_evaluator_validate_es_only_no_ef_row(monkeypatch, _model):
    """Missing-ED class: only ES present -> volume still scored, but no ED/ES pair -> no EF row."""
    case = _fake_case(); del case["ed_img"], case["ed_gt"]
    monkeypatch.setattr(V, "load_arrays", lambda p: case)
    ev = Evaluator(_model, "cpu", EvalCfg(size=SIZE, postproc=False, tta=False))
    dice, ef_rows, _ = ev.validate(["c.npz"])
    assert ef_rows == [] and set(dice) == set(CLASS_NAMES)


def test_evaluator_gather_shapes(monkeypatch, _model):
    """gather(): pooled foreground (logits[M,C=4], labels[M]) bounded by per_vol per ED/ES frame."""
    monkeypatch.setattr(V, "load_arrays", lambda p: _fake_case())
    ev = Evaluator(_model, "cpu", EvalCfg(size=SIZE))
    lg, y = ev.gather(["c.npz"], per_vol=50, seed=0)
    assert lg.ndim == 2 and lg.shape[1] == 4
    assert lg.shape[0] == y.shape[0] and 0 < lg.shape[0] <= 50 * 2


def test_evaluator_gather_skips_missing_frame(monkeypatch, _model):
    """Missing-frame class: an ES-only case makes gather() skip the absent ED tag (the continue)."""
    case = _fake_case(); del case["ed_img"], case["ed_gt"]
    monkeypatch.setattr(V, "load_arrays", lambda p: case)
    ev = Evaluator(_model, "cpu", EvalCfg(size=SIZE))
    lg, y = ev.gather(["c.npz"], per_vol=50, seed=0)
    assert lg.shape[0] == y.shape[0] > 0
