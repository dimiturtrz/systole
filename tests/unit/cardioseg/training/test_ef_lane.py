"""EF-ratio weak-supervision math (ef_lane) — the spacing-cancelling (ED-ES)/ED core + its Huber loss,
extracted out of KaggleEF.loss (which needs a cine pool + GPU forward — that's the shell). Equivalence
classes over cavity-total inputs; synthetic tensors, no model. Mirrors test_volumes' vol_loss idiom.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import cardioseg.training.ef_lane as EL
from cardioseg.training.ef_lane import (
    EfLane,
    KaggleEF,
    VolConsistency,
)

_cav_volume = EfLane.cav_volume
_stack = EfLane.stack
_zscore = EfLane.zscore
build_aux = EfLane.build_aux
ef_ratio = EfLane.ef_ratio
ef_ratio_loss = EfLane.ef_ratio_loss
from core.data.static.labels import LV_CAV
from core.model import Model


def test_ef_ratio_matches_hand_formula():
    """Normal class: EF = (ED-ES)/ED*100 elementwise on cavity pixel totals (units cancel)."""
    ed = torch.tensor([100.0, 200.0])
    es = torch.tensor([40.0, 100.0])
    assert torch.allclose(ef_ratio(ed, es), torch.tensor([60.0, 50.0]))


def test_ef_ratio_scale_invariant():
    """Ratio class: scaling both ED and ES by any factor (spacing/px) leaves EF unchanged."""
    ed, es = torch.tensor([120.0]), torch.tensor([50.0])
    assert torch.allclose(ef_ratio(ed, es), ef_ratio(ed * 7.3, es * 7.3))


def test_ef_ratio_empty_ed_clamped_no_nan():
    """Collapse class: ED=0 (empty prediction) clamps at 1e-6 -> ~0% not NaN/inf."""
    r = ef_ratio(torch.tensor(0.0), torch.tensor(0.0))
    assert torch.isfinite(r) and abs(float(r)) < 1e-3


def test_ef_ratio_loss_zero_when_pred_equals_target():
    """Exact class: predicted EF-ratio == csv target -> zero Huber."""
    ed, es = torch.tensor([100.0]), torch.tensor([40.0])   # EF 60%
    assert float(ef_ratio_loss(ed, es, [60.0])) == 0.0


def test_ef_ratio_loss_positive_on_mismatch():
    """Mismatch class: wrong target -> positive loss."""
    ed, es = torch.tensor([100.0]), torch.tensor([40.0])   # EF 60%
    assert float(ef_ratio_loss(ed, es, [45.0])) > 0


def test_ef_ratio_loss_accepts_tensor_targets():
    """Target-type class: a tensor target behaves like a list target (same value)."""
    ed, es = torch.tensor([100.0]), torch.tensor([40.0])
    as_list = float(ef_ratio_loss(ed, es, [45.0]))
    as_tensor = float(ef_ratio_loss(ed, es, torch.tensor([45.0])))
    assert abs(as_list - as_tensor) < 1e-9


def test_ef_ratio_loss_is_differentiable():
    """Grad class: loss carries a gradient back to the (soft-cavity) inputs — usable as a loss."""
    ed = torch.tensor([100.0], requires_grad=True)
    es = torch.tensor([40.0], requires_grad=True)
    ef_ratio_loss(ed, es, [45.0]).backward()
    assert ed.grad is not None and float(ed.grad.abs().sum()) > 0


# --- _zscore: per-slice standardization the Kaggle lane feeds the net (pure numpy) ---
def test_zscore_zero_mean_unit_std():
    """Normal class: output has ~0 mean and ~1 std (the ε is negligible on a spread input)."""
    s = _zscore(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert abs(float(s.mean())) < 1e-5 and abs(float(s.std()) - 1.0) < 1e-3


def test_zscore_constant_input_no_nan():
    """Degenerate class: a constant slice (std=0) -> finite zeros via the 1e-6 floor, never NaN."""
    s = _zscore(np.full((2, 2), 7.0))
    assert np.isfinite(s).all() and float(np.abs(s).max()) < 1e-3


# --- build_aux: the lane-assembly OFF switches (return [] without touching GPU/data) ---
def _cfg(ef_lambda=1.0, ef_kaggle=False):
    return SimpleNamespace(ef_lambda=ef_lambda, ef_kaggle=ef_kaggle, ef_subjects=4,
                           ef_kaggle_subjects=4, seed=0,
                           generator=SimpleNamespace(data=SimpleNamespace(size=64)))


def test_build_aux_off_when_lambda_nonpositive():
    """Off class: ef_lambda<=0 disables the lane -> empty list (no VolConsistency built)."""
    assert build_aux(_cfg(ef_lambda=0.0), None, None, "cpu", is_static=True) == []


def test_build_aux_off_when_not_static():
    """Source class: a non-static train source has no labeled EDV/ESV frame -> empty list."""
    assert build_aux(_cfg(ef_lambda=1.0), None, None, "cpu", is_static=False) == []


# --- _stack: [D,H,W] numpy -> [D,1,size,size] device tensor (pure grid-fit, runs on CPU) ---
def test_stack_shapes_dtype_device():
    """Fit class: each slice grid-fit to size, stacked to [D,1,size,size] float32 on the given device."""
    vol = np.zeros((3, 20, 24), np.float32); vol[:, 5:15, 5:15] = 1.0
    t = _stack(vol, 16, "cpu")
    assert t.shape == (3, 1, 16, 16) and t.dtype == torch.float32 and t.device.type == "cpu"


# --- forward-path lanes on a real CPU U-Net: the disk loaders (load_arrays/load_sax) are stubbed with
#     synthetic arrays, the model forward + segment-sum run on CPU (amp=False makes autocast('cuda') a
#     no-op). Mirrors test_validate/test_ensemble's stub-the-loader idiom. ---
SIZE = 32


def _cpu_model():
    torch.manual_seed(0)
    return Model.build_unet().eval()


def _fake_case(dz=2, seed=0):
    """A synthetic consolidated-subject dict in the shape load_arrays returns (ED+ES img/gt + spacing),
    with a cavity blob so EDV>0 (ED bigger than ES, as in diastole)."""
    rng = np.random.RandomState(seed)
    ed_gt = np.zeros((dz, 20, 20), np.uint8); ed_gt[:, 6:14, 6:14] = LV_CAV       # big cavity
    es_gt = np.zeros((dz, 20, 20), np.uint8); es_gt[:, 8:12, 8:12] = LV_CAV       # smaller cavity
    return {"ed_img": rng.randn(dz, 20, 20).astype(np.float32),
            "es_img": rng.randn(dz, 20, 20).astype(np.float32),
            "ed_gt": ed_gt, "es_gt": es_gt, "spacing": np.array([10.0, 1.5, 1.5])}


def test_cav_volume_segment_sums_per_item():
    """_cav_volume: one batched forward over ΣDi slices -> per-item soft LV-cav totals [K], grad-carrying.
    Two items of 2 and 3 slices -> a length-2 vector; runs on CPU with amp off."""
    model = _cpu_model()
    stacks = torch.randn(5, 1, SIZE, SIZE, requires_grad=True)   # 5 = 2 + 3 slices
    sizes = torch.tensor([2, 3])
    out = _cav_volume(model, stacks, sizes, LV_CAV, amp=False)
    assert out.shape == (2,) and out.requires_grad
    out.sum().backward()
    assert stacks.grad is not None and float(stacks.grad.abs().sum()) > 0


def test_volconsistency_builds_and_losses(monkeypatch):
    """VolConsistency: stubbed load_arrays -> GPU(cpu)-resident ED/ES stacks + GT EDV/ESV; loss() samples
    subjects, forwards on CPU, returns a finite grad-carrying vol_loss scalar."""
    monkeypatch.setattr(EL, "load_arrays", lambda _p: _fake_case())
    vc = VolConsistency([Path("a.npz"), Path("b.npz")], SIZE, "cpu", k=2)
    assert vc.n == 2 and vc.edv_gt.min() > 0                     # both subjects kept (EDV>0)
    loss = vc.loss(_cpu_model(), amp=False)
    assert loss is not None and torch.isfinite(loss)


def test_volconsistency_skips_empty_cavity_and_missing_frame(monkeypatch):
    """Build-time skips: a subject with no ED frame OR zero-EDV is dropped; empty pool -> loss None."""
    empty = _fake_case(); empty["ed_gt"] = np.zeros_like(empty["ed_gt"])           # EDV 0 -> skip
    monkeypatch.setattr(EL, "load_arrays", lambda _p: empty)
    vc = VolConsistency([Path("a.npz")], SIZE, "cpu", k=2)
    assert vc.n == 0 and vc.loss(_cpu_model(), amp=False) is None


def _fake_sax(L=2, P=3, seed=0):
    """A stubbed load_sax result: L slice-locations, each a [P,H,W] cine + spacing + meta."""
    rng = np.random.RandomState(seed)
    return [(rng.randn(P, 18, 18).astype(np.float32), (8.0, 1.5, 1.5), {}) for _ in range(L)]


def test_kaggleef_builds_pool_and_losses(monkeypatch):
    """KaggleEF: stubbed load_sax fills the host-RAM cine pool (cases with an EF target kept); loss()
    phase-finds + two CPU forwards -> a finite grad-carrying ef_ratio Huber."""
    monkeypatch.setattr(EL.KaggleDsbAdapter, "load_sax", staticmethod(lambda _c: _fake_sax()))
    cases = [Path("1"), Path("2")]
    ef_targets = {"1": {"ef": 55.0}, "2": {"ef": 62.0}}
    lane = KaggleEF(cases, ef_targets, SIZE, "cpu", k=2, pool=8, seed=0)
    assert lane.n == 2
    loss = lane.loss(_cpu_model(), amp=False)
    assert loss is not None and torch.isfinite(loss)


def test_kaggleef_skips_cases_without_ef_target(monkeypatch):
    """Filter class: a case absent from ef_targets (or ef missing) is skipped; no targets -> loss None."""
    monkeypatch.setattr(EL.KaggleDsbAdapter, "load_sax", staticmethod(lambda _c: _fake_sax()))
    lane = KaggleEF([Path("1")], {"1": {"ef": None}}, SIZE, "cpu", k=1, pool=8)
    assert lane.n == 0 and lane.loss(_cpu_model(), amp=False) is None
