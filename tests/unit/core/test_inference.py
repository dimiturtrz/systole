"""Inference-primitive + run-loading tests: predict_volume(_probs), resolve_device, load_run.

Uses an untrained Model.build_unet() on CPU with random input — we assert shapes/invariants/consistency,
not segmentation quality."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cardioseg.evaluation.uncertainty import Uncertainty  # noqa: E402
from core.hparams import Hparams, TrainCfg  # noqa: E402
from core.inference import Inference  # noqa: E402
from core.model import Model  # noqa: E402
from core.run import Run  # noqa: E402

SIZE = 32  # small grid; divisible by the 4 strides (2^4=16)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return Model.build_unet().eval()


def test_resolve_device_prefers_explicit():
    assert Model.resolve_device("cpu") == "cpu"          # explicit wins
    assert Model.resolve_device(None) in ("cuda", "cpu")  # auto, env-dependent


def test_predict_volume_probs_shapes_and_simplex(model):
    """pred is [D,size,size] uint8; mean_softmax is [D,C,size,size] and a proper simplex over C."""
    vol = np.random.RandomState(0).randn(3, 40, 28).astype(np.float32)  # [D,H,W], non-square
    pred, mean = Inference(model, SIZE, "cpu").predict_volume_probs(vol)
    assert pred.shape == (3, SIZE, SIZE) and pred.dtype == np.uint8
    assert tuple(mean.shape) == (3, 4, SIZE, SIZE)        # 4 classes
    assert torch.allclose(mean.sum(1), torch.ones(3, SIZE, SIZE), atol=1e-5)  # softmax simplex


def test_pred_equals_argmax_of_mean(model):
    """predict_volume(tta=True) must equal argmax of the probs primitive (same TTA path)."""
    vol = np.random.RandomState(1).randn(2, 32, 32).astype(np.float32)
    inf = Inference(model, SIZE, "cpu")
    pred_v = inf.predict_volume(vol, tta=True)
    pred_p, mean = inf.predict_volume_probs(vol)
    assert np.array_equal(pred_v, pred_p)
    assert np.array_equal(pred_p, mean.argmax(1).to(torch.uint8).cpu().numpy())


def test_predict_volume_no_tta_shape(model):
    vol = np.random.RandomState(2).randn(2, 32, 32).astype(np.float32)
    pred = Inference(model, SIZE, "cpu").predict_volume(vol, tta=False)
    assert pred.shape == (2, SIZE, SIZE) and pred.dtype == np.uint8


def test_members_and_bald_decomposition(model):
    """4 flip-members -> mean; BALD decomposition: total = aleatoric + epistemic, epistemic >= 0."""
    vol = np.random.RandomState(4).randn(2, 32, 32).astype(np.float32)
    pred, mean, members = Inference(model, SIZE, "cpu").predict_volume_members(vol)
    assert tuple(members.shape) == (4, 2, 4, SIZE, SIZE)        # K=4 flips
    assert torch.allclose(members.mean(0), mean, atol=1e-6)     # mean is the member average
    p, total, conf, ale, epi = Uncertainty.tta_uncertainty(model, vol, SIZE, "cpu")
    assert np.array_equal(p, pred)
    assert (epi >= -1e-6).all()                                 # BALD >= 0 (Jensen)
    assert np.allclose(total, ale + epi, atol=1e-5)            # total = aleatoric + epistemic
    assert total.max() <= 1.0 + 1e-6 and ale.min() >= -1e-6    # normalized to [0,1]


# --- load_run: rebuilds arch from config.json; falls back when absent ---
def _make_run(tmp_path, with_config):
    run = tmp_path / "run"; run.mkdir()
    torch.save(Model.build_unet().state_dict(), run / "model.pth")
    if with_config:
        Hparams.to_json(TrainCfg(), run / "config.json")
    return run


def test_load_run_with_config(tmp_path):
    run = _make_run(tmp_path, with_config=True)
    model, cfg, device = Run.load_run(run, "cpu")
    assert cfg is not None and device == "cpu"
    assert not model.training                    # eval() mode


def test_load_run_legacy_fallback(tmp_path):
    """A run without config.json still loads via the default ModelCfg (cfg is None)."""
    run = _make_run(tmp_path, with_config=False)
    model, cfg, _ = Run.load_run(run, "cpu")
    assert cfg is None and model is not None
