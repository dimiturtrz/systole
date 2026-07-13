"""E2E (full pipeline, real-ish input): train_seg end-to-end on each coded split family — the whole
composition root split -> Source -> Generator -> train loop -> eval -> save, on CPU, 1 epoch.

This is the guard for the composition-root bug CLASS that slipped this session (train_df unbound on the
DYNAMIC branch, acq/residency wiring) — none were import-graph or dead-code findings, only an end-to-end
run through train_seg catches them. Distinct from tests/integration (module-pair seam, mocked) and from
the unit tests (isolated fns).

Kept fast + hermetic: real ACDC val/test (skips without the gated dataset, like test_smoke), but the
SYNTH train pool is mocked tiny (one load_pool patch) so the dynamic/composite epoch is ~20 slices not
~10k, and tracking is a no-op (no mlflow). device=cpu, epochs=1, n_patients caps val/test eval.
"""
import numpy as np
import pytest

from cardioseg.training.train import Train
from core.data.ingest.source import StaticSource
from core.data.static.mri.acdc import AcdcAdapter
from core.hparams import TrainCfg

needs_data = pytest.mark.skipif(not AcdcAdapter().cases(), reason="ACDC data not present (set CARDIAC_DATA_ROOT)")


class _NoTrk:
    """No-op tracker — keeps the E2E off mlflow (the run still exercises every train/eval/save step)."""
    def metric(self, *a, **k): pass
    def summary(self, *a, **k): pass
    def end(self): pass


@needs_data
@pytest.mark.parametrize("split", ["static_main", "synth_main", "synth_composite"])
def test_train_seg_e2e_smoke(split, tmp_path, monkeypatch):
    """train_seg runs the full pipeline for each split family and returns a finite val Dice + writes the
    model. Covers BOTH train_df branches: static (frame) and dynamic/composite (frameless)."""
    # Keep it a smoke, not a fit: tiny synth pool (dynamic epoch ~20 slices), cap every StaticSource to a
    # few real npz (else the full-ACDC val preload dominates at minutes), no-op tracking (no mlflow).
    _orig_paths = StaticSource.paths
    monkeypatch.setattr(StaticSource, "paths", lambda self, _o=_orig_paths: _o(self)[:4])
    monkeypatch.setattr("core.data.dynamic.anatomy.Anatomy.load_pool", lambda p: np.zeros((20, 32, 32), np.int64))
    monkeypatch.setattr("cardioseg.tracking.Tracker.track_run", lambda *a, **k: _NoTrk())

    cfg = TrainCfg()
    cfg.device = "cpu"
    cfg.epochs = 1
    cfg.n_patients = 2          # cap val/test eval (+ legacy train frame) — a smoke, not a real fit
    cfg.out_dir = str(tmp_path)
    cfg.generator.data.split = split
    cfg.generator.data.size = 32                          # tiny input — pool mock is 32², val resized to match

    model, results = Train.train_seg(cfg, quick=True, seeds=[0])

    assert model is not None
    dm = results["val"]["dice_mean"]
    assert isinstance(dm, float) and dm == dm            # finite (not NaN) — the whole chain produced a score
    assert any(tmp_path.rglob("model.pth"))              # save step wrote the model under the staging dir
