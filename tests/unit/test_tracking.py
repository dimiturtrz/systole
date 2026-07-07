"""Tracking is optional + guarded — disabled -> no-op handle that never raises; config flattens."""
from cardioseg.tracking import _flat, _Noop, start


def test_flat_nested():
    assert _flat({"a": 1, "b": {"c": 2, "d": {"e": 3}}}) == {"a": 1, "b.c": 2, "b.d.e": 3}


def test_noop_when_disabled(monkeypatch):
    """CARDIOSEG_NO_MLFLOW -> no-op handle; all calls are safe (training must never depend on it)."""
    monkeypatch.setenv("CARDIOSEG_NO_MLFLOW", "1")
    h = start("cardioseg", "run", {"model": {"channels": 16}, "lr": 1e-3})
    assert isinstance(h, _Noop)
    h.metric("val_dice", 0.9, step=0)
    h.summary({"test": {"dice_mean": 0.84, "ef_mae": 11.0}})
    h.artifact("does/not/exist")
    h.end()                                    # none of these raise
