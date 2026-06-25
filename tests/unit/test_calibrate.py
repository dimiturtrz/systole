"""Temperature scaling: on overconfident logits, fitted T softens (>1) and lowers ECE."""
import numpy as np
import pytest

pytest.importorskip("torch")
from cardioseg.evaluation.calibrate import fit_temperature, _ece_at


def _overconfident(n=3000, c=3, acc=0.7, seed=0):
    """Model that is ~accurate `acc` of the time but emits VERY peaked logits -> overconfident."""
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, c, n)
    pred = np.where(rng.rand(n) < acc, labels, rng.randint(0, c, n))
    logits = np.full((n, c), -4.0, np.float32)
    logits[np.arange(n), pred] = 4.0                   # near-certain on pred, but only `acc` correct
    return logits, labels


def test_temperature_softens_and_calibrates():
    logits, labels = _overconfident()
    T = fit_temperature(logits, labels)
    assert T > 1.0                                      # overconfident -> soften
    assert _ece_at(logits, labels, T) < _ece_at(logits, labels, 1.0)   # calibration improved


def test_temperature_preserves_argmax():
    """T scaling never changes the prediction (accuracy/Dice untouched)."""
    logits, labels = _overconfident()
    T = fit_temperature(logits, labels)
    assert np.array_equal((logits / T).argmax(1), logits.argmax(1))
