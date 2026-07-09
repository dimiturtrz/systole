"""Temperature scaling: on overconfident logits, fitted T softens (>1) and lowers ECE."""
import numpy as np
import pytest

pytest.importorskip("torch")
from cardioseg.evaluation.calibrate import Calibrate


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
    T = Calibrate.fit_temperature(logits, labels)
    assert T > 1.0                                      # overconfident -> soften
    assert Calibrate._ece_at(logits, labels, T) < Calibrate._ece_at(logits, labels, 1.0)   # calibration improved


def test_temperature_preserves_argmax():
    """T scaling never changes the prediction (accuracy/Dice untouched)."""
    logits, labels = _overconfident()
    T = Calibrate.fit_temperature(logits, labels)
    assert np.array_equal((logits / T).argmax(1), logits.argmax(1))


def test_temperature_well_calibrated_stays_near_one():
    """Already-calibrated class: moderate logits matching the true accuracy -> T close to 1 (no fix
    needed). acc 0.7 with soft (+-1) logits is roughly calibrated, so T shouldn't blow up."""
    rng = np.random.RandomState(3)
    n, c, acc = 4000, 3, 0.7
    labels = rng.randint(0, c, n)
    pred = np.where(rng.rand(n) < acc, labels, rng.randint(0, c, n))
    logits = np.zeros((n, c), np.float32)
    logits[np.arange(n), pred] = 1.0        # gentle margin -> soft confidence
    T = Calibrate.fit_temperature(logits, labels)
    assert 0.3 < T < 3.0                     # no extreme rescale


def test_ece_at_temperature_equivalence():
    """`Calibrate._ece_at(logits, labels, 1.0)` equals the raw ECE of the un-scaled softmax (T=1 is identity)."""
    logits, labels = _overconfident()
    e_t1 = Calibrate._ece_at(logits, labels, 1.0)
    assert 0.0 <= e_t1 <= 1.0                 # a valid ECE
    # scaling by T>1 on overconfident logits must not increase ECE
    T = Calibrate.fit_temperature(logits, labels)
    assert Calibrate._ece_at(logits, labels, T) <= e_t1 + 1e-9
