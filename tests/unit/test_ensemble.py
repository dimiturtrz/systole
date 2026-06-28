"""Deep-ensemble BALD: total = aleatoric + epistemic; two weight-diverse models disagree (epi>0)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from core.model import build_unet
from cardioseg.evaluation.ensemble import ensemble_decompose

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
