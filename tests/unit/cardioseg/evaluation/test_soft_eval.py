"""SoftEval pure core (cardioseg.evaluation.soft_eval, class SoftEval._ef) — the EDV/ESV -> EF scalar
(equivalence classes: normal, zero/negative EDV -> NaN). The prediction loop + ECE need a model (shell,
pragma'd)."""
import numpy as np

from cardioseg.evaluation.soft_eval import SoftEval


# --- _ef: EDV/ESV -> EF scalar ---
def test_ef_normal():
    """Normal class: EF = (EDV-ESV)/EDV*100."""
    assert SoftEval._ef(100.0, 40.0) == 60.0


def test_ef_zero_edv_is_nan():
    """Collapse class: EDV<=0 (no cavity) -> NaN, never a divide-by-zero."""
    assert np.isnan(SoftEval._ef(0.0, 0.0))
    assert np.isnan(SoftEval._ef(-5.0, 1.0))
