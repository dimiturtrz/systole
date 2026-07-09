"""Soft-label training tests (equivalence classes): the honest-boundary-target machinery.
Key contract: with σ=0 (or a one-hot target) the soft path reduces to the existing hard Dice+CE,
so turning soft labels OFF is bit-for-bit the old behaviour."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from core.data.dynamic.augment import Augmentor
from core.losses import DiceCECfg, SoftDiceCE


# --- soften: shape, channels sum to 1, σ=0 crisp, σ>0 soft at boundary / hard interior ---
def test_soften_shape_and_normalized():
    m = torch.zeros(2, 8, 8, dtype=torch.long)
    m[:, 4:, :] = 1                                  # two-class split -> a boundary at row 4
    s = Augmentor.soften(m, sigma=1.0, n_classes=4)
    assert s.shape == (2, 4, 8, 8)
    assert torch.allclose(s.sum(1), torch.ones(2, 8, 8), atol=1e-5)   # channels sum to 1


def test_soften_sigma0_is_crisp_onehot():
    m = torch.randint(0, 4, (1, 6, 6))
    s = Augmentor.soften(m, sigma=0.0, n_classes=4)
    assert set(s.unique().tolist()) <= {0.0, 1.0}    # crisp
    assert (s.argmax(1) == m).all()


def test_soften_boundary_is_fractional_interior_hard():
    m = torch.zeros(1, 8, 8, dtype=torch.long)
    m[:, 4:, :] = 1
    s = Augmentor.soften(m, sigma=1.0, n_classes=4)
    # a boundary voxel (row 3/4) is a mix; a deep-interior voxel stays ~1
    assert 0.05 < s[0, 1, 4, 0] < 0.95              # just inside class 1 at the border -> fractional
    assert s[0, 0, 0, 0] > 0.98                     # top-left interior of class 0 -> ~hard


# --- SoftDiceCE: a one-hot soft target reproduces the hard Dice+CE (off = current behaviour) ---
def test_softdicece_matches_hard_on_onehot():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 8, 8)
    y = torch.randint(0, 4, (2, 8, 8))
    onehot = Augmentor.soften(y, sigma=0.0, n_classes=4)                 # crisp target
    soft = SoftDiceCE()(logits, onehot)
    hard = DiceCECfg().build()(logits, y[:, None])                    # MONAI DiceCELoss on int target
    assert torch.isfinite(soft)
    assert abs(float(soft) - float(hard)) < 1e-4               # soft path == hard path at σ=0


def test_softdicece_runs_on_soft_target():
    logits = torch.randn(2, 4, 8, 8)
    y = torch.zeros(2, 8, 8, dtype=torch.long); y[:, 4:, :] = 1
    loss = SoftDiceCE()(logits, Augmentor.soften(y, sigma=1.0, n_classes=4))
    assert torch.isfinite(loss) and loss > 0
