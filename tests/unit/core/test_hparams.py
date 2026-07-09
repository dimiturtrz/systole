"""Validated-config tests (equivalence classes): one accepting + one rejecting representative per
constraint. The point of pydantic here is rejecting bad --set/config at load — so the reject cases
are the value (cardiac-seg-8y9)."""
import pytest
from pydantic import ValidationError

from core.data.dynamic.augment import AugCfg
from core.hparams import Hparams, TrainCfg


# --- accept: valid defaults + roundtrip ---
def test_defaults_valid():
    cfg = TrainCfg()
    # split: TEST = unseen vendors (Canon+GE) + cmrxmotion (motion set), VAL = ACDC (held-out
    # centre), TRAIN = the rest
    g = cfg.generator
    assert g.data.test_vendors == ("Canon", "GE") and g.data.val_datasets == ("acdc",)
    assert g.data.test_datasets == ("cmrxmotion",) and cfg.loss.kind == "dice_ce"
    assert 0 <= g.aug.gamma_p <= 1


def test_json_roundtrip(tmp_path):
    cfg = TrainCfg()
    p = tmp_path / "config.json"
    Hparams.to_json(cfg, p)
    assert Hparams.from_json(p) == cfg               # serialize -> load reproduces the config exactly


# --- reject: out-of-bounds / wrong enum, one per constraint class ---
def test_reject_out_of_range_prob():
    with pytest.raises(ValidationError):
        AugCfg(gamma_p=1.5)                  # gamma_p in [0,1]


def test_reject_val_frac_ge_1():
    from core.data.static.store import DataCfg
    with pytest.raises(ValidationError):
        DataCfg(val_frac=1.0)                # val_frac in (0,1)


def test_reject_bad_loss_kind():
    from core.hparams import TrainCfg
    with pytest.raises(ValidationError):
        TrainCfg.model_validate({"loss": {"kind": "dice_ce_bogus"}})   # no such union variant


def test_loss_union_picks_variant_and_ignores_old_flat_fields():
    # OLD flat config (kind + every loss's params) must still load: discriminator picks the variant,
    # extra fields are dropped. Guards config.json backward-compat for registered models.
    from core.hparams import DiceCEHDCfg, TrainCfg
    flat = {"kind": "dice_ce_hd", "hd_weight": 0.02, "tversky_alpha": 0.3, "her_weight": 0.5}
    t = TrainCfg.model_validate({"loss": flat})
    assert isinstance(t.loss, DiceCEHDCfg)
    assert t.loss.hd_weight == 0.02
    assert not hasattr(t.loss, "tversky_alpha")   # other variants' params are gone, not carried


def test_reject_bad_spatial_dims():
    from core.model import ModelCfg
    with pytest.raises(ValidationError):
        ModelCfg(spatial_dims=5)             # Literal[2,3]


# --- apply_overrides: valid applies (incl tuple), invalid rejected at assignment ---
def test_override_valid_scalar_and_tuple():
    cfg = Hparams.apply_overrides(TrainCfg(), ["generator.aug.gamma_p=0.5", "generator.data.test_vendors=('GE',)"])
    assert cfg.generator.aug.gamma_p == 0.5 and cfg.generator.data.test_vendors == ("GE",)


def test_override_out_of_bounds_rejected():
    with pytest.raises(ValidationError):
        Hparams.apply_overrides(TrainCfg(), ["generator.aug.gamma_p=5"])   # validate_assignment fires


# --- _coerce on Optional[str] fields (currently-None): "none"-literal -> None, else string ---
def test_override_optional_none_literal():
    cfg = Hparams.apply_overrides(TrainCfg(), ["device=None"])
    assert cfg.device is None                            # not the string "None"


def test_override_optional_string_preserved():
    cfg = Hparams.apply_overrides(TrainCfg(), ["device=cuda"])
    assert cfg.device == "cuda"                          # real value still passes through


# --- N4Cfg: nested in DataCfg, recorded in config.json, bounds enforced ---
def test_n4cfg_in_config_roundtrip(tmp_path):
    cfg = TrainCfg()
    assert "n4_params" in cfg.model_dump()["generator"]["data"]   # recorded even when n4=False
    p = tmp_path / "c.json"; Hparams.to_json(cfg, p)
    assert Hparams.from_json(p) == cfg


def test_flat_config_backcompat():
    """Pre-refactor config.json had data/aug/synth at the top level; the TrainCfg before-validator
    lifts them under `generator` so registered models / cached configs still load."""
    cfg = TrainCfg.model_validate({"data": {"test_vendors": ("GE",)}, "aug": {"gamma_p": 0.5}})
    assert cfg.generator.data.test_vendors == ("GE",) and cfg.generator.aug.gamma_p == 0.5


def test_n4cfg_reject_bad_shrink():
    from core.preprocessing.n4 import N4Cfg
    with pytest.raises(ValidationError):
        N4Cfg(shrink=0)                                  # shrink >= 1
