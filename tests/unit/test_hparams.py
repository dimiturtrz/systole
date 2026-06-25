"""Validated-config tests (equivalence classes): one accepting + one rejecting representative per
constraint. The point of pydantic here is rejecting bad --set/config at load — so the reject cases
are the value (cardiac-seg-8y9)."""
import pytest
from pydantic import ValidationError

from cardioseg.hparams import TrainCfg, AugCfg, apply_overrides, to_json, from_json


# --- accept: valid defaults + roundtrip ---
def test_defaults_valid():
    cfg = TrainCfg()
    assert cfg.data.test_datasets == ("acdc",) and cfg.loss.kind == "dice_ce"
    assert 0 <= cfg.aug.gamma_p <= 1


def test_json_roundtrip(tmp_path):
    cfg = TrainCfg()
    p = tmp_path / "config.json"
    to_json(cfg, p)
    assert from_json(p) == cfg               # serialize -> load reproduces the config exactly


# --- reject: out-of-bounds / wrong enum, one per constraint class ---
def test_reject_out_of_range_prob():
    with pytest.raises(ValidationError):
        AugCfg(gamma_p=1.5)                  # gamma_p in [0,1]


def test_reject_val_frac_ge_1():
    from cardioseg.hparams import DataCfg
    with pytest.raises(ValidationError):
        DataCfg(val_frac=1.0)                # val_frac in (0,1)


def test_reject_bad_loss_kind():
    from cardioseg.hparams import LossCfg
    with pytest.raises(ValidationError):
        LossCfg(kind="dice_ce_bogus")        # Literal enum


def test_reject_bad_spatial_dims():
    from cardioseg.hparams import ModelCfg
    with pytest.raises(ValidationError):
        ModelCfg(spatial_dims=5)             # Literal[2,3]


# --- apply_overrides: valid applies (incl tuple), invalid rejected at assignment ---
def test_override_valid_scalar_and_tuple():
    cfg = apply_overrides(TrainCfg(), ["aug.gamma_p=0.5", "data.test_vendors=('GE',)"])
    assert cfg.aug.gamma_p == 0.5 and cfg.data.test_vendors == ("GE",)


def test_override_out_of_bounds_rejected():
    with pytest.raises(ValidationError):
        apply_overrides(TrainCfg(), ["aug.gamma_p=5"])   # validate_assignment fires


# --- _coerce on Optional[str] fields (currently-None): "none"-literal -> None, else string ---
def test_override_optional_none_literal():
    cfg = apply_overrides(TrainCfg(), ["device=None"])
    assert cfg.device is None                            # not the string "None"


def test_override_optional_string_preserved():
    cfg = apply_overrides(TrainCfg(), ["device=cuda"])
    assert cfg.device == "cuda"                          # real value still passes through
