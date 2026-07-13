"""Pure train.py orchestration helpers (equivalence classes) — the arg->cfg mapping, seed resolution,
split/output-dir policy pulled out of the GPU training loop. The .backward() loop needs a GPU+dataset
(shell, pragma'd); everything HERE is deterministic string/dict/path logic, testable off-GPU.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cardioseg.training.train import Train
from core.hparams import TrainCfg

_val_dice = Train.val_dice
apply_cli_args = Train.apply_cli_args
check_multiseed_split = Train.check_multiseed_split
n_train_of = Train.n_train_of
parse_seeds = Train.parse_seeds
resolve_seeds = Train.resolve_seeds
seed_out_dir = Train.seed_out_dir
split_tag_of = Train.split_tag_of


# --- parse_seeds: None/empty class vs csv class ---
def test_parse_seeds_none_and_empty_return_none():
    """No --seeds (None or '') -> None, so train_seg falls back to the single cfg.seed."""
    assert parse_seeds(None) is None
    assert parse_seeds("") is None


def test_parse_seeds_csv():
    """A comma list -> the int seed list, order preserved."""
    assert parse_seeds("0,1,2") == [0, 1, 2]
    assert parse_seeds("7") == [7]


# --- resolve_seeds: explicit list vs fallback ---
def test_resolve_seeds_fallback_to_cfg_when_empty():
    """seeds falsy -> [cfg_seed] (single-seed default)."""
    assert resolve_seeds(3, None) == [3]
    assert resolve_seeds(3, []) == [3]


def test_resolve_seeds_uses_explicit():
    """seeds given -> that list, cfg_seed ignored."""
    assert resolve_seeds(3, [0, 1]) == [0, 1]


# --- check_multiseed_split: the invalid combo raises, valid ones no-op ---
def test_multiseed_without_split_raises():
    """>1 seed + no coded split = illegal (legacy splits key on seed); ValueError."""
    with pytest.raises(ValueError, match="multi-seed needs a coded"):
        check_multiseed_split([0, 1], None)


def test_multiseed_with_split_ok():
    """>1 seed WITH a coded split is fine — no raise."""
    check_multiseed_split([0, 1], "static_main")


def test_single_seed_needs_no_split():
    """1 seed never needs a split, regardless of split being None."""
    check_multiseed_split([0], None)


# --- seed_out_dir: single-seed shares base, multi-seed suffixes ---
def test_seed_out_dir_single_is_base():
    """single=True -> the shared base dir verbatim (no _s suffix)."""
    assert seed_out_dir(Path("/x/run"), 5, single=True) == Path("/x/run")


def test_seed_out_dir_multi_suffixes_sibling():
    """single=False -> `<base>_s<seed>` as a sibling of base."""
    assert seed_out_dir(Path("/x/run"), 5, single=False) == Path("/x/run_s5")


# --- split_tag_of: coded split | vendor-join | 'legacy' fallback chain ---
def test_split_tag_prefers_coded_split():
    """A coded d.split wins the tag."""
    assert split_tag_of(SimpleNamespace(split="static_main@1", test_vendors=("GE",))) == "static_main@1"


def test_split_tag_joins_vendors_when_no_split():
    """No split -> '+'.join(test_vendors)."""
    assert split_tag_of(SimpleNamespace(split=None, test_vendors=("Canon", "GE"))) == "Canon+GE"


def test_split_tag_legacy_when_nothing():
    """No split, no vendors -> 'legacy'."""
    assert split_tag_of(SimpleNamespace(split=None, test_vendors=())) == "legacy"


# --- n_train_of: dynamic (slice-count) vs static/legacy (row-count) ---
def test_n_train_dynamic_uses_gen_slices():
    """A dynamic OR composite train source has no patient frame -> count resident slices (gen.n)."""
    gen = SimpleNamespace(n=128)
    assert n_train_of(SimpleNamespace(kind="dynamic"), gen, train_df=[]) == 128
    assert n_train_of(SimpleNamespace(kind="composite"), gen, train_df=None) == 128


def test_n_train_static_uses_frame_rows():
    """A static source -> patient-row count of the train frame."""
    src = SimpleNamespace(kind="static")
    gen = SimpleNamespace(X=SimpleNamespace(shape=(999,)))
    assert n_train_of(src, gen, train_df=[1, 2, 3]) == 3


def test_n_train_legacy_none_source_uses_frame():
    """Legacy path (train_src None) -> frame rows."""
    gen = SimpleNamespace(X=SimpleNamespace(shape=(999,)))
    assert n_train_of(None, gen, train_df=[1, 2]) == 2


# --- apply_cli_args: set scalars win, unset skipped, --set applied last ---
def _args(**kw):
    base = dict(epochs=None, batch=None, patience=None, workers=None, seed=None, n_patients=None,
                ef_lambda=None, n4=False, ef_learn=False, ef_kaggle=False, out=None, overrides=[])
    base.update(kw)
    return SimpleNamespace(**base)


def test_apply_cli_args_sets_provided_scalars():
    """Provided scalar args copy onto the cfg; None args leave the default untouched."""
    cfg = TrainCfg()
    d_epochs = cfg.epochs
    apply_cli_args(cfg, _args(batch=8, seed=3))
    assert cfg.batch == 8 and cfg.seed == 3 and cfg.epochs == d_epochs


def test_apply_cli_args_store_flags():
    """The boolean store-flags flip the right nested cfg fields."""
    cfg = TrainCfg()
    apply_cli_args(cfg, _args(n4=True, ef_learn=True, ef_kaggle=True, out="runs/x"))
    assert cfg.generator.data.n4 is True and cfg.ef_learn is True
    assert cfg.ef_kaggle is True and cfg.out_dir == "runs/x"


def test_apply_cli_args_overrides_applied_last():
    """--set overrides win over the scalar mapping (deep override, applied after)."""
    cfg = TrainCfg()
    apply_cli_args(cfg, _args(overrides=["generator.aug.gamma_p=0.5"]))
    assert cfg.generator.aug.gamma_p == 0.5


def test_apply_cli_args_returns_same_cfg():
    """Maps IN PLACE and returns the same object (composition-root convenience)."""
    cfg = TrainCfg()
    assert apply_cli_args(cfg, _args()) is cfg


def test_apply_cli_args_accepts_plain_dict():
    """Input-type class: a plain dict works like a Namespace (vars-normalized, no getattr)."""
    cfg = TrainCfg()
    apply_cli_args(cfg, {"batch": 16, "n4": True})
    assert cfg.batch == 16 and cfg.generator.data.n4 is True


# --- _val_dice: the early-stop Dice signal (pure batched tensor math; a fixed-logit model stands in
#     for the U-Net, so the GPU is not needed). Foreground classes = (1,2,3), bg=0. ---
class _EchoModel(torch.nn.Module):
    """argmax(forward(x)) == the class map carried in x[:,0] — a deterministic stand-in for the U-Net,
    so _val_dice's Dice accumulation is testable batch-by-batch (each call sees its own x). x[:,0] holds
    the intended per-pixel class as a float."""

    def __init__(self, n_classes=4):
        super().__init__()
        self.n = n_classes

    def forward(self, x):
        cls = x[:, 0].round().long()                        # [B,H,W] intended pred
        logits = torch.zeros(x.shape[0], self.n, *cls.shape[1:])
        for c in range(self.n):
            logits[:, c] = (cls == c).float() * 10.0
        return logits


def _x_from_pred(pred):
    return pred.float()[:, None]                            # [B,1,H,W] carrying the class map


def test_val_dice_perfect_prediction_is_one():
    """Match class: prediction == GT on every foreground class -> mean Dice 1.0 (pooled over slices)."""
    y = torch.zeros(2, 4, 4, dtype=torch.long)
    y[0, :2, :2] = 1; y[0, 2:, 2:] = 2; y[1, :2] = 3
    assert abs(_val_dice(_EchoModel(), _x_from_pred(y), y, batch=1, device="cpu") - 1.0) < 1e-6


def test_val_dice_absent_class_scores_zero():
    """Empty-denominator class: a foreground class in neither pred nor GT contributes 0 (no div-by-0)."""
    # GT + pred use only class 1; classes 2,3 absent -> their Dice defined as 0, dragging the mean.
    y = torch.ones(1, 4, 4, dtype=torch.long)
    d = _val_dice(_EchoModel(), _x_from_pred(y), y, batch=1, device="cpu")
    assert abs(d - 1.0 / 3) < 1e-6      # class1 Dice 1, classes 2&3 Dice 0 -> mean 1/3


def test_val_dice_batching_invariant():
    """Batch class: the pooled Dice is independent of the batch size it's accumulated over."""
    y = torch.zeros(4, 4, 4, dtype=torch.long)
    y[:, :2, :2] = 1; y[:, 2:, 2:] = 2
    x = _x_from_pred(y)
    d1 = _val_dice(_EchoModel(), x, y, batch=1, device="cpu")
    d4 = _val_dice(_EchoModel(), x, y, batch=4, device="cpu")
    assert abs(d1 - d4) < 1e-6
