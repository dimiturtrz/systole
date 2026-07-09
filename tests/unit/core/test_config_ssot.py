"""SSOT (single source of truth) guard: single-sourced constants must stay equal to every consumer
that reads them. If someone re-hardcodes a value, one of these fails (that's the point). These are
CROSS-MODULE invariants (config <-> dataset/preprocess/persist), not tests of core.config itself —
the Config resolver + flagship ref are mirrored in test_config.py.
"""
from cardioseg.preprocessing.normalization.persist import _DATASETS
from core.config import (
    DEFAULT_INPLANE,
    DEFAULT_SIZE,
    KNOWN_DATASETS,
)
from core.data.dynamic.dataset import SIZE
from core.data.static.store import DataCfg
from core.preprocessing.preprocess import TARGET_INPLANE


def test_size_single_sourced():
    assert DEFAULT_SIZE == DataCfg().size == SIZE


def test_inplane_single_sourced():
    assert DEFAULT_INPLANE == DataCfg().inplane == TARGET_INPLANE


def test_dataset_vocab_single_sourced():
    assert KNOWN_DATASETS == DataCfg().sources == _DATASETS


def test_cmrxmotion_held_out_by_default():
    """cmrxmotion is single-vendor Siemens — if it's a known source but NOT held out, it would
    silently pollute Siemens train. Guard the invariant: wired AND held out by dataset."""
    assert "cmrxmotion" in KNOWN_DATASETS
    assert "cmrxmotion" in DataCfg().test_datasets
