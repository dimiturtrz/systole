"""SSOT guard: the single-sourced constants must stay equal to every consumer that reads them.
If someone re-hardcodes a value, one of these fails (that's the point)."""
from core.config import (
    DEFAULT_SIZE, DEFAULT_INPLANE, KNOWN_DATASETS, FLAGSHIP_REF, flagship_model, flagship_dir,
)
from core.hparams import DataCfg


def test_size_single_sourced():
    from cardioseg.training.dataset import SIZE
    assert DEFAULT_SIZE == DataCfg().size == SIZE


def test_inplane_single_sourced():
    from core.preprocessing.preprocess import TARGET_INPLANE
    assert DEFAULT_INPLANE == DataCfg().inplane == TARGET_INPLANE


def test_dataset_vocab_single_sourced():
    from cardioseg.preprocessing.normalization.persist import _DATASETS
    assert KNOWN_DATASETS == DataCfg().sources == _DATASETS


def test_flagship_is_registry_ref():
    """Flagship resolves from the mlflow registry (production alias), not a hardcoded dir. Don't
    invoke resolve here (that hits mlflow) — just guard the ref + that the helpers are callable."""
    assert FLAGSHIP_REF == "production"
    assert callable(flagship_model) and callable(flagship_dir)


def test_windows_path_on_posix_fails_loud(monkeypatch):
    """Leak guard: a Windows drive path ('D:/…') on POSIX must RAISE, not silently create repo/D:/…
    (that leaked dataset meta.csv into git once). On Windows the same path is a valid absolute root."""
    import core.config as cfg
    import pytest
    monkeypatch.setenv("CARDIAC_DATA", "D:/data/volumetric/mri")
    monkeypatch.setattr(cfg.os, "name", "posix")
    with pytest.raises(RuntimeError, match="POSIX"):
        cfg._root()
    monkeypatch.setattr(cfg.os, "name", "nt")
    assert cfg._root() == "D:/data/volumetric/mri"     # valid absolute root on Windows


def test_cmrxmotion_held_out_by_default():
    """cmrxmotion is single-vendor Siemens — if it's a known source but NOT held out, it would
    silently pollute Siemens train. Guard the invariant: wired AND held out by dataset."""
    assert "cmrxmotion" in KNOWN_DATASETS
    assert "cmrxmotion" in DataCfg().test_datasets
