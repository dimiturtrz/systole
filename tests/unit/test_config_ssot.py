"""SSOT (single source of truth) guard: the single-sourced constants must stay equal to every consumer that reads them.
If someone re-hardcodes a value, one of these fails (that's the point)."""
from core.config import (
    DEFAULT_INPLANE,
    DEFAULT_SIZE,
    FLAGSHIP_REF,
    KNOWN_DATASETS,
)
from core.data.static.store import DataCfg


def test_size_single_sourced():
    from core.data.dynamic.dataset import SIZE
    assert DEFAULT_SIZE == DataCfg().size == SIZE


def test_inplane_single_sourced():
    from core.preprocessing.preprocess import TARGET_INPLANE
    assert DEFAULT_INPLANE == DataCfg().inplane == TARGET_INPLANE


def test_dataset_vocab_single_sourced():
    from cardioseg.preprocessing.normalization.persist import _DATASETS
    assert KNOWN_DATASETS == DataCfg().sources == _DATASETS


def test_flagship_is_registry_ref():
    """Flagship resolves from the mlflow registry (production alias), not a hardcoded dir. Don't
    invoke resolve here (that hits mlflow) — just guard the ref."""
    assert FLAGSHIP_REF == "production"


def test_root_delegates_to_resolver(monkeypatch):
    """config._root reads CARDIAC_DATA then adapts it via core.paths.resolve_data_root (the full
    OS×path matrix incl. the Windows-on-POSIX leak guard is tested in test_paths). Here: it picks up
    the env + the resolver leaves an OS-native path unchanged."""
    import core.config as cfg
    monkeypatch.setenv("CARDIAC_DATA", "/data/foo/bar")
    assert "foo/bar" in cfg._root()                    # env honored + resolved (no crash)


def test_cmrxmotion_held_out_by_default():
    """cmrxmotion is single-vendor Siemens — if it's a known source but NOT held out, it would
    silently pollute Siemens train. Guard the invariant: wired AND held out by dataset."""
    assert "cmrxmotion" in KNOWN_DATASETS
    assert "cmrxmotion" in DataCfg().test_datasets
