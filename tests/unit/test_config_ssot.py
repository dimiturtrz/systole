"""SSOT guard: the single-sourced constants must stay equal to every consumer that reads them.
If someone re-hardcodes a value, one of these fails (that's the point)."""
from cardioseg.config import (
    DEFAULT_SIZE, DEFAULT_INPLANE, KNOWN_DATASETS, FLAGSHIP_RUN, flagship_model,
)
from cardioseg.hparams import DataCfg


def test_size_single_sourced():
    from cardioseg.training.dataset import SIZE
    assert DEFAULT_SIZE == DataCfg().size == SIZE


def test_inplane_single_sourced():
    from cardioseg.preprocessing.preprocess import TARGET_INPLANE
    assert DEFAULT_INPLANE == DataCfg().inplane == TARGET_INPLANE


def test_dataset_vocab_single_sourced():
    from cardioseg.preprocessing.normalization.persist import _DATASETS
    assert KNOWN_DATASETS == DataCfg().sources == _DATASETS


def test_flagship_model_path():
    assert flagship_model() == f"{FLAGSHIP_RUN}/model.pth"
