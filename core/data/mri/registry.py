"""Dataset adapter registry — name -> adapter. Add a dataset = one file + one line here.

Lifted out of training.train so every entrypoint (train, validate, distribution, exporters,
the normalization parser) shares one source of truth.
"""
from core.data.mri.base import DatasetAdapter
from core.data.mri.acdc import AcdcAdapter
from core.data.mri.mnm2 import Mnm2Adapter
from core.data.mri.mnms1 import Mnms1Adapter
from core.data.mri.cmrxmotion import CmrxMotionAdapter

_ADAPTERS: dict[str, DatasetAdapter] = {
    a.name: a for a in (AcdcAdapter(), Mnm2Adapter(), Mnms1Adapter(), CmrxMotionAdapter())
}


def get_adapter(name: str) -> DatasetAdapter:
    """Adapter for a dataset name (acdc / mnm2 / mnms1)."""
    if name not in _ADAPTERS:
        raise KeyError(f"unknown dataset {name!r}; have {sorted(_ADAPTERS)}")
    return _ADAPTERS[name]


def dataset_names() -> list[str]:
    return sorted(_ADAPTERS)
