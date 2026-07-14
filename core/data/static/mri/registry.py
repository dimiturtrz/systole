"""Dataset adapter registry — name -> adapter. Add a dataset = one file + one line here.

Lifted out of training.train so every entrypoint (train, validate, distribution, exporters,
the normalization parser) shares one source of truth.
"""
from core.data.static.mri.acdc import AcdcAdapter
from core.data.static.mri.base import Dataset, DatasetAdapter
from core.data.static.mri.cmrxmotion import CmrxMotionAdapter
from core.data.static.mri.mnm2 import Mnm2Adapter
from core.data.static.mri.mnms1 import Mnms1Adapter
from core.data.static.mri.scd import ScdAdapter

# The wired 4-class segmentation cohort (canonical RV/myo/LV-cav): the DataCfg.sources default + the
# testsets SEG scope. SCD (seg_lv) and the EF/synthetic sets are deliberately not here. Dataset itself
# lives in base (cycle-free so the adapters can name themselves Dataset.X); re-exposed here as the
# registry is the name->adapter authority.
SEG_DATASETS: tuple[Dataset, ...] = (Dataset.ACDC, Dataset.MNM2, Dataset.MNMS1, Dataset.CMRXMOTION)

_ADAPTERS: dict[str, DatasetAdapter] = {
    a.name: a for a in (AcdcAdapter(), Mnm2Adapter(), Mnms1Adapter(), CmrxMotionAdapter(), ScdAdapter())
}


class AdapterRegistry:
    """Dataset adapter registry (staticmethod holder); named AdapterRegistry to avoid clashing with
    core.registry.Registry (the mlflow model registry)."""

    @staticmethod
    def get_adapter(name: str) -> DatasetAdapter:
        """Adapter for a dataset name (acdc / mnm2 / mnms1)."""
        if name not in _ADAPTERS:
            raise KeyError(f"unknown dataset {name!r}; have {sorted(_ADAPTERS)}")
        return _ADAPTERS[name]
