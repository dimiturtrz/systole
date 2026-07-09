"""Consolidated data store: scans + metadata as one homogeneous, query-able thing per dataset, mirroring
raw/ into processed/<dataset>/<paramkey>/ (data/*.npz + meta.csv). Concatenating the meta.csv of the
datasets you ask for *is* the data cloud — no separate inventory.

Split by DEPENDENCY WEIGHT so the many read-only consumers don't drag the preprocessing pipeline:

  query      READ + metadata surface — preprocessing-free (DataCfg, load_arrays, param_key, MetaBuilder…)
  normalize  the normalization stage — OWNS N4/Nyúl/resample
  build      BUILD + cloud-load engine — imports normalize (heavy); `load`, `load_cfg`, `build`

This package re-exports only the LIGHT `query` surface, so `from core.data.static.store import DataCfg`
stays preprocessing-free. Cloud builders import the heavy half explicitly:
`from core.data.static.store import build as store` (keeps `store.load` / `store.load_cfg` working).
"""
from core.data.static.store.query import (
    META_FIELDS,
    SOURCE_DATASETS,
    AcqReference,
    DataCfg,
    MetaBuilder,
    Store,
)

__all__ = ["META_FIELDS", "SOURCE_DATASETS", "AcqReference", "DataCfg", "MetaBuilder", "Store"]
