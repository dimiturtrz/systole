"""The store's BUILD + cloud-load engine — the heavy half that materializes processed/ and returns the
data cloud. Imports the normalization stage (and thus the preprocessing pipeline, transitively); the
read-only surface (`query`) does not, so consumers that only read npz/meta never drag preprocessing.
Consumers that build/load the cloud import from here (e.g. `from core.data.static.store import build as
store` keeps `store.load`/`store.load_cfg` working).

`load` is process-if-missing: it consolidates a dataset on first request, then reads its meta.csv. The
adapters (data/mri/*) stay raw->canonical readers; this turns them into a homogeneous, query-able store.
Splits are queries over it (core.data.ingest.splits).
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from core.config import DEFAULT_INPLANE
from core.data.static.geo import COUNTRY_CONTINENT
from core.data.static.mri.registry import get_adapter
from core.data.static.store.normalize import Normalizer
from core.data.static.store.query import (  # noqa: F401  re-exported: `build as store` keeps store.<light-symbol> working
    SOURCE_DATASETS,
    DataCfg,
    MetaBuilder,
    dataset_dir,
    load_arrays,
    param_key,
)
from core.obs import Obs
from core.preprocessing.n4 import N4Cfg

log = logging.getLogger("cardioseg.store")


def build(name: str, inplane: float = DEFAULT_INPLANE, *, n4: bool = False,  # noqa: PLR0913  low-level store primitive; config-object path is load_cfg(DataCfg)  # pragma: no cover  npz writes over real adapter volumes (preprocess_case reads NIfTI/DICOM); meta core = MetaBuilder, tested
          n4_params: N4Cfg | None = None, workers: int | None = None, rebuild: bool = False,
          nyul: bool = False, nyul_standard=None, norm: str = "zscore") -> Path:
    """Consolidate one dataset into processed/<name>/<paramkey>/ (data/*.npz + meta.csv).

    Process-if-missing: skips subjects already written; re-emits meta.csv each call. Parallel
    (ThreadPool — resample/N4 release the GIL). `nyul`+`nyul_standard` apply Nyúl harmonization
    (qfz) to a separate _nyul cache. Returns the processed dir.
    """
    out = dataset_dir(name, inplane, n4=n4, n4_params=n4_params, nyul=nyul, norm=norm)
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    adapter = get_adapter(name)
    cases = adapter.cases()
    todo = cases if rebuild else [c for c in cases if not (data_dir / f"{c.name}.npz").exists()]
    normalizer = Normalizer(inplane, n4=n4, n4_params=n4_params, nyul=nyul, nyul_standard=nyul_standard, norm=norm)

    def _one(case: Path):
        arrays = normalizer.apply_case(case, adapter.load_ed_es)
        npz = {k: v for k, v in arrays.items() if k != "patient"}
        np.savez_compressed(data_dir / f"{case.name}.npz", **npz)

    if todo:
        workers = workers or max(1, (os.cpu_count() or 4) - 2)
        log.info("consolidating %s: %d subjects -> %s (%d threads, n4=%s)", name, len(todo), out, workers, n4)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for _ in Obs.progress(ex.map(_one, todo), f"consolidate {name}", total=len(todo)):
                pass

    MetaBuilder(name, adapter).write(data_dir, out)          # (re)emit meta.csv (sidecar parse, no reload)
    return out


def load(names: list[str] | str | None = None, inplane: float = DEFAULT_INPLANE, *,  # noqa: PLR0913  low-level store primitive; config-object path is load_cfg(DataCfg)
         n4: bool = False, n4_params: N4Cfg | None = None, workers: int | None = None,
         nyul: bool = False, norm: str = "zscore") -> pl.DataFrame:
    """Ensure each requested dataset is consolidated, then return ONE polars frame over all of them
    (the data cloud, for these params). Adds an absolute `path` column to each npz. names=None -> all.
    nyul=True harmonizes to the cohort standard (reference/nyul.yaml; fit it first with
    `python -m core.data.static.store --fit-nyul`)."""
    names = SOURCE_DATASETS if names is None else ([names] if isinstance(names, str) else list(names))
    std = Normalizer.load_standard() if nyul else None
    if nyul and std is None:
        raise RuntimeError("nyul=True but no reference/nyul.yaml — fit it first: "
                           "python -m core.data.static.store --fit-nyul")
    frames = []
    for name in names:
        out = dataset_dir(name, inplane, n4=n4, n4_params=n4_params, nyul=nyul, norm=norm)
        if not (out / "meta.csv").exists():
            build(name, inplane, n4=n4, n4_params=n4_params, workers=workers, nyul=nyul,  # pragma: no cover  triggers a real consolidation build (reads adapter data from disk)
                  nyul_standard=std, norm=norm)
        # Pin `labelled` to Boolean — don't rely on polars schema inference (newer polars reads the
        # "true"/"false" column as String, breaking the `pl.col('labelled')` filter cross-platform).
        df = pl.read_csv(out / "meta.csv", infer_schema_length=10000,
                         schema_overrides={"labelled": pl.Boolean})
        df = df.with_columns((pl.lit(str(out / "data")) + "/" + pl.col("file")).alias("path"))
        frames.append(df)
    cloud = pl.concat(frames, how="vertical_relaxed")
    # continent is DERIVED from country (SSOT in data/geo) — queryable column, never hand-stored.
    return cloud.with_columns(
        pl.col("country").replace_strict(COUNTRY_CONTINENT, default=None).alias("continent"))


def load_cfg(d, sources=None, workers: int | None = None) -> pl.DataFrame:
    """Load the cloud a model (DataCfg `d`) trained under — ALL its preprocessing params (inplane, n4,
    nyul, norm), not just some. Callers that pass only inplane/n4 silently read zscore npz for a nyul/
    blood-norm model (a trap). `sources` overrides d.sources (e.g. the matrix's full eval cloud)."""
    return load(list(sources if sources is not None else d.sources), inplane=d.inplane, n4=d.n4,
                n4_params=d.n4_params, nyul=d.nyul, norm=d.norm, workers=workers)
