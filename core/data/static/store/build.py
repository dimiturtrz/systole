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
from core.data.static.mri.registry import AdapterRegistry
from core.data.static.store.normalize import Normalizer
from core.data.static.store.query import (  # noqa: F401  re-exported: `Build as store` keeps store.<light-symbol> working
    SOURCE_DATASETS,
    AcqReference,
    DataCfg,
    MetaBuilder,
    Recipe,
    Store,
)
from core.obs import Obs

log = logging.getLogger("cardioseg.store")


class Build:
    """The store's BUILD + cloud-load engine as staticmethods (the free build funcs folded in). Consumers
    alias `from core.data.static.store.build import Build as store` -> `store.load`/`store.load_cfg` (and
    the re-exported light surface `store.load_arrays`) keep working."""
    load_arrays = staticmethod(Store.load_arrays)

    @staticmethod
    def build(  # pragma: no cover  npz writes over real adapter volumes (preprocess_case reads NIfTI/DICOM)
        name: str, recipe: Recipe | None = None, *, workers: int | None = None,
        rebuild: bool = False, nyul_standard=None,
    ) -> Path:
        """Consolidate one dataset into processed/<name>/<paramkey>/ (data/*.npz + meta.csv).

        Process-if-missing: skips subjects already written; re-emits meta.csv each call. Parallel
        (ThreadPool — resample/N4 release the GIL). A `nyul` recipe + `nyul_standard` apply Nyúl
        harmonization (qfz) to a separate _nyul cache. Returns the processed dir.
        """
        recipe = recipe or Recipe()
        out = Store(recipe).dataset_dir(name)
        data_dir = out / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        adapter = AdapterRegistry.get_adapter(name)
        cases = adapter.cases()
        todo = cases if rebuild else [c for c in cases if not (data_dir / f"{c.name}.npz").exists()]
        normalizer = Normalizer(recipe, nyul_standard)

        def _one(case: Path):
            arrays = normalizer.apply_case(case, adapter.load_ed_es)
            npz = {k: v for k, v in arrays.items() if k != "patient"}
            np.savez_compressed(data_dir / f"{case.name}.npz", **npz)

        if todo:
            workers = workers or max(1, (os.cpu_count() or 4) - 2)
            log.info("consolidating %s: %d subjects -> %s (%d threads, n4=%s)",
                     name, len(todo), out, workers, recipe.n4)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for _ in Obs.progress(ex.map(_one, todo), f"consolidate {name}", total=len(todo)):
                    pass

        MetaBuilder(name, adapter).write(data_dir, out)          # (re)emit meta.csv (sidecar parse, no reload)
        return out

    @staticmethod
    def load(names: list[str] | str | None = None, recipe: Recipe | None = None, *,
             workers: int | None = None) -> pl.DataFrame:
        """Ensure each requested dataset is consolidated, then return ONE polars frame over all of them
        (the data cloud, for this recipe). Adds an absolute `path` column to each npz. names=None -> all.
        A `nyul` recipe harmonizes to the cohort standard (reference/nyul.yaml; fit it first with
        `python -m core.data consolidate --fit-nyul`)."""
        names = SOURCE_DATASETS if names is None else ([names] if isinstance(names, str) else list(names))
        recipe = recipe or Recipe()
        std = Normalizer.load_standard() if recipe.nyul else None
        if recipe.nyul and std is None:
            raise RuntimeError("nyul=True but no reference/nyul.yaml — fit it first: "
                               "python -m core.data consolidate --fit-nyul")
        frames = []
        store = Store(recipe)
        for name in names:
            out = store.dataset_dir(name)
            if not (out / "meta.csv").exists():
                Build.build(name, recipe, workers=workers, nyul_standard=std)  # pragma: no cover  real build from disk
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

    @staticmethod
    def load_cfg(d, sources=None, workers: int | None = None) -> pl.DataFrame:
        """Load the cloud a model (DataCfg `d`) trained under — ALL its preprocessing params (inplane, n4,
        nyul, norm), not just some. Callers that pass only inplane/n4 silently read zscore npz for a nyul/
        blood-norm model (a trap). `sources` overrides d.sources (e.g. the matrix's full eval cloud)."""
        return Build.load(list(sources if sources is not None else d.sources), d.recipe, workers=workers)

    @staticmethod
    def add_args(ap):
        ap.add_argument("--names", nargs="*", default=None, help="datasets (default: all)")
        ap.add_argument("--inplane", type=float, default=DEFAULT_INPLANE)
        ap.add_argument("--n4", action="store_true")
        ap.add_argument("--nyul", action="store_true", help="harmonize to the Nyúl standard (fit it first)")
        ap.add_argument("--fit-nyul", action="store_true", dest="fit_nyul",
                        help="fit the Nyúl standard from the cohort -> reference/nyul.yaml, then exit")
        ap.add_argument("--migrate-meta", action="store_true", dest="migrate_meta",
                        help="re-emit meta.csv for built stores with the current schema/adapter.meta() "
                             "(no image reload) — run after adding metadata fields, then exit")
        ap.add_argument("--fit-acquisition", action="store_true", dest="fit_acq",
                        help="mine real per-(vendor,field) TR/TE/flip from built DICOM stores -> "
                             "reference/acquisition.yaml (acquisition_for then overrides the derivation), then exit")

    @staticmethod
    def run(args):
        if args.fit_nyul:
            std = Normalizer.fit_standard(args.names, inplane=args.inplane)
            log.info(f"fit Nyúl standard -> {Normalizer.ref_path()}\n  {[round(float(v), 3) for v in std]}")
            raise SystemExit
        if args.fit_acq:
            acq = AcqReference.fit()
            log.info(f"fit acquisition reference -> reference/acquisition.yaml\n  {list(acq)}")
            raise SystemExit
        if args.migrate_meta:
            paths = MetaBuilder.migrate(args.names)
            log.info(f"migrated meta.csv (current schema, no image reload) for {len(paths)} store(s):")
            for p in paths:
                log.info(f"  {p}")
            raise SystemExit
        df = Build.load(args.names, Recipe(inplane=args.inplane, n4=args.n4, nyul=args.nyul))
        log.info(f"\n=== data cloud: {len(df)} subjects ===")
        log.info(df.group_by("dataset").agg(pl.len().alias("n"),
              pl.col("labelled").sum().alias("labelled")).sort("dataset"))
        log.info(df.group_by("vendor").agg(pl.len().alias("n")).sort("n", descending=True))
