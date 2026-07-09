"""CLI: consolidate datasets into processed/<ds>/<paramkey>/, and the reference-fitting one-shots
(Nyúl standard, acquisition mine, meta migration). `python -m core.data.static.store [--names …] …`."""
import argparse
import logging

import polars as pl

from core.config import DEFAULT_INPLANE
from core.data.static.store.build import load
from core.data.static.store.normalize import Normalizer
from core.data.static.store.query import AcqReference, MetaBuilder
from core.obs import setup

log = logging.getLogger("cardioseg.store")


def main():
    setup()
    ap = argparse.ArgumentParser(description="consolidate datasets into processed/<ds>/<paramkey>/")
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
    args = ap.parse_args()
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
    df = load(args.names, inplane=args.inplane, n4=args.n4, nyul=args.nyul)
    log.info(f"\n=== data cloud: {len(df)} subjects ===")
    log.info(df.group_by("dataset").agg(pl.len().alias("n"),
          pl.col("labelled").sum().alias("labelled")).sort("dataset"))
    log.info(df.group_by("vendor").agg(pl.len().alias("n")).sort("n", descending=True))


if __name__ == "__main__":
    main()
