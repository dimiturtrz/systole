"""Data-build CLI dispatcher (bd cardiac-seg-ox6p): ONE entry point over the offline data/pool builders,
`python -m core.data <command> [args]`. Each command class exposes add_args/run (the strategy), so this
router is command-agnostic — no per-command branching."""
import argparse

from core.data.analysis.directed import Directed
from core.data.dynamic.anatomy import Anatomy
from core.data.dynamic.inverse import Inverse
from core.data.dynamic.mrxcat import Mrxcat
from core.data.ingest.testsets import TestSets
from core.data.static.mri.kaggle_dsb import KaggleDsbAdapter
from core.data.static.reference_build import ReferenceBuild
from core.data.static.store.build import Build
from core.obs import Obs

COMMANDS = {
    "consolidate": Build,
    "build-pool": Anatomy,
    "mrxcat": Mrxcat,
    "twin": Inverse,
    "directed": Directed,
    "reference": ReferenceBuild,
    "lock-testsets": TestSets,
    "kaggle-ef": KaggleDsbAdapter,
}


def main():
    Obs.setup()
    ap = argparse.ArgumentParser(prog="python -m core.data",
                                 description="offline data / pool builders")
    sub = ap.add_subparsers(dest="command", required=True)
    for name, cls in COMMANDS.items():
        doc = (cls.__doc__ or "").strip().split("\n")[0]
        cls.add_args(sub.add_parser(name, help=doc))
    args = ap.parse_args()
    COMMANDS[args.command].run(args)


if __name__ == "__main__":
    main()
