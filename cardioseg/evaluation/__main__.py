"""Evaluation + reporting CLI dispatcher (bd cardiac-seg-pyxe): ONE entry point over the eval commands,
`python -m cardioseg.evaluation <command> [args]`. Each command class exposes add_args/run (the strategy),
so this router is command-agnostic — no per-command branching."""
import argparse

from cardioseg.preprocessing.normalization.persist import Persist
from core.obs import Obs

from .calibrate import Calibrate
from .distribution import Distribution
from .ensemble import Ensemble
from .matrix import Matrix
from .modelcard import ModelCard
from .overlay import Overlay
from .results import Results
from .soft_eval import SoftEval
from .sync_numbers import SyncNumbers
from .uncertainty import Uncertainty

COMMANDS = {
    "calibrate": Calibrate, "distribution": Distribution, "ensemble": Ensemble,
    "matrix": Matrix, "modelcard": ModelCard, "overlay": Overlay, "results": Results,
    "soft_eval": SoftEval, "sync_numbers": SyncNumbers, "uncertainty": Uncertainty,
    "persist": Persist,
}


def main():
    Obs.setup()
    ap = argparse.ArgumentParser(prog="python -m cardioseg.evaluation",
                                 description="cardioseg evaluation + reporting commands")
    sub = ap.add_subparsers(dest="command", required=True)
    for name, cls in COMMANDS.items():
        doc = (cls.__doc__ or "").strip().split("\n")[0]
        cls.add_args(sub.add_parser(name, help=doc))
    args = ap.parse_args()
    COMMANDS[args.command].run(args)


if __name__ == "__main__":
    main()
