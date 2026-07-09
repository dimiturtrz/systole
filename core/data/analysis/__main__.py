"""Analysis/diagnostic CLI dispatcher (bd cardiac-seg-9c5x): ONE entry point over the synth-vs-real
diagnostics, `python -m core.data.analysis <command> [args]`. Each command class exposes add_args/run
(the strategy) — router is command-agnostic, no per-command branching."""
import argparse

from core.data.analysis.attribution import Attribution
from core.data.analysis.eda import Eda
from core.data.analysis.render import Render
from core.data.analysis.shape_coverage import ShapeCoverage
from core.data.analysis.sim2real import Sim2Real
from core.data.analysis.static_compare import StaticCompare
from core.data.analysis.synth_fidelity import SynthFidelity
from core.obs import Obs

COMMANDS = {
    "attribution": Attribution, "eda": Eda, "render": Render, "shape-coverage": ShapeCoverage,
    "sim2real": Sim2Real, "static-compare": StaticCompare, "synth-fidelity": SynthFidelity,
}


def main():
    Obs.setup()
    ap = argparse.ArgumentParser(prog="python -m core.data.analysis",
                                 description="synth-vs-real diagnostics")
    sub = ap.add_subparsers(dest="command", required=True)
    for name, cls in COMMANDS.items():
        doc = (cls.__doc__ or "").strip().split("\n")[0]
        cls.add_args(sub.add_parser(name, help=doc))
    args = ap.parse_args()
    COMMANDS[args.command].run(args)


if __name__ == "__main__":
    main()
