"""Export CLI dispatcher (bd cardiac-seg-6e1a): ONE entry point over the artifact exporters,
`python -m core.export <onnx|mesh> [args]`. Each command class exposes add_args/run (the strategy)."""
import argparse

from core.export_onnx import ExportOnnx
from core.mesh import Mesh
from core.obs import Obs

COMMANDS = {"onnx": ExportOnnx, "mesh": Mesh}


def main():
    Obs.setup()
    ap = argparse.ArgumentParser(prog="python -m core.export", description="artifact exporters")
    sub = ap.add_subparsers(dest="command", required=True)
    for name, cls in COMMANDS.items():
        doc = (cls.__doc__ or "").strip().split("\n")[0]
        cls.add_args(sub.add_parser(name, help=doc))
    args = ap.parse_args()
    COMMANDS[args.command].run(args)


if __name__ == "__main__":
    main()
