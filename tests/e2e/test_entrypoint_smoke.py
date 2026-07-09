"""Entry-point smoke (tier-1, minimal): every CLI `main` imports + builds its ArgumentParser + reaches
`--help` without crashing. The cheap guard that the everything-in-a-class migration (bd cardiac-seg-y2fi)
didn't break CLI wiring — `--help` short-circuits argparse BEFORE any real work, so it exercises the
import + module-level class-dispatch + parser setup with no data/GPU. In-process via runpy (no subprocess
spawn) so it stays fast.

Tier-2 (deep e2e: run each main with mocked I/O + assert output) is the incremental fill under bd j4m2.
The 28->5 entry-point consolidation (bd axri) folded the eval/data/analysis/export one-offs into group
dispatchers — each `python -m <group> <subcommand>` shares one argparse router.
"""
import runpy
import sys

import pytest

# The 5 CLI entry points -> `python -m <mod> --help` exits 0 (SystemExit) after building the parser.
# The group dispatchers (cardioseg.evaluation, core.data, core.data.analysis) run their package __main__.py.
_ARGPARSE_ENTRYPOINTS = [
    "cardioseg.evaluation",
    "cardioseg.training.train",
    "core.data",
    "core.data.analysis",
    "core.export",
]

@pytest.mark.parametrize("mod", _ARGPARSE_ENTRYPOINTS)
def test_entrypoint_help_exits_clean(mod, monkeypatch):
    """`python -m <mod> --help` builds the parser and exits 0 — import + CLI dispatch wiring intact."""
    monkeypatch.setattr(sys, "argv", [mod.split(".")[-1], "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module(mod, run_name="__main__")
    assert exc.value.code in (0, None)   # argparse --help exits 0
