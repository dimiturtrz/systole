"""Entry-point smoke (tier-1, minimal): every CLI `main` imports + builds its ArgumentParser + reaches
`--help` without crashing. The cheap guard that the everything-in-a-class migration (bd cardiac-seg-y2fi)
didn't break CLI wiring — `--help` short-circuits argparse BEFORE any real work, so it exercises the
import + module-level class-dispatch + parser setup with no data/GPU. In-process via runpy (no subprocess
spawn) so it stays fast.

Tier-2 (deep e2e: run each main with mocked I/O + assert output) is the incremental fill under bd j4m2 —
plus a standing note that 28 entry points is probably too many (CLI consolidation, future). The two mains
without argparse (`sync_numbers`, `render`) can't be `--help`-smoked; they're xfail here pending a mocked
tier-2 smoke.
"""
import runpy
import sys

import pytest

# The 26 CLI entry points that use argparse -> `python -m <mod> --help` exits 0 (SystemExit) after building
# the parser. `core.data.static.store` runs its package __main__.py.
_ARGPARSE_ENTRYPOINTS = [
    "cardioseg.evaluation.calibrate",
    "cardioseg.evaluation.distribution",
    "cardioseg.evaluation.ensemble",
    "cardioseg.evaluation.matrix",
    "cardioseg.evaluation.modelcard",
    "cardioseg.evaluation.overlay",
    "cardioseg.evaluation.results",
    "cardioseg.evaluation.soft_eval",
    "cardioseg.evaluation.uncertainty",
    "cardioseg.preprocessing.normalization.persist",
    "cardioseg.training.train",
    "core.data.analysis.attribution",
    "core.data.analysis.eda",
    "core.data.analysis.shape_coverage",
    "core.data.analysis.sim2real",
    "core.data.analysis.static_compare",
    "core.data.analysis.synth_fidelity",
    "core.data.dynamic.anatomy",
    "core.data.dynamic.inverse",
    "core.data.dynamic.mrxcat",
    "core.data.ingest.testsets",
    "core.data.static.mri.kaggle_dsb",
    "core.data.static.reference_build",
    "core.data.static.store",
    "core.export_onnx",
    "core.mesh",
]

# No argparse -> main() runs real work, can't be --help-smoked. Tier-2 (mocked e2e) tracked in bd j4m2.
_NO_ARGPARSE_ENTRYPOINTS = ["cardioseg.evaluation.sync_numbers", "core.data.analysis.render"]


@pytest.mark.parametrize("mod", _ARGPARSE_ENTRYPOINTS)
def test_entrypoint_help_exits_clean(mod, monkeypatch):
    """`python -m <mod> --help` builds the parser and exits 0 — import + CLI dispatch wiring intact."""
    monkeypatch.setattr(sys, "argv", [mod.split(".")[-1], "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module(mod, run_name="__main__")
    assert exc.value.code in (0, None)   # argparse --help exits 0


@pytest.mark.parametrize("mod", _NO_ARGPARSE_ENTRYPOINTS)
@pytest.mark.xfail(reason="no argparse --help; needs a mocked tier-2 e2e (bd cardiac-seg-j4m2)", strict=False)
def test_entrypoint_no_argparse_pending_tier2(mod, monkeypatch):
    monkeypatch.setattr(sys, "argv", [mod.split(".")[-1], "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module(mod, run_name="__main__")
    assert exc.value.code in (0, None)
