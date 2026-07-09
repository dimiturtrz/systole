"""Mirror tests for cardioseg.pipeline — the target-driven DAG runner (bd cardiac-seg-v4j6). Covers the
pure surface: plan (dep resolution + dedup), the per-stage is_done artifact gates, Ctx.run, and the
Pipeline.run skip-and-record loop (fake stages — no GPU/IO)."""
from pathlib import Path

import pytest

from cardioseg.pipeline import (
    AnalysisStage,
    Ctx,
    EvaluateStage,
    ExportStage,
    Pipeline,
    TrainStage,
)
from core.hparams import TrainCfg


@pytest.mark.parametrize("targets, expected", [
    (["data"], ["data"]),                                  # leaf: no deps
    (["train"], ["data", "train"]),                        # chain: deps first
    (["evaluate"], ["data", "train", "evaluate"]),         # transitive chain
    (["export"], ["data", "train", "export"]),
    (["analysis"], ["data", "analysis"]),                  # opt-in branch off data
    (["evaluate", "export"], ["data", "train", "evaluate", "export"]),  # shared deps deduped once
])
def test_plan_resolves_deps_in_order(targets, expected):
    """plan = each stage after all its deps, deduped."""
    assert Pipeline.plan(targets) == expected


def _ctx(tmp: Path, seeds=(0,)) -> Ctx:
    cfg = TrainCfg()
    cfg.out_dir = str(tmp)
    return Ctx(cfg, list(seeds), None, quick=False, with_analysis=False)


def test_ctx_run_single_seed_is_base(tmp_path):
    """single seed -> the base dir itself is the run dir."""
    assert _ctx(tmp_path).run == tmp_path


def test_ctx_run_multiseed_is_sibling(tmp_path):
    """multi-seed -> seed-0 lands in a `<base>_s0` sibling."""
    assert _ctx(tmp_path, seeds=(0, 1)).run == tmp_path.parent / f"{tmp_path.name}_s0"


@pytest.mark.parametrize("stage, artifact", [
    (TrainStage(), "model.pth"),
    (EvaluateStage(), "RESULTS.json"),
    (ExportStage(), "model.onnx"),
    (AnalysisStage(), "analysis.png"),
])
def test_is_done_gates_on_artifact(tmp_path, stage, artifact):
    """is_done is False until the stage's artifact exists, then True (the idempotent gate)."""
    ctx = _ctx(tmp_path)
    assert stage.is_done(ctx) is False
    (tmp_path / artifact).write_text("x")
    assert stage.is_done(ctx) is True


def test_data_stage_never_done(tmp_path):
    """data is process-if-missing -> always 'runs' (a no-op scan when built), never gated done."""
    from cardioseg.pipeline import DataStage
    assert DataStage().is_done(_ctx(tmp_path)) is False


def test_run_skips_done_stages_and_records(monkeypatch):
    """Pipeline.run runs only not-done stages (here: pretend `data` done) and returns what it ran."""
    ran = []
    for name, st in Pipeline._STAGES.items():
        monkeypatch.setattr(st, "is_done", lambda ctx, n=name: n == "data")
        monkeypatch.setattr(st, "run", lambda ctx, n=name: ran.append(n))
    did = Pipeline.run(object(), ["evaluate"])
    assert did == ["train", "evaluate"]
    assert ran == ["train", "evaluate"]
    assert "data" not in did
