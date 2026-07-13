"""Target-driven pipeline over the entry-point nodes (bd cardiac-seg-v4j6): one command materializes a
target from nothing — `python -m cardioseg.pipeline --target evaluate`.

The DAG is `data -> [analysis] -> train -> {evaluate, export}`. Each stage is idempotent (an `is_done`
gate over its on-disk artifact), so the runner does ONLY what's missing and is resumable. PULL, not push:
training DEMANDS data (the store is process-if-missing, so `data` materializes on request); a node never
triggers an upstream one. This LAYERS over the five dispatchers — it does not replace them; each node
stays independently runnable. The payoff of the add_args/run split (bd axri) is that a stage calls the
node's DOMAIN method in-process (Build.load_cfg / Train.train_seg / Results.build / ExportOnnx.export),
no subprocess, no argparse round-trip.

    python -m cardioseg.pipeline --target evaluate --split static_main --alias production
    python -m cardioseg.pipeline --target export --with-analysis --split synth_main
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import ClassVar

from cardioseg.evaluation.results import Results
from cardioseg.training.train import Train
from core.data.analysis.render import Render
from core.data.ingest.splits import Splits
from core.data.static import splits
from core.data.static.store.build import Build as store
from core.export_onnx import ExportOnnx
from core.hparams import TrainCfg
from core.obs import Obs

log = logging.getLogger("cardioseg.pipeline")


class Ctx:
    """Shared pipeline state (the one TrainCfg + resolved seeds/alias/flags) threaded through every stage.
    Owns the derived run dir + the val-sample resolver so the stages stay one-liners."""

    def __init__(self, cfg: TrainCfg, seeds: list[int], alias: str | None,
                 *, quick: bool, with_analysis: bool):
        self.cfg, self.seeds, self.alias = cfg, seeds, alias
        self.quick, self.with_analysis = quick, with_analysis

    @property
    def run(self) -> Path:
        """The primary run's artifact dir (single-seed base, or the seed-0 sibling) — where every stage
        reads/writes its artifact, and what `is_done` probes."""
        base = Path(self.cfg.out_dir or ".staging/run")
        return Train.seed_out_dir(base, self.seeds[0], single=len(self.seeds) == 1)

    def val_sample(self) -> str:  # pragma: no cover  (store.load + split resolution over the real data tree)
        """One held-out val npz path — the ONNX exporter's parity sample."""
        d = self.cfg.generator.data
        return splits.Splits.paths(splits.ModelSplit(d, store.load_cfg(d)).val)[0]


class Stage:
    """One pipeline node: an idempotent unit of work over the shared Ctx. `is_done` is the cheap artifact
    gate (skip if already materialized); `run` does the work by calling the node's domain method."""

    name: str = ""
    deps: tuple[str, ...] = ()

    def is_done(self, ctx: Ctx) -> bool:
        raise NotImplementedError

    def run(self, ctx: Ctx) -> None:
        raise NotImplementedError


class DataStage(Stage):
    """Materialize the consolidated store for the cfg. Never 'done' by design — Build is process-if-missing,
    so `run` is a no-op scan when built; this keeps the dependency explicit without a paramkey-path probe."""

    name = "data"

    def is_done(self, ctx: Ctx) -> bool:
        return False

    def run(self, ctx: Ctx) -> None:  # pragma: no cover  (store consolidation over the real data tree)
        store.load_cfg(ctx.cfg.generator.data, workers=ctx.cfg.workers)


class AnalysisStage(Stage):
    """Opt-in synth-vs-real visual diagnostic before training (the 'look before you train' check)."""

    name = "analysis"
    deps = ("data",)

    def is_done(self, ctx: Ctx) -> bool:
        return (ctx.run / "analysis.png").exists()

    def run(self, ctx: Ctx) -> None:  # pragma: no cover  (GPU render of real-vs-synth panels)
        Render.render_synth_vs_real(out_png=ctx.run / "analysis.png")


class TrainStage(Stage):
    """Train the model (this already emits card + attribution + ONNX + registry on a non-quick run)."""

    name = "train"
    deps = ("data",)

    def is_done(self, ctx: Ctx) -> bool:
        return (ctx.run / "model.pth").exists()

    def run(self, ctx: Ctx) -> None:  # pragma: no cover  (composition-root GPU training)
        Train.train_seg(ctx.cfg, alias=ctx.alias, quick=ctx.quick, seeds=ctx.seeds)


class EvaluateStage(Stage):
    """Emit the canonical per-run RESULTS.json (val ACDC + held-out vendors, EF, strata)."""

    name = "evaluate"
    deps = ("train",)

    def is_done(self, ctx: Ctx) -> bool:
        return (ctx.run / "RESULTS.json").exists()

    def run(self, ctx: Ctx) -> None:  # pragma: no cover  (GPU eval over the val/test frames)
        (ctx.run / "RESULTS.json").write_text(json.dumps(Results.build(ctx.run), indent=2))


class ExportStage(Stage):
    """Parity-gated ONNX export. Usually already 'done' by a non-quick train; a target here re-exports
    (e.g. after a --quick run that skipped it)."""

    name = "export"
    deps = ("train",)

    def is_done(self, ctx: Ctx) -> bool:
        return (ctx.run / "model.onnx").exists()

    def run(self, ctx: Ctx) -> None:  # pragma: no cover  (torch->ONNX + INT8 parity gate)
        ExportOnnx.export(ctx.run, ctx.val_sample())


class Pipeline:
    """The DAG runner: resolve a target to its transitive-dependency plan, then run each stage whose
    artifact is missing. Command-agnostic — no per-stage branching, the stages ARE the strategy."""

    _STAGES: ClassVar[dict[str, Stage]] = {s.name: s for s in
                                           (DataStage(), AnalysisStage(), TrainStage(), EvaluateStage(), ExportStage())}

    @staticmethod
    def plan(targets: list[str]) -> list[str]:
        """Topological order of the requested targets + all transitive deps (each stage before its
        dependents), de-duplicated. Pure — testable without any IO."""
        order: list[str] = []
        seen: set[str] = set()

        def visit(name: str) -> None:
            if name in seen:
                return
            seen.add(name)
            for dep in Pipeline._STAGES[name].deps:
                visit(dep)
            order.append(name)

        for t in targets:
            visit(t)
        return order

    @staticmethod
    def run(ctx: Ctx, targets: list[str]) -> list[str]:
        """Run the resolved plan, skipping already-done stages. Returns the stage names actually run."""
        did = []
        for name in Pipeline.plan(targets):
            stage = Pipeline._STAGES[name]
            if stage.is_done(ctx):
                log.info("skip %s (already done)", name)
                continue
            log.info("=> %s", name)
            stage.run(ctx)
            did.append(name)
        return did


def main():
    ap = argparse.ArgumentParser(prog="python -m cardioseg.pipeline",
                                 description="target-driven data->train->evaluate->export pipeline")
    ap.add_argument("--target", default="evaluate",
                    choices=["data", "analysis", "train", "evaluate", "export"],
                    help="materialize this target + its deps (default: evaluate)")
    ap.add_argument("--with-analysis", action="store_true", dest="with_analysis",
                    help="also run the opt-in synth-vs-real diagnostic before training")
    ap.add_argument("--split", default=None, help="coded-filter split family (e.g. static_main / synth_main)")
    ap.add_argument("--seeds", default=None, help="comma-sep seeds (multi-seed needs a coded --split)")
    ap.add_argument("--alias", default=None, help="registry alias to set (e.g. 'production')")
    ap.add_argument("--out", default=None, help="run artifact dir (default .staging/run)")
    ap.add_argument("--quick", action="store_true", help="skip the ONNX/attribution/registry tail in train")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides", help="deep cfg overrides (--set X=Y)")
    args = ap.parse_args()
    Obs.setup()

    cfg = TrainCfg()
    if args.split:
        name = args.split.split("@", 1)[0]
        if name not in Splits.list_splits():
            raise SystemExit(f"unknown split {name!r}; known: {Splits.list_splits()}")
        cfg.generator.data.split = args.split
    Train.apply_cli_args(cfg, {"out": args.out, "overrides": args.overrides})
    seeds = Train.resolve_seeds(cfg.seed, Train.parse_seeds(args.seeds))

    ctx = Ctx(cfg, seeds, args.alias, quick=args.quick, with_analysis=args.with_analysis)
    targets = [args.target] + (["analysis"] if args.with_analysis else [])
    did = Pipeline.run(ctx, targets)
    log.info("pipeline done — ran: %s", did or "(all up to date)")


if __name__ == "__main__":
    main()
