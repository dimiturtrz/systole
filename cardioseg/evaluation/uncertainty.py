"""Uncertainty + calibration for the shipped model (iq7), WITHOUT dropout — we use the variance the
4-flip TTA already produces. Per-voxel uncertainty = predictive entropy of the mean-over-flips softmax
(captures both flip-disagreement and low-confidence). Outputs, on the held-out set:
  - a per-voxel uncertainty map (saved as an overlay PNG for one patient),
  - a per-case uncertainty score (mean foreground entropy) -> flag the high-uncertainty cases,
  - ECE + a reliability diagram (is the softmax confidence calibrated?),
  - a sanity check that uncertainty concentrates on the boundary.

    python -m cardioseg.evaluation.uncertainty --run runs/gen --eval acdc

MC-dropout was tried (cardiac-seg-bp4) but dropout regressed EF ~2pp with no Dice gain, so the
flagship has no dropout; TTA-variance is the no-cost uncertainty signal instead.
"""
import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Any, Callable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Bool, Float, Integer
from scipy.ndimage import binary_erosion
from sklearn.metrics import average_precision_score, roc_auc_score

from core.config import FLAGSHIP_REF
from core.data.static import splits
from core.data.static.labels import Labels
from core.data.static.mri.base import Dataset, Phase
from core.data.static.store.build import Build as store
from core.inference import Inference
from core.preprocessing.preprocess import SIZE, Preprocess
from core.registry import Registry
from core.run import Run
from core.types import shapecheck

from ..tracking import Tracker

log = logging.getLogger("cardioseg.uncertainty")


class Uncertainty:
    """TTA-variance uncertainty + calibration folds (the free helpers folded in as staticmethods): the pure
    entropy/BALD decomposition, ECE binning, boundary-vs-interior split, per-case foreground folds, and the
    per-case collection orchestration + reliability/overlay plots for the eval CLI."""

    @staticmethod
    @shapecheck
    def tta_uncertainty(model: torch.nn.Module, vol_img: Float[np.ndarray, "d h w"], size: int, device: str):
        """Decompose predictive uncertainty over the 4 TTA flips (a cheap K-member ensemble).
        Returns (pred, total, conf, aleatoric, epistemic) — total/aleatoric/epistemic each [D,size,size]
        in [0,1] (normalized by log C):
            total      = H[mean]            (entropy of the mean softmax)
            aleatoric  = mean_k H[p_k]      (expected per-member entropy — irreducible ambiguity)
            epistemic  = total - aleatoric  (BALD / mutual information — reducible model uncertainty)
        NB the 4 flips are a *weak* ensemble (input-perturbation, not weight diversity), so epistemic
        here is a lower-bound proxy, not deep-ensemble gold."""
        # mean [D,C,H,W]; members [K,D,C,H,W]
        pred, mean, members = Inference(model, size, device).predict_volume_members(vol_img)
        logc = np.log(mean.shape[1])
        total = -(mean * (mean + 1e-12).log()).sum(1) / logc                        # H[mean]
        aleat = (-(members * (members + 1e-12).log()).sum(2)).mean(0) / logc        # mean_k H[p_k]
        epi = (total - aleat).clamp(min=0)                                          # BALD (>= 0 by Jensen)
        conf = mean.max(1).values
        return pred, total.cpu().numpy(), conf.cpu().numpy(), aleat.cpu().numpy(), epi.cpu().numpy()

    @staticmethod
    @shapecheck
    def _boundary(mask: Integer[np.ndarray, "h w"]) -> Bool[np.ndarray, "h w"]:
        """1-voxel boundary band of the foreground (per slice)."""
        fg = mask > 0
        return fg & ~binary_erosion(fg, iterations=1)

    @staticmethod
    @shapecheck
    def ece(conf: Float[np.ndarray, "*n"], correct: Float[np.ndarray, "*n"], n_bins: int = 15):
        """Expected Calibration Error + per-bin (conf, acc, weight) for a reliability diagram."""
        edges = np.linspace(0, 1, n_bins + 1)
        e: float = 0.0
        bins: list[tuple[float, float, float]] = []
        for lo, hi in itertools.pairwise(edges):
            m = (conf > lo) & (conf <= hi)
            if m.sum() == 0:
                continue
            c, a, w = conf[m].mean(), correct[m].mean(), m.mean()
            e += w * abs(a - c)
            bins.append((float(c), float(a), float(w)))
        return float(e), bins

    @staticmethod
    @shapecheck
    def boundary_interior_uncertainty(pred: Integer[np.ndarray, "d h w"], ent: Float[np.ndarray, "d h w"]):
        """Per-slice mean entropy on the foreground boundary band vs the interior. Returns two lists
        (bnd means, interior means) — a slice contributes to each only where that region is non-empty."""
        bnd_u: list[Any] = []
        int_u: list[Any] = []
        for z in range(pred.shape[0]):
            b = Uncertainty._boundary(pred[z]); inte = (pred[z] > 0) & ~b
            if b.any(): bnd_u.append(ent[z][b].mean())
            if inte.any(): int_u.append(ent[z][inte].mean())
        return bnd_u, int_u

    @staticmethod
    @shapecheck
    def foreground_uncertainty(pred: Integer[np.ndarray, "*grid"], gt: Integer[np.ndarray, "*grid"], maps: tuple[Float[np.ndarray, "*grid"], Float[np.ndarray, "*grid"], Float[np.ndarray, "*grid"], Float[np.ndarray, "*grid"]]):
        """Pure per-case fold: select foreground voxels (pred>0 OR gt>0) from the per-voxel maps
        `maps = (ent, conf, ale, epi)` and return the pooled arrays (conf, correct, ent, ale, epi) +
        the scalar per-case uncertainty (mean fg entropy, 0 if empty)."""
        ent, conf, ale, epi = maps
        fg = (pred > 0) | (gt > 0)
        score = float(ent[fg].mean()) if fg.any() else 0.0
        return conf[fg], (pred == gt)[fg], ent[fg], ale[fg], epi[fg], score

    @staticmethod
    def _collect_uncertainty(model: torch.nn.Module, df: Any, device: str, out: Path, eval_name: str):
        """Per-case foreground-voxel uncertainty over ED/ES frames: TTA entropy/confidence, aleatoric/
        epistemic (BALD), boundary-vs-interior, per-case scores — plus one overlay PNG (ACDC ED only).
        Returns the pooled sample lists (confs, corrects, ents, ales, epis, bnd_u, int_u, cases) for the
        caller to aggregate."""
        confs: list[Float[np.ndarray, "..."]] = []; corrects: list[Bool[np.ndarray, "..."]] = []; ents: list[Float[np.ndarray, "..."]] = []; ales: list[Float[np.ndarray, "..."]] = []; epis: list[Float[np.ndarray, "..."]] = []; bnd_u: list[float] = []; int_u: list[float] = []; cases: list[dict[str, float | str]] = []
        saved = False
        for r in df.iter_rows(named=True):
            case = store.load_arrays(r["path"])
            for tag in (p.lower() for p in Phase):
                if f"{tag}_img" not in case:
                    continue
                pred, ent, conf, ale, epi = Uncertainty.tta_uncertainty(model, case[f"{tag}_img"], SIZE, device)
                gt = Preprocess.stack_slices(case[f"{tag}_gt"], SIZE, dtype=np.uint8)
                cf, ok, en, al, ep, score = Uncertainty.foreground_uncertainty(pred, gt, (ent, conf, ale, epi))
                confs.append(cf); corrects.append(ok); ents.append(en); ales.append(al); epis.append(ep)
                cases.append({"case": f"{Path(r['path']).stem}_{tag.upper()}", "uncertainty": score})
                b, i = Uncertainty.boundary_interior_uncertainty(pred, ent)
                bnd_u.extend(b); int_u.extend(i)
                if not saved and eval_name == "acdc" and tag == Phase.ED.lower():  # one overlay PNG
                    z = int(np.argmax([(g > 0).sum() for g in gt]))
                    Uncertainty._save_overlay(
                        case[f"{tag}_img"], pred[z], ent[z], z, Path(r["path"]).stem,
                        out / "uncertainty_map.png", Preprocess.fit_square, SIZE, plt)
                    saved = True
        return confs, corrects, ents, ales, epis, bnd_u, int_u, cases

    @staticmethod
    def _reliability_plot(bins: list[tuple[float, float, float]], ece_val: float, out: Path) -> None:  # pragma: no cover  matplotlib rendering + PNG file write
        """Reliability diagram (foreground confidence vs accuracy per bin) -> out/reliability.png."""
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        if bins:
            cs, as_, _ = zip(*bins, strict=True)
            ax.plot([0, 1], [0, 1], "--", color="#999", lw=1)
            ax.plot(cs, as_, "o-", color="#5b8def")
        ax.set_xlabel("confidence (max softmax)"); ax.set_ylabel("accuracy")
        ax.set_title(f"Reliability (foreground) — ECE {ece_val:.3f}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        fig.tight_layout(); fig.savefig(out / "reliability.png", dpi=110); plt.close(fig)

    @staticmethod
    def _save_overlay(vol: Float[np.ndarray, "d h w"], pred: Integer[np.ndarray, "h w"], ent: Float[np.ndarray, "h w"], z: int, name: str, path: Path, fit_square: Callable[..., Any], size: int, plt: Any) -> None:  # noqa: PLR0913
        img = fit_square(vol[z].astype(np.float32), size, 0.0)
        cmap = Labels.overlay_cmap()
        fig, ax = plt.subplots(1, 3, figsize=(9, 3.2))
        ax[0].imshow(img, cmap="gray"); ax[0].set_title(f"{name} ED  z={z}")
        ax[1].imshow(img, cmap="gray")
        ax[1].imshow(pred, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        ax[1].set_title("prediction")
        ax[2].imshow(img, cmap="gray")
        im = ax[2].imshow(ent, cmap="inferno", alpha=.7, vmin=0, vmax=ent.max() or 1)
        ax[2].set_title("uncertainty (entropy)")
        for a in ax: a.axis("off")
        fig.colorbar(im, ax=ax[2], fraction=.046)
        fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)

    @staticmethod
    def add_args(ap: argparse.ArgumentParser) -> None:
        ap.add_argument("--run", default=FLAGSHIP_REF)
        ap.add_argument("--eval", default=Dataset.ACDC, choices=[Dataset.ACDC, "canon"])

    @staticmethod
    def run(args: argparse.Namespace) -> None:  # pragma: no cover  CLI entrypoint: mlflow model loading (network) + GPU + tracking + file writes
        run = Registry.resolve(args.run)
        model, _, device = Run.load_run(run)

        df = splits.Splits.eval_set(args.eval)   # 'canon' -> unseen-vendor slice; else whole dataset

        out = run / "plots"
        out.mkdir(parents=True, exist_ok=True)
        confs, corrects, ents, ales, epis, bnd_u, int_u, cases = Uncertainty._collect_uncertainty(
            model, df, device, out, args.eval)

        conf = np.concatenate(confs); correct = np.concatenate(corrects).astype(float)
        e, bins = Uncertainty.ece(conf, correct)
        cases.sort(key=lambda x: -x["uncertainty"])
        bratio = float(np.mean(bnd_u) / max(np.mean(int_u), 1e-6))

        # does uncertainty predict error? wrong (pred!=gt) is the rare positive; entropy is the detector.
        # AUPRC is the honest read under imbalance (compare to base rate); ROC-AUC for comparability.
        ent_all = np.concatenate(ents); wrong = 1.0 - correct
        base = float(wrong.mean())
        rocauc = float(roc_auc_score(wrong, ent_all))
        auprc = float(average_precision_score(wrong, ent_all))

        # aleatoric/epistemic decomposition (BALD) over foreground — how much uncertainty is reducible.
        # Members are TTA flips (weak proxy for a posterior), so epistemic is a LOWER BOUND: a deep
        # ensemble surfaces more model disagreement, so the true reducible fraction is >= this (bd zxcm).
        ale_m = float(np.concatenate(ales).mean()); epi_m = float(np.concatenate(epis).mean())
        epi_frac = epi_m / max(ale_m + epi_m, 1e-9)

        (out / "uncertainty.json").write_text(json.dumps(
            {"ece": round(e, 4), "boundary_vs_interior_ratio": round(bratio, 2),
             "error_detection": {"auprc": round(auprc, 3), "base_rate": round(base, 3),
                                 "lift_over_base": round(auprc / max(base, 1e-6), 1), "rocauc": round(rocauc, 3)},
             "decomposition": {"aleatoric": round(ale_m, 4), "epistemic": round(epi_m, 4),
                               "epistemic_fraction_tta_lb": round(epi_frac, 3)},
             "n_cases": len(cases), "most_uncertain": cases[:8]}, indent=2))

        Uncertainty._reliability_plot(bins, e, out)

        log.info(f"ECE {e:.3f} | boundary/interior {bratio:.2f}x | error-detect AUPRC {auprc:.3f} "
                 f"(base {base:.3f}, {auprc/max(base,1e-6):.1f}x) ROC-AUC {rocauc:.3f} | "
                 f"aleatoric {ale_m:.3f} / epistemic {epi_m:.3f} ({epi_frac:.0%} reducible, TTA lower bound) | "
                 f"most-uncertain: {cases[0]['case']} ({cases[0]['uncertainty']:.3f})")
        log.info(f"-> {out}/uncertainty_map.png, reliability.png, uncertainty.json")

        tracker = Tracker("cardioseg", run.name)
        trk = tracker.track_run(run_dir=run)      # resume the train run
        ev = args.eval
        trk.metric(f"{ev}_ece", e); trk.metric(f"{ev}_epistemic_frac_tta_lb", epi_frac)
        trk.metric(f"{ev}_err_auprc", auprc); trk.metric(f"{ev}_boundary_ratio", bratio)
        trk.artifact(out / "uncertainty.json"); trk.artifact(out / "reliability.png")
        trk.end()
