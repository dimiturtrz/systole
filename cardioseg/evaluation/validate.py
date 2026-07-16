"""Validation for ACDC segmentation: predict full short-axis volumes, score
per-class Dice (pooled over slices) and per-patient ejection fraction vs ground
truth. Lives in evaluation/ (not training/) — measuring a model is evaluation.

EF uses each patient's own spacing. Note EF is a volume *ratio*, so a constant
spacing cancels; the per-patient spacing matters once absolute volumes (mL) are
reported, and is the honest thing to carry through regardless.
"""
import logging
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Integer
from pydantic import BaseModel

from core.config import _VALIDATE
from core.data.static.mri.base import Phase
from core.data.static.store import Store
from core.evaluate import CLASSES, Evaluate

# The predict_volume kernel moved to core.inference (shared by the viewer + uncertainty decomposition);
# this module keeps the validation ORCHESTRATION (score a set of npz cases -> Dice/EF/boundary tables).
from core.inference import Inference
from core.measure import Measure
from core.postprocess import Postprocess
from core.preprocessing.preprocess import SIZE, Preprocess
from core.types import Spacing, shapecheck

load_arrays = Store.load_arrays

log = logging.getLogger("cardioseg.validate")

CLASS_NAMES = {k: name for k, (name, _) in CLASSES.items()}   # single source: evaluate.CLASSES


class EvalCfg(BaseModel):
    """Evaluation knobs, set ONCE per Evaluator (init-once, call-many). `size` = the model's trained
    input resolution (== DataCfg.size / the run's config.json) — a MODEL property, not a per-call arg;
    postproc (largest-CC) + tta (test-time flips) are inference options."""
    model_config = _VALIDATE
    size: int = SIZE
    postproc: bool = True
    tta: bool = True
    boundary: bool = True          # compute HD95/ASSD (per-class 3D distance transforms — the eval's heaviest
    #                                CPU step); turn OFF for Dice/EF-only sweeps (result-identical Dice/EF)


class _ClassScores:
    """Running per-class Dice numerator/denominator + per-volume boundary distances. Holds the three
    accumulators as its OWN state, so scoring a volume is `.add(pred, gt, spacing)` — no dict-threading
    through a 6-arg helper. `.dice()` / `.surface()` finalize. Dice is computed one-hot over all classes
    at once (bd cardiac-seg-8fux); only HD95 stays per-label (a distance transform per class)."""

    _CLS = np.fromiter(CLASS_NAMES, dtype=int)                           # [C] foreground label ids (1,2,3)

    def __init__(self, *, boundary: bool = True):
        self.boundary = boundary
        self.inter = dict.fromkeys(CLASS_NAMES, 0.0)
        self.denom = dict.fromkeys(CLASS_NAMES, 0.0)
        self.surf = {c: {"hd95": [], "assd": []} for c in CLASS_NAMES}   # per-volume boundary distances (mm)

    @shapecheck
    def add(self, pred: Integer[np.ndarray, "*grid"], gt: Integer[np.ndarray, "*grid"], spacing: Spacing):
        """Fold one predicted volume into the accumulators."""
        # Dice numerator/denominator for ALL classes in one shot: one-hot pred/gt to [...,C] and reduce
        # over the spatial axes (no python class loop). 2*|P∩G| and |P|+|G| per class.
        axes = tuple(range(pred.ndim))
        pred_onehot = pred[..., None] == self._CLS                       # [...,C] bool one-hot
        gt_onehot = gt[..., None] == self._CLS
        inter_c = 2.0 * np.logical_and(pred_onehot, gt_onehot).sum(axis=axes)   # [C]
        denom_c = pred_onehot.sum(axis=axes) + gt_onehot.sum(axis=axes)         # [C]
        for i, cl in enumerate(CLASS_NAMES):
            self.inter[cl] += float(inter_c[i])
            self.denom[cl] += float(denom_c[i])
        if not self.boundary:                                           # Dice/EF-only sweep — skip the EDT cost
            return
        for cl in CLASS_NAMES:
            surface_dist = Evaluate.surface_distances(pred, gt, cl, spacing)   # 3D boundary distances (mm), per-label
            if surface_dist.size:
                metrics = Evaluate.surface_metrics(surface_dist)
                self.surf[cl]["hd95"].append(metrics.hd95); self.surf[cl]["assd"].append(metrics.assd)

    def dice(self) -> dict[int, float]:
        return {cl: (self.inter[cl] / self.denom[cl] if self.denom[cl] else float("nan"))
                for cl in CLASS_NAMES}

    def surface(self) -> dict:
        # median over volumes — robust report (HD95 already drops per-volume outliers; median across
        # cases drops the odd failed volume too — "worst case decides, but report robust").
        return {cl: {"hd95": float(np.median(self.surf[cl]["hd95"])) if self.surf[cl]["hd95"] else float("nan"),
                     "assd": float(np.median(self.surf[cl]["assd"])) if self.surf[cl]["assd"] else float("nan")}
                for cl in CLASS_NAMES}


class Evaluator:
    """Scores a trained model over subject npz files. Holds the model + device + EvalCfg as STATE —
    construct once, call `.validate(paths)` on any subject set. (Replaces the old
    `validate(model, paths, size, device, postproc, tta)` where size/device/model were threaded as
    args; they're state, not inputs — bd cardiac-seg-01fh.)"""

    def __init__(self, model, device: str, cfg: EvalCfg | None = None):
        self.model, self.device, self.cfg = model, device, cfg or EvalCfg()

    @staticmethod
    @shapecheck
    def _foreground_samples(
        logits: Float[np.ndarray, "*n c"], y: Integer[np.ndarray, "*n"], per_vol, rng
    ) -> tuple[Float[np.ndarray, "*m c"], Integer[np.ndarray, "*m"]]:
        """Pure core of `gather`: from flattened logits [N,C] + labels [N], keep foreground voxels
        (GT>0 OR argmax>0) and subsample to <= per_vol. Returns (logits_kept [M,C], labels_kept [M])."""
        pred = logits.argmax(1)
        foreground_indices = np.where((y > 0) | (pred > 0))[0]                # foreground voxels
        if foreground_indices.size > per_vol:
            foreground_indices = rng.choice(foreground_indices, per_vol, replace=False)
        return logits[foreground_indices], y[foreground_indices]

    def validate(self, npz_paths) -> tuple[dict[int, float], list[dict], dict]:
        """Return (dice_per_class, ef_rows, surf_per_class). dice_per_class: {1,2,3 -> Dice pooled over
        all val slices}. ef_rows: {patient, group, ef_gt, ef_pred, edv_gt, edv_pred}. `npz_paths` are
        consolidated-subject npz files (data/store.py) — resampled+z-scored ed/es img+gt, spacing,
        group; dataset-agnostic (canonical labels)."""
        model, device, size = self.model, self.device, self.cfg.size
        postproc, tta = self.cfg.postproc, self.cfg.tta
        scores = _ClassScores(boundary=self.cfg.boundary)        # holds the Dice/boundary accumulators
        ef_rows = []
        for npz_path in npz_paths:
            case = load_arrays(npz_path)
            spacing = tuple(float(s) for s in case["spacing"])   # per-patient (z,y,x)
            vols = {}
            for tag in ("ED", "ES"):
                if f"{tag.lower()}_img" not in case:
                    continue
                pred = Inference(model, size, device).predict_volume(case[f"{tag.lower()}_img"], tta=tta)
                if postproc:
                    pred = Postprocess.largest_cc_per_class(pred)
                gt = Preprocess.stack_slices(case[f"{tag.lower()}_gt"], size)
                vols[tag] = (pred, gt)
                scores.add(pred, gt, spacing)
            if "ED" in vols and "ES" in vols:
                ef_pred, edv_pred, _ = Measure.ejection_fraction(vols["ED"][0], vols["ES"][0], spacing)
                ef_gt, edv_gt, _ = Measure.ejection_fraction(vols["ED"][1], vols["ES"][1], spacing)
                ef_rows.append({"patient": Path(npz_path).stem, "group": case.get("group"),
                                "ef_gt": ef_gt, "ef_pred": ef_pred, "edv_gt": edv_gt, "edv_pred": edv_pred})
        return scores.dice(), ef_rows, (scores.surface() if self.cfg.boundary else None)

    def gather(self, npz_paths, per_vol: int = 4000, seed: int = 0):
        """Foreground (logits[N,C], labels[N]) over subjects — single forward, NO TTA/postproc —
        subsampled to ~per_vol voxels/volume (bounded memory, plenty for a 1-param temperature fit +
        ECE). For calibration. (Was calibrate._gather(model, paths, size, device, …); model/size/device
        are now state.)"""
        size = self.cfg.size
        rng = np.random.RandomState(seed)
        logit_rows, label_rows = [], []
        self.model.eval()
        for p in npz_paths:
            case = load_arrays(p)
            for tag in (phase.lower() for phase in Phase):
                if f"{tag}_img" not in case:
                    continue
                squared_slices = np.stack([
                    Preprocess.fit_square(slice_img.astype(np.float32), size, 0.0)
                    for slice_img in case[f"{tag}_img"]
                ])
                gt = Preprocess.stack_slices(case[f"{tag}_gt"], size, dtype=np.int64)
                with torch.no_grad():
                    logits = self.model(torch.from_numpy(squared_slices)[:, None].to(self.device))   # [D,C,H,W]
                logits = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1]).cpu().numpy()  # [Npix,C]
                logits_kept, labels_kept = self._foreground_samples(logits, gt.reshape(-1), per_vol, rng)
                logit_rows.append(logits_kept); label_rows.append(labels_kept)
        return np.concatenate(logit_rows), np.concatenate(label_rows)


    @staticmethod
    def summarize(dice_per_class, ef_rows, surf_per_class=None):
        """Print the Dice table + (boundary table) + EF table, return a JSON-able metrics dict."""
        log.info("\n=== VAL Dice (per class, pooled over slices) ===")
        for cl, name in CLASS_NAMES.items():
            log.info(f"  {name:7} (label {cl}): {dice_per_class[cl]:.3f}")
        mean_dice = float(np.nanmean([dice_per_class[c] for c in CLASS_NAMES]))
        log.info(f"  mean: {mean_dice:.3f}")

        if surf_per_class:
            log.info("\n=== VAL boundary (median over volumes, mm) ===")
            for cl, name in CLASS_NAMES.items():
                log.info(f"  {name:7} HD95 {surf_per_class[cl]['hd95']:5.2f}  ASSD {surf_per_class[cl]['assd']:5.2f}")

        log.info("\n=== VAL EF: GT vs predicted ===")
        errors = []
        for row in ef_rows:
            abs_error = abs(row["ef_gt"] - row["ef_pred"])
            errors.append(abs_error)
            log.info(f"  {row['patient']:11} {row['group']!s:5}  GT {row['ef_gt']:5.1f}%  "
                     f"pred {row['ef_pred']:5.1f}%  |d| {abs_error:4.1f}")
        ef_mae = float(np.mean(errors)) if errors else float("nan")
        if errors:
            log.info(f"  EF MAE = {ef_mae:.1f}%  (n={len(errors)})")

        return {
            "dice": {CLASS_NAMES[c]: dice_per_class[c] for c in CLASS_NAMES},
            "dice_mean": mean_dice,
            "ef_mae": ef_mae,
            "ef_rows": ef_rows,
            "boundary": ({CLASS_NAMES[c]: surf_per_class[c] for c in CLASS_NAMES}
                         if surf_per_class else None),
        }
