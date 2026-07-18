"""RV-omission diagnostics for a zero-real model — the reproducible core of the nttu epic (bd nttu.6).

Four modes via `python -m cardioseg.evaluation rv_omission --run <zero-real> --mode M`, M in
`{probe, bias, gated, deficit}`:

- **probe** (nttu.5 root-cause): on every slice where GT-RV is present but argmax omits it (<OMIT_PX RV
  pixels, PRE-largest-CC), is there raw RV softmax to recover (recall/threshold-fixable) or is it truly
  zero (coverage/detection-absent)? Splits the tail and names the winning class in the GT-RV region.
- **bias** (nttu.7/we55 lever): fit a constant added to the RV logit pre-softmax on VAL (leak-free dose-
  response → `select_bias`, best mean Dice with a cav-regression guard), then apply that one b* to the
  model's own cross-vendor TEST, reported pooled + PER-VENDOR. The verdict (49b7) predicts the biggest
  lift on the unseen vendors (GE/Canon), where RV collapses under OOD color. `core.inference.Inference`
  carries the fitted b* as an opt-in `logit_bias` for the real pipeline.
- **gated** (ru27): the conditional fix for `bias`'s vendor-heterogeneity — apply the RV bias ONLY on
  under-fired slices (per-slice max RV softmax < `GATE_TAU`, the nttu.5 omission signature). Fits b* on
  val under the gate, then reports base vs global vs GATED per vendor: gating should recover the
  collapsed vendors (Canon) without the global bias's GE over-segmentation (we55).
- **deficit** (egeh): is the RV gap a hard-omission cliff or a broad partial-quality continuum? Reports
  the per-slice RV-Dice histogram over the model's own cross-vendor test, the true 0-px omission count,
  and whether those omissions have a confident-RV neighbour (2.5D signal). Verdict: omission is a
  negligible tail; the deficit is broad partial under-segmentation — not a coverage/omission lever.

Zero-real models live in staging (unregistered), so `--run` is explicit (a run dir / alias / id resolvable
by the registry), e.g. `--run .staging/refac_proc`.
"""
import argparse
import logging
from itertools import pairwise
from typing import Any, NamedTuple, Sequence

import numpy as np
import torch
from jaxtyping import Float, Integer, UInt8

from core.data.static import splits
from core.data.static.splits import ModelSplit
from core.data.static.store import Store
from core.data.static.store.build import Build as store
from core.evaluate import Evaluate
from core.hparams import Hparams, TrainCfg
from core.model import Model
from core.postprocess import Postprocess
from core.preprocessing.preprocess import Preprocess
from core.registry import Registry

log = logging.getLogger("cardioseg.rv_omission")


class Session(NamedTuple):
    """The loaded inference session threaded through the diagnostics: model + its cfg + device."""
    model: torch.nn.Module
    cfg: TrainCfg
    device: str

RV = 1
OMIT_PX = 20
FLIPS: tuple[list[int], ...] = ([], [2], [3], [2, 3])
CLASS_NAMES = ("bg", "RV", "myo", "cav")
FOREGROUND = {"RV": 1, "myo": 2, "cav": 3}
ACTIVATION_FLOOR = 0.05  # maxP below this = true-zero-activation (coverage), above = recall-recoverable
BIASES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
GATE_TAU = 0.6  # nttu.5: under-fired RV peaks 0.21-0.57, healthy RV wins >0.6 -> gate the bias below this
DICE_EDGES = (0.0, 0.05, 0.3, 0.6, 0.8, 1.01)  # egeh: RV per-slice Dice buckets (deficit-shape histogram)
CONFIDENT_PX = 20  # egeh: a slice "has confident RV" if argmax fires >= this many RV pixels


class RvOmission:
    """RV-omission diagnostics: the pure classification/scoring cores (testable) + a GPU/IO shell
    (`run`, pragma-excluded). `probe` splits the omission tail recall-vs-coverage; `bias` sweeps the
    inference-time RV logit-bias recall lever."""

    @staticmethod
    def omission_row(
        rv_prob: Float[np.ndarray, "h w"],
        argmax: Integer[np.ndarray, "h w"],
        gt: Integer[np.ndarray, "h w"],
        pred: Integer[np.ndarray, "h w"],
    ) -> dict[str, float | int | str] | None:
        """One omitted-slice record, or None if the slice isn't an omission. Omission = GT-RV present
        (>OMIT_PX) yet argmax fired <OMIT_PX RV pixels. Reports the max/mean RV softmax INSIDE the GT-RV
        region and which class won there. Pure numpy — the recall-vs-coverage evidence, testable off-GPU."""
        gz = gt == RV
        if gz.sum() <= OMIT_PX or (argmax == RV).sum() >= OMIT_PX:
            return None
        pin = rv_prob[gz]
        winners = np.bincount(pred[gz].astype(int), minlength=len(CLASS_NAMES))
        win_cls = int(winners.argmax())
        return {"gtpx": int(gz.sum()), "maxp_in_gt": float(pin.max()), "meanp_in_gt": float(pin.mean()),
                "win": CLASS_NAMES[win_cls], "win_frac": float(winners[win_cls] / gz.sum())}

    @staticmethod
    def split_verdict(maxps: list[float]) -> dict[str, float | int]:
        """Recall-vs-coverage split over the omitted slices' in-GT max RV softmax: how many carry
        recoverable activation (>= floor) vs are true-zero (coverage). Pure — the nttu.5 conclusion."""
        arr = np.array(maxps) if maxps else np.zeros(0)
        recoverable = int((arr >= ACTIVATION_FLOOR).sum())
        return {"n": len(arr), "recoverable": recoverable, "zero_activation": len(arr) - recoverable,
                "min": float(arr.min()) if arr.size else float("nan"),
                "med": float(np.median(arr)) if arr.size else float("nan"),
                "max": float(arr.max()) if arr.size else float("nan")}

    @staticmethod
    def _tta_mean(logits: Float[torch.Tensor, "kd c h w"], d: int, bias: float = 0.0) -> Float[torch.Tensor, "d c h w"]:
        """TTA-mean softmax after adding `bias` to the RV logit pre-softmax. `logits` is the stacked
        [K*D,C,H,W] forward; `d` the per-flip depth. Un-flips each block back before averaging."""
        biased = logits.clone()
        biased[:, RV] += bias
        probs = torch.softmax(biased, dim=1)
        flips = [torch.flip(probs[i * d:(i + 1) * d], dm) if dm else probs[i * d:(i + 1) * d]
                 for i, dm in enumerate(FLIPS)]
        return torch.stack(flips).mean(0)

    @staticmethod
    def dice_buckets(dices: list[float], edges: Sequence[float]) -> list[tuple[float, float, int]]:
        """Histogram per-slice RV Dice into [edges[i], edges[i+1]) bins — the deficit SHAPE (egeh: is the
        RV gap a hard-omission cliff or a broad partial-quality continuum?). Pure, testable off-GPU."""
        arr = np.array(dices) if dices else np.zeros(0)
        return [(lo, hi, int(((arr >= lo) & (arr < hi)).sum())) for lo, hi in pairwise(edges)]

    @staticmethod
    def biased_pred(logits: Float[torch.Tensor, "kd c h w"], d: int, bias: float = 0.0) -> UInt8[np.ndarray, "d h w"]:
        """argmax label map after a GLOBAL RV logit bias (every slice). The we55 lever — over-segments
        vendors whose RV is already healthy (bd ru27)."""
        return RvOmission._tta_mean(logits, d, bias).argmax(1).to(torch.uint8).cpu().numpy()

    @staticmethod
    def gated_biased_pred(logits: Float[torch.Tensor, "kd c h w"], d: int, bias: float = 0.0,
                          tau: float = 0.6) -> UInt8[np.ndarray, "d h w"]:
        """CONDITIONAL RV bias (ru27): apply `bias` only on slices whose unbiased per-slice max RV
        softmax is below `tau` — the nttu.5 omission signature (RV present but out-competed). Slices
        where RV already wins strongly (>tau) keep the unbiased argmax, so the collapse is recovered
        WITHOUT over-segmenting healthy-RV vendors (the we55 global-bias failure mode)."""
        mean0 = RvOmission._tta_mean(logits, d, 0.0)
        meanb = RvOmission._tta_mean(logits, d, bias)
        gated = mean0[:, RV].amax(dim=(1, 2)) < tau                   # [d] per-slice under-fired mask
        out = torch.where(gated[:, None, None], meanb.argmax(1), mean0.argmax(1))
        return out.to(torch.uint8).cpu().numpy()

    @staticmethod
    def select_bias(sweep: dict[float, dict[str, float]], *, cav_guard: float = 0.01) -> float:  # pragma: no cover
        """Val-best RV bias by mean foreground Dice, guarding cav against regressing more than `cav_guard`
        below the unbiased (b=0) cav — nttu.7: over-biasing RV steals softmax mass from the cavity (the
        other blood pool). b=0 is always eligible, so a lever that only hurts falls back to no bias. Pure —
        the leak-free selection (fit on VAL only), testable off-GPU."""
        base_cav = sweep[0.0]["cav"]
        best_b, best_mean = 0.0, -1.0
        for b, d in sorted(sweep.items()):
            if d["cav"] < base_cav - cav_guard:
                continue
            m = float(np.mean(list(d.values())))
            if m > best_mean:
                best_mean, best_b = m, b
        return best_b

    @staticmethod
    def add_args(ap: argparse.ArgumentParser) -> None:
        ap.add_argument("--run", required=True,
                        help="zero-real model (run dir / alias / id), e.g. .staging/refac_proc")
        ap.add_argument("--mode", choices=("probe", "bias", "gated", "deficit"), default="probe")
        ap.add_argument("--testset", default="cmrxmotion",
                        help="probe eval set (bias uses the model's own val/test split)")

    @staticmethod
    def _load(run_ref: str) -> Session:  # pragma: no cover
        run = Registry.resolve(run_ref)
        cfg = Hparams.from_json(run / "config.json")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Model.build_unet(cfg.model).to(device)
        model.load_state_dict(torch.load(run / "model.pth", map_location=device))
        model.eval()
        return Session(model, cfg, device)

    @staticmethod
    def _logits(
        s: Session, vol_img: Float[np.ndarray, "d h w"]
    ) -> tuple[Float[torch.Tensor, "kd c h w"], int]:  # pragma: no cover
        size = s.cfg.generator.data.size
        xs = np.stack([Preprocess.fit_square(vol_img[z].astype(np.float32), size, 0.0)
                       for z in range(vol_img.shape[0])])
        with torch.no_grad():
            x = torch.from_numpy(xs)[:, None].to(s.device)
            batched = torch.cat([torch.flip(x, dm) if dm else x for dm in FLIPS], dim=0)
            return s.model(batched), x.shape[0]

    @staticmethod
    def _probe(s: Session, frame: Any) -> dict[str, float | int]:  # pragma: no cover
        size = s.cfg.generator.data.size
        rows = []
        for r in frame.iter_rows(named=True):
            case = Store.load_arrays(r["path"])
            gt = Preprocess.stack_slices(case["ed_gt"], size)
            lg, d = RvOmission._logits(s, case["ed_img"])
            probs = torch.softmax(lg[:d], dim=1)                 # identity-flip block = un-augmented
            rv_prob = probs[:, RV].cpu().numpy()
            pred = probs.argmax(1).to(torch.uint8).cpu().numpy()
            for z in range(gt.shape[0]):
                row = RvOmission.omission_row(rv_prob[z], pred[z], gt[z], pred[z])
                if row:
                    rows.append(row)
        verdict = RvOmission.split_verdict([float(x["maxp_in_gt"]) for x in rows])
        log.info("omitted slices n=%d | recoverable(>=%.2f) %d | zero-activation %d | maxP min/med/max %.3f/%.3f/%.3f",
                 verdict["n"], ACTIVATION_FLOOR, verdict["recoverable"], verdict["zero_activation"],
                 verdict["min"], verdict["med"], verdict["max"])
        for x in sorted(rows, key=lambda r: r["maxp_in_gt"]):
            log.info("  gtpx %4d  maxP_gt %.3f  meanP_gt %.3f  win %-3s %.2f",
                     x["gtpx"], x["maxp_in_gt"], x["meanp_in_gt"], x["win"], x["win_frac"])
        return verdict

    @staticmethod
    def _deficit(s: Session, frame: Any) -> None:  # pragma: no cover
        """egeh: is the RV gap a hard-omission cliff or a broad partial-quality continuum? Reports the
        per-slice RV-Dice histogram over GT-RV-present slices, the true 0-px omission count, and — for
        those omissions — whether the missing apical RV is available in an adjacent z-slice (2.5D signal)."""
        size = s.cfg.generator.data.size
        dices: list[float] = []
        n_omit, neigh_hit = 0, 0
        apical: list[float] = []
        for r in frame.iter_rows(named=True):
            case = Store.load_arrays(r["path"])
            gt = Preprocess.stack_slices(case["ed_gt"], size)
            lg, d = RvOmission._logits(s, case["ed_img"])
            pred = torch.softmax(lg[:d], dim=1).argmax(1).to(torch.uint8).cpu().numpy()
            rvpx = np.array([int((pred[z] == RV).sum()) for z in range(pred.shape[0])])
            for z in range(pred.shape[0]):
                gz, pz = gt[z] == RV, pred[z] == RV
                if int(gz.sum()) <= OMIT_PX:
                    continue
                inter = int((gz & pz).sum())
                dices.append(2 * inter / int(gz.sum() + pz.sum()) if int(gz.sum() + pz.sum()) else 1.0)
                if rvpx[z] < OMIT_PX:                          # true 0-px omission
                    n_omit += 1
                    apical.append(z / max(pred.shape[0] - 1, 1))
                    nb = [zz for zz in (z - 1, z + 1) if 0 <= zz < pred.shape[0]]
                    if any(rvpx[zz] >= CONFIDENT_PX for zz in nb):
                        neigh_hit += 1
        log.info("=== RV per-slice Dice on GT-RV-present slices (n=%d) — deficit shape ===", len(dices))
        for lo, hi, m in RvOmission.dice_buckets(dices, DICE_EDGES):
            log.info("  Dice [%.2f,%.2f): %5d (%4.1f%%)", lo, hi, m, 100 * m / max(len(dices), 1))
        arr = np.array(dices) if dices else np.zeros(1)
        log.info("  mean per-slice RV Dice %.3f  median %.3f", float(arr.mean()), float(np.median(arr)))
        log.info("true 0-px omissions: %d (%.2f%% of slices) | apical z-pos med %.2f | with confident-RV "
                 "neighbor +-1: %d/%d", n_omit, 100 * n_omit / max(len(dices), 1),
                 float(np.median(apical)) if apical else float("nan"), neigh_hit, n_omit)

    @staticmethod
    def _mean(d: dict[str, float]) -> float:  # pragma: no cover
        return float(np.mean([d[c] for c in FOREGROUND]))

    @staticmethod
    def _score_cases(
        s: Session,
        frame: Any,
        biases: Sequence[float],
        tau: float | None = None,
    ) -> list[tuple[str | None, dict[float, dict[str, float]]]]:  # pragma: no cover
        """Per-case foreground Dice at each bias, keeping the case vendor. Forward runs ONCE per case;
        the bias is a cheap logit add on the cached logits. `tau=None` = global bias; a float gates the
        bias to under-fired slices (ru27). Returns [(vendor, {b: {class: dice}})]."""
        size = s.cfg.generator.data.size
        cases = []
        for r in frame.iter_rows(named=True):
            case = Store.load_arrays(r["path"])
            gt = Preprocess.stack_slices(case["ed_gt"], size)
            lg, d = RvOmission._logits(s, case["ed_img"])
            per = {}
            for b in biases:
                raw = RvOmission.biased_pred(lg, d, b) if tau is None else RvOmission.gated_biased_pred(lg, d, b, tau)
                pred = Postprocess.largest_cc_per_class(raw)
                per[b] = {c: float(Evaluate.dice(pred, gt, lab)) for c, lab in FOREGROUND.items()}
            cases.append((r.get("vendor"), per))
        return cases

    @staticmethod
    def _agg(
        cases: list[tuple[str | None, dict[float, dict[str, float]]]],
        biases: Sequence[float],
    ) -> dict[float, dict[str, float]]:  # pragma: no cover
        """Pooled mean per-bias per-class over [(vendor, {b:{class:dice}})]."""
        return {b: {c: float(np.mean([pc[b][c] for _, pc in cases])) for c in FOREGROUND} for b in biases}

    @staticmethod
    def _score_arms(
        s: Session, frame: Any, b: float, tau: float
    ) -> list[tuple[str | None, dict[str, dict[str, float]]]]:  # pragma: no cover
        """Per case, from ONE forward: base (b=0), global (b every slice), gated (b on under-fired
        slices only). The ru27 head-to-head. Returns [(vendor, {arm: {class: dice}})]."""
        size = s.cfg.generator.data.size
        cases = []
        for r in frame.iter_rows(named=True):
            case = Store.load_arrays(r["path"])
            gt = Preprocess.stack_slices(case["ed_gt"], size)
            lg, d = RvOmission._logits(s, case["ed_img"])
            preds = {"base": RvOmission.biased_pred(lg, d, 0.0), "global": RvOmission.biased_pred(lg, d, b),
                     "gated": RvOmission.gated_biased_pred(lg, d, b, tau)}
            row = {a: {c: float(Evaluate.dice(Postprocess.largest_cc_per_class(p), gt, lab))
                       for c, lab in FOREGROUND.items()} for a, p in preds.items()}
            cases.append((r.get("vendor"), row))
        return cases

    @staticmethod
    def _agg_arm(
        cases: list[tuple[str | None, dict[str, dict[str, float]]]],
        arm: str,
        b: float,
    ) -> dict[float, dict[str, float]]:  # pragma: no cover
        """A base-vs-`arm` agg keyed {0.0: base, b: arm} so `_report` renders the arm's Δ over base."""
        return {0.0: {c: float(np.mean([row["base"][c] for _, row in cases])) for c in FOREGROUND},
                b: {c: float(np.mean([row[arm][c] for _, row in cases])) for c in FOREGROUND}}

    @staticmethod
    def _report(agg: dict[float, dict[str, float]], label: str, b_star: float) -> None:  # pragma: no cover
        base, biased = agg[0.0], agg[b_star]
        log.info("=== %s ===  RV     myo    cav    mean", label)
        for tag, dd in ((f"b=0.0 {'':4}", base), (f"b={b_star:<4.1f}", biased)):
            log.info("  %s %.3f  %.3f  %.3f  %.3f", tag, dd["RV"], dd["myo"], dd["cav"], RvOmission._mean(dd))
        log.info("  Δ       %+.3f %+.3f %+.3f %+.3f", biased["RV"] - base["RV"], biased["myo"] - base["myo"],
                 biased["cav"] - base["cav"], RvOmission._mean(biased) - RvOmission._mean(base))

    @staticmethod
    def _bias(s: Session, val_frame: Any, test_frame: Any) -> None:  # pragma: no cover
        # fit on VAL (leak-free): full dose-response, then select b* (best mean, cav-guarded)
        val_sweep = RvOmission._agg(RvOmission._score_cases(s, val_frame, BIASES), BIASES)
        log.info("=== VAL dose-response (selection) n=%d ===  bias  RV     myo    cav    mean", len(val_frame))
        for b in BIASES:
            dd = val_sweep[b]
            log.info("  b=%-4.1f %.3f  %.3f  %.3f  %.3f", b, dd["RV"], dd["myo"], dd["cav"], RvOmission._mean(dd))
        b_star = RvOmission.select_bias(val_sweep)
        log.info(">>> selected RV logit-bias b*=%.1f (val mean %.3f vs b=0 %.3f)", b_star,
                 RvOmission._mean(val_sweep[b_star]), RvOmission._mean(val_sweep[0.0]))
        if b_star == 0.0:
            log.info("no bias improves val mean without cav regression — lever declined (keep b=0)")
            return
        # apply ONCE to test: pooled + per-vendor (the verdict predicts the biggest lift on unseen vendors)
        test_cases = RvOmission._score_cases(s, test_frame, (0.0, b_star))
        RvOmission._report(RvOmission._agg(test_cases, (0.0, b_star)), f"TEST pooled n={len(test_cases)}", b_star)
        for v in sorted({str(v) for v, _ in test_cases}):
            sub = [(vv, pc) for vv, pc in test_cases if str(vv) == v]
            RvOmission._report(RvOmission._agg(sub, (0.0, b_star)), f"TEST {v} n={len(sub)}", b_star)

    @staticmethod
    def _gated(s: Session, val_frame: Any, test_frame: Any) -> None:  # pragma: no cover
        """ru27: fit b* on VAL under the CONDITIONAL gate (τ=GATE_TAU, signature-derived), then apply
        once to TEST and report base vs GLOBAL vs GATED per vendor. The verdict: gating recovers the
        collapsed vendors (Canon) as the global bias did, but leaves healthy-RV vendors (GE) untouched."""
        val_sweep = RvOmission._agg(RvOmission._score_cases(s, val_frame, BIASES, GATE_TAU), BIASES)
        log.info("=== VAL gated dose-response (τ=%.2f) n=%d ===  bias  RV     myo    cav    mean",
                 GATE_TAU, len(val_frame))
        for b in BIASES:
            dd = val_sweep[b]
            log.info("  b=%-4.1f %.3f  %.3f  %.3f  %.3f", b, dd["RV"], dd["myo"], dd["cav"], RvOmission._mean(dd))
        b_star = RvOmission.select_bias(val_sweep)
        log.info(">>> selected gated RV logit-bias b*=%.1f (val mean %.3f vs b=0 %.3f)", b_star,
                 RvOmission._mean(val_sweep[b_star]), RvOmission._mean(val_sweep[0.0]))
        if b_star == 0.0:
            log.info("no gated bias improves val mean without cav regression — lever declined (keep b=0)")
            return
        arms = RvOmission._score_arms(s, test_frame, b_star, GATE_TAU)
        for label, sub in [(f"TEST pooled n={len(arms)}", arms),
                           *[(f"TEST {v}", [(vv, row) for vv, row in arms if str(vv) == v])
                             for v in sorted({str(v) for v, _ in arms})]]:
            RvOmission._report(RvOmission._agg_arm(sub, "global", b_star), f"{label} GLOBAL", b_star)
            RvOmission._report(RvOmission._agg_arm(sub, "gated", b_star), f"{label} GATED", b_star)

    @staticmethod
    def run(args: argparse.Namespace) -> None:  # pragma: no cover
        s = RvOmission._load(args.run)
        if args.mode == "probe":
            RvOmission._probe(s, splits.Splits.eval_set(args.testset))
            return
        meta = store.load(list(s.cfg.generator.data.sources))
        model_split = ModelSplit(s.cfg.generator.data, meta)
        if args.mode == "deficit":
            RvOmission._deficit(s, model_split.test)
            return
        fit = RvOmission._gated if args.mode == "gated" else RvOmission._bias
        fit(s, model_split.val, model_split.test)
