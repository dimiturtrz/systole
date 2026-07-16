"""RV-omission diagnostics for a zero-real model — the reproducible core of the nttu epic (bd nttu.6).

Two modes, one entry (`python -m cardioseg.evaluation rv_omission --run <zero-real> --mode {probe,bias}`):

- **probe** (nttu.5 root-cause): on every slice where GT-RV is present but argmax omits it (<OMIT_PX RV
  pixels, PRE-largest-CC), is there raw RV softmax to recover (recall/threshold-fixable) or is it truly
  zero (coverage/detection-absent)? Splits the tail and names the winning class in the GT-RV region.
- **bias** (nttu.7/we55 lever): fit a constant added to the RV logit pre-softmax on VAL (leak-free dose-
  response → `select_bias`, best mean Dice with a cav-regression guard), then apply that one b* to the
  model's own cross-vendor TEST, reported pooled + PER-VENDOR. The verdict (49b7) predicts the biggest
  lift on the unseen vendors (GE/Canon), where RV collapses under OOD color. `core.inference.Inference`
  carries the fitted b* as an opt-in `logit_bias` for the real pipeline.

Zero-real models live in staging (unregistered), so `--run` is explicit (a run dir / alias / id resolvable
by the registry), e.g. `--run .staging/refac_proc`.
"""
import logging

import numpy as np
import torch
from jaxtyping import Float, Integer, UInt8

from core.data.static import splits
from core.data.static.splits import ModelSplit
from core.data.static.store import Store
from core.data.static.store.build import Build as store
from core.evaluate import Evaluate
from core.hparams import Hparams
from core.model import Model
from core.postprocess import Postprocess
from core.preprocessing.preprocess import Preprocess
from core.registry import Registry

log = logging.getLogger("cardioseg.rv_omission")

RV = 1
OMIT_PX = 20
FLIPS = ([], [2], [3], [2, 3])
CLASS_NAMES = ("bg", "RV", "myo", "cav")
FOREGROUND = {"RV": 1, "myo": 2, "cav": 3}
ACTIVATION_FLOOR = 0.05  # maxP below this = true-zero-activation (coverage), above = recall-recoverable
BIASES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)


class RvOmission:
    """RV-omission diagnostics: the pure classification/scoring cores (testable) + a GPU/IO shell
    (`run`, pragma-excluded). `probe` splits the omission tail recall-vs-coverage; `bias` sweeps the
    inference-time RV logit-bias recall lever."""

    @staticmethod
    def omission_row(rv_prob: Float[np.ndarray, "h w"], argmax: Integer[np.ndarray, "h w"],
                     gt: Integer[np.ndarray, "h w"], pred: Integer[np.ndarray, "h w"]) -> dict | None:
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
    def split_verdict(maxps: list[float]) -> dict:
        """Recall-vs-coverage split over the omitted slices' in-GT max RV softmax: how many carry
        recoverable activation (>= floor) vs are true-zero (coverage). Pure — the nttu.5 conclusion."""
        arr = np.array(maxps) if maxps else np.zeros(0)
        recoverable = int((arr >= ACTIVATION_FLOOR).sum())
        return {"n": len(arr), "recoverable": recoverable, "zero_activation": len(arr) - recoverable,
                "min": float(arr.min()) if arr.size else float("nan"),
                "med": float(np.median(arr)) if arr.size else float("nan"),
                "max": float(arr.max()) if arr.size else float("nan")}

    @staticmethod
    def biased_pred(logits: Float[torch.Tensor, "kd c h w"], d: int, bias: float) -> UInt8[np.ndarray, "d h w"]:
        """argmax label map after adding `bias` to the RV logit pre-softmax, TTA-averaged over FLIPS.
        `logits` is the stacked [K*D,C,H,W] forward; `d` the per-flip depth. The targeted-recall op."""
        biased = logits.clone()
        biased[:, RV] += bias
        probs = torch.softmax(biased, dim=1)
        flips = [torch.flip(probs[i * d:(i + 1) * d], dm) if dm else probs[i * d:(i + 1) * d]
                 for i, dm in enumerate(FLIPS)]
        return torch.stack(flips).mean(0).argmax(1).to(torch.uint8).cpu().numpy()

    @staticmethod
    def select_bias(sweep: dict[float, dict[str, float]], *, cav_guard: float = 0.01) -> float:
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
    def add_args(ap):
        ap.add_argument("--run", required=True,
                        help="zero-real model (run dir / alias / id), e.g. .staging/refac_proc")
        ap.add_argument("--mode", choices=("probe", "bias"), default="probe")
        ap.add_argument("--testset", default="cmrxmotion",
                        help="probe eval set (bias uses the model's own val/test split)")

    @staticmethod
    def _load(run_ref: str):  # pragma: no cover  (registry resolve + torch model load)
        run = Registry.resolve(run_ref)
        cfg = Hparams.from_json(run / "config.json")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Model.build_unet(cfg.model).to(device)
        model.load_state_dict(torch.load(run / "model.pth", map_location=device))
        model.eval()
        return model, cfg, device

    @staticmethod
    def _logits(model, vol_img, size: int, device: str):  # pragma: no cover  (GPU forward)
        xs = np.stack([Preprocess.fit_square(vol_img[z].astype(np.float32), size, 0.0)
                       for z in range(vol_img.shape[0])])
        with torch.no_grad():
            x = torch.from_numpy(xs)[:, None].to(device)
            batched = torch.cat([torch.flip(x, dm) if dm else x for dm in FLIPS], dim=0)
            return model(batched), x.shape[0]

    @staticmethod
    def _probe(model, cfg, device, frame):  # pragma: no cover  (GPU inference over the eval frame)
        size = cfg.generator.data.size
        rows = []
        for r in frame.iter_rows(named=True):
            case = Store.load_arrays(r["path"])
            gt = Preprocess.stack_slices(case["ed_gt"], size)
            lg, d = RvOmission._logits(model, case["ed_img"], size, device)
            probs = torch.softmax(lg[:d], dim=1)                 # identity-flip block = un-augmented
            rv_prob = probs[:, RV].cpu().numpy()
            pred = probs.argmax(1).to(torch.uint8).cpu().numpy()
            for z in range(gt.shape[0]):
                row = RvOmission.omission_row(rv_prob[z], pred[z], gt[z], pred[z])
                if row:
                    rows.append(row)
        verdict = RvOmission.split_verdict([x["maxp_in_gt"] for x in rows])
        log.info("omitted slices n=%d | recoverable(>=%.2f) %d | zero-activation %d | maxP min/med/max %.3f/%.3f/%.3f",
                 verdict["n"], ACTIVATION_FLOOR, verdict["recoverable"], verdict["zero_activation"],
                 verdict["min"], verdict["med"], verdict["max"])
        for x in sorted(rows, key=lambda r: r["maxp_in_gt"]):
            log.info("  gtpx %4d  maxP_gt %.3f  meanP_gt %.3f  win %-3s %.2f",
                     x["gtpx"], x["maxp_in_gt"], x["meanp_in_gt"], x["win"], x["win_frac"])
        return verdict

    @staticmethod
    def _mean(d: dict[str, float]) -> float:
        return float(np.mean([d[c] for c in FOREGROUND]))

    @staticmethod
    def _score_cases(model, cfg, device, frame, biases):  # pragma: no cover  (GPU forward per case)
        """Per-case foreground Dice at each bias, keeping the case vendor. Forward runs ONCE per case;
        the bias is a cheap logit add on the cached logits. Returns [(vendor, {b: {class: dice}})]."""
        size = cfg.generator.data.size
        cases = []
        for r in frame.iter_rows(named=True):
            case = Store.load_arrays(r["path"])
            gt = Preprocess.stack_slices(case["ed_gt"], size)
            lg, d = RvOmission._logits(model, case["ed_img"], size, device)
            per = {}
            for b in biases:
                pred = Postprocess.largest_cc_per_class(RvOmission.biased_pred(lg, d, b))
                per[b] = {c: float(Evaluate.dice(pred, gt, lab)) for c, lab in FOREGROUND.items()}
            cases.append((r.get("vendor"), per))
        return cases

    @staticmethod
    def _agg(cases, biases) -> dict[float, dict[str, float]]:
        """Pooled mean per-bias per-class over [(vendor, {b:{class:dice}})]."""
        return {b: {c: float(np.mean([pc[b][c] for _, pc in cases])) for c in FOREGROUND} for b in biases}

    @staticmethod
    def _report(agg, label, b_star):  # pragma: no cover  (logging)
        base, biased = agg[0.0], agg[b_star]
        log.info("=== %s ===  RV     myo    cav    mean", label)
        for tag, dd in ((f"b=0.0 {'':4}", base), (f"b={b_star:<4.1f}", biased)):
            log.info("  %s %.3f  %.3f  %.3f  %.3f", tag, dd["RV"], dd["myo"], dd["cav"], RvOmission._mean(dd))
        log.info("  Δ       %+.3f %+.3f %+.3f %+.3f", biased["RV"] - base["RV"], biased["myo"] - base["myo"],
                 biased["cav"] - base["cav"], RvOmission._mean(biased) - RvOmission._mean(base))

    @staticmethod
    def _bias(model, cfg, device, val_frame, test_frame):  # pragma: no cover  (GPU sweep + fit + transfer)
        # fit on VAL (leak-free): full dose-response, then select b* (best mean, cav-guarded)
        val_sweep = RvOmission._agg(RvOmission._score_cases(model, cfg, device, val_frame, BIASES), BIASES)
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
        test_cases = RvOmission._score_cases(model, cfg, device, test_frame, (0.0, b_star))
        RvOmission._report(RvOmission._agg(test_cases, (0.0, b_star)), f"TEST pooled n={len(test_cases)}", b_star)
        for v in sorted({str(v) for v, _ in test_cases}):
            sub = [(vv, pc) for vv, pc in test_cases if str(vv) == v]
            RvOmission._report(RvOmission._agg(sub, (0.0, b_star)), f"TEST {v} n={len(sub)}", b_star)

    @staticmethod
    def run(args):  # pragma: no cover  (loads a zero-real model + GPU inference over eval frames)
        model, cfg, device = RvOmission._load(args.run)
        if args.mode == "probe":
            RvOmission._probe(model, cfg, device, splits.Splits.eval_set(args.testset))
        else:
            meta = store.load(list(cfg.generator.data.sources))
            model_split = ModelSplit(cfg.generator.data, meta)
            RvOmission._bias(model, cfg, device, model_split.val, model_split.test)
