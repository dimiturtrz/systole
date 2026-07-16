"""RV-omission diagnostics for a zero-real model — the reproducible core of the nttu epic (bd nttu.6).

Two modes, one entry (`python -m cardioseg.evaluation rv_omission --run <zero-real> --mode {probe,bias}`):

- **probe** (nttu.5 root-cause): on every slice where GT-RV is present but argmax omits it (<OMIT_PX RV
  pixels, PRE-largest-CC), is there raw RV softmax to recover (recall/threshold-fixable) or is it truly
  zero (coverage/detection-absent)? Splits the tail and names the winning class in the GT-RV region.
- **bias** (nttu.7 lever): sweep a constant added to the RV logit pre-softmax; report per-class Dice on
  VAL (selection, leak-free) + a test set (transfer). Confirms TARGETED recall recovers the tail for free.

Zero-real models live in staging (unregistered), so `--run` is explicit (a run dir / alias / id resolvable
by the registry), e.g. `--run .staging/refac_proc`.
"""
import logging

import numpy as np
import torch

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
    def omission_row(rv_prob: np.ndarray, argmax: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> dict | None:
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
    def biased_pred(logits: torch.Tensor, d: int, bias: float) -> np.ndarray:
        """argmax label map after adding `bias` to the RV logit pre-softmax, TTA-averaged over FLIPS.
        `logits` is the stacked [K*D,C,H,W] forward; `d` the per-flip depth. The targeted-recall op."""
        biased = logits.clone()
        biased[:, RV] += bias
        probs = torch.softmax(biased, dim=1)
        flips = [torch.flip(probs[i * d:(i + 1) * d], dm) if dm else probs[i * d:(i + 1) * d]
                 for i, dm in enumerate(FLIPS)]
        return torch.stack(flips).mean(0).argmax(1).to(torch.uint8).cpu().numpy()

    @staticmethod
    def add_args(ap):
        ap.add_argument("--run", required=True,
                        help="zero-real model (run dir / alias / id), e.g. .staging/refac_proc")
        ap.add_argument("--mode", choices=("probe", "bias"), default="probe")
        ap.add_argument("--testset", default="cmrxmotion", help="eval set for probe / bias-transfer")

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
    def _bias(model, cfg, device, val_frame, test_frame):  # pragma: no cover  (GPU sweep over two frames)
        size = cfg.generator.data.size

        def score(frame):
            cache = []
            for r in frame.iter_rows(named=True):
                case = Store.load_arrays(r["path"])
                gt = Preprocess.stack_slices(case["ed_gt"], size)
                lg, d = RvOmission._logits(model, case["ed_img"], size, device)
                cache.append((lg, d, gt))
            out = {}
            for b in BIASES:
                per = {c: [] for c in FOREGROUND}
                for lg, d, gt in cache:
                    pred = Postprocess.largest_cc_per_class(RvOmission.biased_pred(lg, d, b))
                    for c, lab in FOREGROUND.items():
                        per[c].append(Evaluate.dice(pred, gt, lab))
                out[b] = {c: float(np.mean(v)) for c, v in per.items()}
            return out

        for name, frame in (("VAL (selection)", val_frame), ("TEST (transfer)", test_frame)):
            res = score(frame)
            log.info("=== %s n=%d ===  bias  RV     myo    cav    mean", name, len(frame))
            for b in BIASES:
                dd = res[b]
                log.info("  b=%-4.1f %.3f  %.3f  %.3f  %.3f", b, dd["RV"], dd["myo"], dd["cav"],
                         float(np.mean([dd[c] for c in FOREGROUND])))
        return None

    @staticmethod
    def run(args):  # pragma: no cover  (loads a zero-real model + GPU inference over eval frames)
        model, cfg, device = RvOmission._load(args.run)
        test_frame = splits.Splits.eval_set(args.testset)
        if args.mode == "probe":
            RvOmission._probe(model, cfg, device, test_frame)
        else:
            meta = store.load(list(cfg.generator.data.sources))
            val_frame = ModelSplit(cfg.generator.data, meta).val
            RvOmission._bias(model, cfg, device, val_frame, test_frame)
