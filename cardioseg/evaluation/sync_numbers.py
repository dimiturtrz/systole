"""Fill the docs' number tables from cardioseg/RESULTS.json (the one canonical source) — so a metric
lives in ONE place and the markdown can't drift. Each renderable table sits between HTML markers:

    <!-- results:compare -->
    ...auto-generated, do not hand-edit...
    <!-- /results:compare -->

Run after regenerating RESULTS.json (cardioseg.evaluation results):  python -m cardioseg.evaluation sync_numbers
Prose-embedded numbers ("Dice 0.89" mid-sentence) are NOT templated — keep those few; the tables
(where the drift happened) are now generated.
"""
import json
import logging
import re
from pathlib import Path

from core.data.static.mri.pathology import PathologyClass

log = logging.getLogger("cardioseg.sync_numbers")

ROOT = Path(__file__).resolve().parents[2]  # repo root (…/cardioseg/evaluation/ -> repo)
R = json.loads((ROOT / "cardioseg/RESULTS.json").read_text())
_A = R["flagship"]["acdc"]      # val: ACDC (held-out centre / protocol shift)
_C = R["flagship"]["canon"]     # test: unseen vendor Canon
_G = R["flagship"]["ge"]        # test: unseen vendor GE
_NN = R["nnunet"]
_E = R["efficiency"]


class SyncNumbers:
    """Doc number-syncing: the table/prose renderers off the committed RESULTS.json + the pure
    marker-block injection (the free helpers folded in as staticmethods). The per-file read/write loop
    stays in module-level `main` (the shell); everything here is pure and testable."""

    @staticmethod
    def compare() -> str:   # ours vs SOTA on the UNSEEN vendors (Canon, GE) — same split + macro eval
        o, n = _E["ours"], _E["nnunet"]
        return "\n".join([
            "| model | params | FLOPs | Canon Dice | Canon EF MAE | GE Dice | GE EF MAE |",
            "|---|---|---|---|---|---|---|",
            f"| **ours** (ONNX-deployable) | **{o['params']}** | **{o['flops']}** | "
            f"{_C['dice']['mean']:.2f} | {_C['ef_mae']}% | {_G['dice']['mean']:.2f} | {_G['ef_mae']}% |",
            f"| nnU-Net (SOTA baseline) | {n['params']} | {n['flops']} | "
            f"{_NN['canon']['dice']['mean']:.2f} | **{_NN['canon']['ef_mae']}%** | "
            f"{_NN['ge']['dice']['mean']:.2f} | **{_NN['ge']['ef_mae']}%** |",
        ])

    @staticmethod
    def acdc() -> str:
        d, h, s = _A["dice"], _A["hd95"], _A["assd"]
        rows = ["| structure | Dice | HD95 (mm) | ASSD (mm) |", "|---|---|---|---|"]
        for k, lab in [("LV-cav", "LV cavity"), ("LV-myo", "LV myocardium"), ("RV", "RV cavity")]:
            rows.append(f"| {lab} | {d[k]:.2f} | {h[k]:.1f} | {s[k]:.2f} |")
        rows.append(f"| **mean** | **{d['mean']:.2f}** | | |")
        return "\n".join(rows)

    @staticmethod
    def strata() -> str:
        st = _A.get("strata", {})
        order = [(PathologyClass.DILATED, "dilated (DCM)"), (PathologyClass.ISCHEMIC, "ischemic (MINF)"),
                 (PathologyClass.RV_CONGENITAL, "rv_congenital"), (PathologyClass.NORMAL, "normal (NOR)"),
                 (PathologyClass.HYPERTROPHIC, "**hypertrophic (HCM)**")]
        rows = ["| pathology | gtEF | mean Dice | EF MAE | EF bias |", "|---|---|---|---|---|"]
        for g, lab in order:
            s = st.get(g)
            if s:
                rows.append(
                    f"| {lab} | {s['gt_ef_mean']:.0f}% | {s['dice_mean']:.2f} | "
                    f"{s['ef_mae']:.1f}% | {s['ef_bias']:+.1f}% |")
        return "\n".join(rows)

    @staticmethod
    def _cal(ax: dict) -> str:
        """The disclosed calibrated-EF cell for one axis: `raw → cal`, or `—` before RESULTS.json carries it."""
        return f"{ax['ef_mae']} → **{ax['ef_mae_cal']}**%" if "ef_mae_cal" in ax else "—"

    @staticmethod
    def axis() -> str:
        cal = R.get("ef_calibration")
        rows = [
            "| held-out axis | role | n | mean Dice | EF MAE | EF MAE (cal)† |", "|---|---|---|---|---|---|",
            f"| **ACDC** — centre / protocol shift | val | {_A['n']} | {_A['dice']['mean']:.2f} | "
            f"{_A['ef_mae']}% | {SyncNumbers._cal(_A)} |",
            f"| **Canon** — unseen vendor | test | {_C['n']} | {_C['dice']['mean']:.2f} | "
            f"{_C['ef_mae']}% | {SyncNumbers._cal(_C)} |",
            f"| **GE** — unseen vendor | test | {_G['n']} | {_G['dice']['mean']:.2f} | "
            f"{_G['ef_mae']}% | {SyncNumbers._cal(_G)} |",
        ]
        if cal:
            rows.append(f"\n† post-hoc linear EF correction `ef_corr = {cal['slope']}·ef_pred + {cal['intercept']}`, "
                        "fit on VAL (ACDC) only, applied to all axes — **disclosed, not substituted** for the "
                        "reported (raw) EF. Vendor-systematic bias transfers OOD; see "
                        "`interpretations/ef/2026-07-15_ef_defensibility.md`.")
        return "\n".join(rows)

    @staticmethod
    def cardcompare() -> str:  # unseen-vendor comparison used in both model cards
        return "\n".join([
            "| unseen-vendor (held out) | nnU-Net (50ep/fold0) | this model |", "|---|---|---|",
            f"| Canon mean Dice | {_NN['canon']['dice']['mean']:.3f} | {_C['dice']['mean']:.3f} |",
            f"| GE mean Dice | {_NN['ge']['dice']['mean']:.3f} | {_G['dice']['mean']:.3f} |",
            f"| Canon / GE EF MAE | **{_NN['canon']['ef_mae']} / {_NN['ge']['ef_mae']}%** | "
            f"{_C['ef_mae']} / {_G['ef_mae']}% |",
        ])

    @staticmethod
    def nnucompare() -> str:  # nnU-Net README: per-structure rows + Δ on held-out GE (n=69, the larger vendor)
        ours, nnunet = _G["dice"], _NN["ge"]["dice"]

        def dice_delta(structure):
            return (nnunet[structure] - ours[structure]) * 100
        return "\n".join([
            "| segmenter (held-out GE, n=69) | mean Dice | LV-cav | myo | RV | EF MAE | notes |",
            "|---|---|---|---|---|---|---|",
            f"| our 2D U-Net (+ heavy aug + early stop + largest-CC + TTA) | "
            f"{ours['mean']:.3f} | {ours['LV-cav']:.3f} | "
            f"{ours['LV-myo']:.3f} | {ours['RV']:.3f} | {_G['ef_mae']}% | deployable / ONNX |",
            f"| **nnU-Net** (50 ep, 1 fold) | **{nnunet['mean']:.3f}** | "
            f"**{nnunet['LV-cav']:.3f}** | **{nnunet['LV-myo']:.3f}** | "
            f"**{nnunet['RV']:.3f}** | **{_NN['ge']['ef_mae']}%** | baseline / not deployed |",
            f"| Δ (nnU-Net − ours) | +{dice_delta('mean'):.1f} | "
            f"+{dice_delta('LV-cav'):.1f} | +{dice_delta('LV-myo'):.1f} | "
            f"+{dice_delta('RV'):.1f} | {_NN['ge']['ef_mae'] - _G['ef_mae']:+.1f} | |",
        ])

    @staticmethod
    def headline() -> str:
        return (f"Unseen-vendor (held out: Canon n={_C['n']} + GE n={_G['n']}) mean Dice "
                f"**{_C['dice']['mean']:.2f}** / **{_G['dice']['mean']:.2f}**, "
                f"EF MAE {_C['ef_mae']} / {_G['ef_mae']}%; "
                f"ACDC centre-shift (val, n={_A['n']}) Dice **{_A['dice']['mean']:.2f}**, EF MAE **{_A['ef_mae']}%** "
                f"(bias {_A['ef_bias']:+.1f}%, LoA [{_A['ef_loa'][0]:.0f}, {_A['ef_loa'][1]:+.0f}]).")

    @staticmethod
    def inject_blocks(txt: str, blocks: dict) -> tuple[str, int]:
        """Replace each `<!-- results:KEY -->...<!-- /results:KEY -->` span with `\\n{fn()}\\n` between the
        markers. Returns (new_text, n_blocks_substituted). Pure string transform (the file read/write is the
        shell in `main`) — so the marker-matching + idempotent re-render is testable on an in-memory string.
        A block present in the text but not in `blocks` is left untouched; a block in `blocks` not in the
        text is skipped. `fn()` output that contains no marker keeps the operation idempotent."""
        total = 0
        for key, fn in blocks.items():
            pat = re.compile(rf"(<!-- results:{key} -->).*?(<!-- /results:{key} -->)", re.DOTALL)
            if pat.search(txt):
                txt = pat.sub(lambda m, fn=fn: f"{m.group(1)}\n{fn()}\n{m.group(2)}", txt)
                total += 1
        return txt, total

    @staticmethod
    def add_args(ap):
        pass

    @staticmethod
    def run(args):  # pragma: no cover  (per-file read/write loop over the doc TARGETS; inject_blocks is the pure core)
        total = 0
        for f in TARGETS:
            p = ROOT / f
            txt = orig = p.read_text()
            txt, n = SyncNumbers.inject_blocks(txt, BLOCKS)
            total += n
            if txt != orig:
                p.write_text(txt)
                log.info(f"  updated {f}")
        log.info(f"synced {total} blocks from cardioseg/RESULTS.json")


BLOCKS = {"compare": SyncNumbers.compare, "acdc": SyncNumbers.acdc, "strata": SyncNumbers.strata,
          "headline": SyncNumbers.headline, "axis": SyncNumbers.axis,
          "cardcompare": SyncNumbers.cardcompare, "nnucompare": SyncNumbers.nnucompare}
TARGETS = ["README.md", "cardioseg/README.md", "cardioseg/MODEL_CARD.md",
           "baselines/nnunet/MODEL_CARD.md", "baselines/nnunet/README.md"]
