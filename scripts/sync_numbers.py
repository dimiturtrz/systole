"""Fill the docs' number tables from cardioseg/RESULTS.json (the one canonical source) — so a metric
lives in ONE place and the markdown can't drift. Each renderable table sits between HTML markers:

    <!-- results:compare -->
    ...auto-generated, do not hand-edit...
    <!-- /results:compare -->

Run after regenerating RESULTS.json (cardioseg.evaluation.results):  python scripts/sync_numbers.py
Prose-embedded numbers ("Dice 0.89" mid-sentence) are NOT templated — keep those few; the tables
(where the drift happened) are now generated.
"""
import json
import re
from pathlib import Path

R = json.loads(Path("cardioseg/RESULTS.json").read_text())
_A = R["flagship"]["acdc"]
_C = R["flagship"]["canon"]
_NN = R["nnunet"]
_E = R["efficiency"]


def compare() -> str:
    d, n = _A["dice"], _NN["acdc"]["dice"]
    return "\n".join([
        "| | params | FLOPs | LV-cav | myo | RV | **mean Dice** | EF MAE |",
        "|---|---|---|---|---|---|---|---|",
        f"| **ours** (ONNX-deployable) | **{_E['ours']['params']}** | **{_E['ours']['flops']}** | "
        f"{d['LV-cav']:.2f} | {d['LV-myo']:.2f} | {d['RV']:.2f} | **{d['mean']:.2f}** | {_A['ef_mae']}% |",
        f"| nnU-Net (SOTA baseline) | {_E['nnunet']['params']} | {_E['nnunet']['flops']} | "
        f"{n['LV-cav']:.2f} | {n['LV-myo']:.2f} | {n['RV']:.2f} | **{n['mean']:.2f}** | {_NN['acdc']['ef_mae']}% |",
    ])


def acdc() -> str:
    d, h, s = _A["dice"], _A["hd95"], _A["assd"]
    rows = ["| structure | Dice | HD95 (mm) | ASSD (mm) |", "|---|---|---|---|"]
    for k, lab in [("LV-cav", "LV cavity"), ("LV-myo", "LV myocardium"), ("RV", "RV cavity")]:
        rows.append(f"| {lab} | {d[k]:.2f} | {h[k]:.1f} | {s[k]:.2f} |")
    rows.append(f"| **mean** | **{d['mean']:.2f}** | | |")
    return "\n".join(rows)


def strata() -> str:
    st = _A.get("strata", {})
    order = [("dilated", "dilated (DCM)"), ("ischemic", "ischemic (MINF)"), ("rv_congenital", "rv_congenital"),
             ("normal", "normal (NOR)"), ("hypertrophic", "**hypertrophic (HCM)**")]
    rows = ["| pathology | gtEF | mean Dice | EF MAE | EF bias |", "|---|---|---|---|---|"]
    for g, lab in order:
        s = st.get(g)
        if s:
            rows.append(f"| {lab} | {s['gt_ef_mean']:.0f}% | {s['dice_mean']:.2f} | {s['ef_mae']:.1f}% | {s['ef_bias']:+.1f}% |")
    return "\n".join(rows)


def headline() -> str:
    return (f"ACDC-150 mean Dice **{_A['dice']['mean']:.2f}**, EF MAE **{_A['ef_mae']}%** "
            f"(bias {_A['ef_bias']:+.1f}%, LoA [{_A['ef_loa'][0]:.0f}, {_A['ef_loa'][1]:+.0f}]); "
            f"Canon-9 unseen-vendor **{_C['dice']['mean']:.2f}**; nnU-Net baseline "
            f"**{_NN['acdc']['dice']['mean']:.3f}** Dice / {_NN['acdc']['ef_mae']}% EF.")


BLOCKS = {"compare": compare, "acdc": acdc, "strata": strata, "headline": headline}
TARGETS = ["README.md", "cardioseg/README.md", "cardioseg/MODEL_CARD.md",
           "baselines/nnunet/MODEL_CARD.md", "baselines/nnunet/README.md"]


def main():
    total = 0
    for f in TARGETS:
        p = Path(f)
        txt = orig = p.read_text()
        for key, fn in BLOCKS.items():
            pat = re.compile(rf"(<!-- results:{key} -->).*?(<!-- /results:{key} -->)", re.DOTALL)
            if pat.search(txt):
                txt = pat.sub(lambda m, fn=fn: f"{m.group(1)}\n{fn()}\n{m.group(2)}", txt)
                total += 1
        if txt != orig:
            p.write_text(txt)
            print(f"  updated {f}")
    print(f"synced {total} blocks from cardioseg/RESULTS.json")


if __name__ == "__main__":
    main()
