"""Fill the docs' number tables from cardioseg/RESULTS.json (the one canonical source) — so a metric
lives in ONE place and the markdown can't drift. Each renderable table sits between HTML markers:

    <!-- results:compare -->
    ...auto-generated, do not hand-edit...
    <!-- /results:compare -->

Run after regenerating RESULTS.json (cardioseg.evaluation.results):  python -m cardioseg.evaluation.sync_numbers
Prose-embedded numbers ("Dice 0.89" mid-sentence) are NOT templated — keep those few; the tables
(where the drift happened) are now generated.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root (…/cardioseg/evaluation/ -> repo)
R = json.loads((ROOT / "cardioseg/RESULTS.json").read_text())
_A = R["flagship"]["acdc"]      # val: ACDC (held-out centre / protocol shift)
_C = R["flagship"]["canon"]     # test: unseen vendor Canon
_G = R["flagship"]["ge"]        # test: unseen vendor GE
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


def axis() -> str:
    return "\n".join([
        "| held-out axis | role | n | mean Dice | EF MAE |", "|---|---|---|---|---|",
        f"| **ACDC** — centre / protocol shift | val | {_A['n']} | {_A['dice']['mean']:.2f} | {_A['ef_mae']}% |",
        f"| **Canon** — unseen vendor | test | {_C['n']} | {_C['dice']['mean']:.2f} | {_C['ef_mae']}% |",
        f"| **GE** — unseen vendor | test | {_G['n']} | {_G['dice']['mean']:.2f} | {_G['ef_mae']}% |",
    ])


def cardcompare() -> str:  # 3-metric comparison used in both model cards
    a, n = _A, _NN["acdc"]
    return "\n".join([
        "| ACDC-150 | nnU-Net (50ep/fold0) | this model |", "|---|---|---|",
        f"| mean Dice | {n['dice']['mean']:.3f} | {a['dice']['mean']:.3f} |",
        f"| EF MAE / bias | {n['ef_mae']}% / {n['ef_bias']:+.1f}% | {a['ef_mae']}% / {a['ef_bias']:+.1f}% |",
        f"| Canon-9 Dice | {_NN['canon']['dice_mean']:.3f} | {_C['dice']['mean']:.2f} |",
    ])


def nnucompare() -> str:  # nnU-Net README: per-structure rows + Δ (Dice points = ×100)
    a, n, e = _A["dice"], _NN["acdc"]["dice"], _E
    dd = lambda k: (n[k] - a[k]) * 100
    return "\n".join([
        "| segmenter | mean Dice | LV-cav | myo | RV | EF MAE | notes |", "|---|---|---|---|---|---|---|",
        f"| our 2D U-Net (+ heavy aug + early stop + largest-CC + TTA) | {a['mean']:.3f} | {a['LV-cav']:.3f} | "
        f"{a['LV-myo']:.3f} | {a['RV']:.3f} | {_A['ef_mae']}% | deployable / ONNX |",
        f"| **nnU-Net** (50 ep, 1 fold) | **{n['mean']:.3f}** | **{n['LV-cav']:.3f}** | **{n['LV-myo']:.3f}** | "
        f"**{n['RV']:.3f}** | **{_NN['acdc']['ef_mae']}%** | baseline / not deployed |",
        f"| Δ (nnU-Net − ours) | +{dd('mean'):.1f} | +{dd('LV-cav'):.1f} | +{dd('LV-myo'):.1f} | "
        f"+{dd('RV'):.1f} | {_NN['acdc']['ef_mae'] - _A['ef_mae']:+.1f} | |",
    ])


def headline() -> str:
    return (f"Unseen-vendor (held out: Canon n={_C['n']} + GE n={_G['n']}) mean Dice "
            f"**{_C['dice']['mean']:.2f}** / **{_G['dice']['mean']:.2f}**, EF MAE {_C['ef_mae']} / {_G['ef_mae']}%; "
            f"ACDC centre-shift (val, n={_A['n']}) Dice **{_A['dice']['mean']:.2f}**, EF MAE **{_A['ef_mae']}%** "
            f"(bias {_A['ef_bias']:+.1f}%, LoA [{_A['ef_loa'][0]:.0f}, {_A['ef_loa'][1]:+.0f}]).")


BLOCKS = {"compare": compare, "acdc": acdc, "strata": strata, "headline": headline,
          "axis": axis, "cardcompare": cardcompare, "nnucompare": nnucompare}
TARGETS = ["README.md", "cardioseg/README.md", "cardioseg/MODEL_CARD.md",
           "baselines/nnunet/MODEL_CARD.md", "baselines/nnunet/README.md"]


def main():
    total = 0
    for f in TARGETS:
        p = ROOT / f
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
