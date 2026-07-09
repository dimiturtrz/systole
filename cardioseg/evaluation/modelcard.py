"""Auto-generate a per-run model card from artifacts — so the card always matches the shipped model.

Reads runs/<run>/{config.json, metrics.json} and renders runs/<run>/MODEL_CARD.md: model details
+ the data/split criteria + performance tables, all pulled from the artifacts (never hand-typed),
plus standard templated narrative (intended-use, limitations). Called automatically at training end
(cardioseg.training.train) and re-runnable:

    python -m cardioseg.evaluation.modelcard --run runs/gen

The curated repo card (cardioseg/MODEL_CARD.md) keeps the deeper analysis; this is the factual,
always-in-sync per-run card.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from core.data.static.labels import CLASS_NAMES
from core.data.static.reference import Reference
from core.registry import Registry

log = logging.getLogger("cardioseg.modelcard")


class ModelCard:
    """Per-run model-card generation — the pure renderer (`render_card`) + its perf/reference helpers,
    plus the disk shell (`generate`) that reads config.json/metrics.json + probes the reference store."""

    _ORDER = tuple(reversed(CLASS_NAMES))  # display order: LV-cav, LV-myo, RV
    _REF_KEYS = (("ef_normal", "Normal LV EF"), ("edv_normal_ml", "Normal EDV"), ("esv_normal_ml", "Normal ESV"))

    @staticmethod
    def _perf_table(r: dict) -> str:
        dice, b = r.get("dice", {}), (r.get("boundary") or {})
        rows = ["| structure | Dice | HD95 (mm) | ASSD (mm) |", "|---|---|---|---|"]
        for k in ModelCard._ORDER:
            if k not in dice:
                continue
            bd = b.get(k, {}) if b else {}
            hd = f"{bd['hd95']:.2f}" if bd.get("hd95") == bd.get("hd95") and "hd95" in bd else "—"
            ad = f"{bd['assd']:.2f}" if "assd" in bd and bd["assd"] == bd["assd"] else "—"
            rows.append(f"| {k} | {dice[k]:.3f} | {hd} | {ad} |")
        rows.append(f"| **mean** | **{r.get('dice_mean', float('nan')):.3f}** | | |")
        return "\n".join(rows)

    @staticmethod
    def reference_rows(provenances: dict) -> list[str]:
        """Pure provenance -> markdown rows: `{key: provenance_dict}` -> the reference-range bullet list.
        A key whose provenance is missing or whose `value` isn't a [lo,hi] list is skipped. Extracted from
        `_reference_section` (whose `Reference()` store probe is the shell) so the row formatting +
        unit/n/based-on rendering is testable off synthetic provenance dicts."""
        rows = []
        for key, label in ModelCard._REF_KEYS:
            p = provenances.get(key)
            if p and isinstance(p.get("value"), list):
                unit = "%" if key == "ef_normal" else " mL"
                rows.append(f"- {label}: **{p['value'][0]}–{p['value'][1]}{unit}** (p5–p95, n={p.get('n','?')}). "
                            f"<sub>{p.get('based_on','')}</sub>")
        return rows

    @staticmethod
    def _reference_section() -> list[str]:  # pragma: no cover  (probes the on-disk <data>/reference/ store; row formatting is reference_rows)
        """Surface the derived reference ranges (with cohort provenance) IF the local reference store is
        present; omitted entirely when absent (graceful fallback — the card still generates on a fresh
        clone with no <data>/reference/). Shows provenance regardless of verified, but only verified
        values would actually be used by consumers."""
        ref = Reference()
        if not ref.present():
            return []
        rows = ModelCard.reference_rows({key: ref.provenance("normal_ranges", key) for key, _ in ModelCard._REF_KEYS})
        if not rows:
            return []
        return ["", "## Reference ranges (derived from our GT, for context)",
                "*From the local reference store (`<data>/reference/`, not committed); each traces to its "
                "cohort. Clinical context only — the model is not evaluated against these.*", *rows]

    @staticmethod
    def render_card(run_name: str, cfg: dict, m: dict, ref_section: list[str] | None = None) -> str:
        """The pure card renderer: (config dict, metrics dict) -> the MODEL_CARD.md TEXT. No file IO — the
        disk read + reference-store probe live in `generate`, so the whole markdown (model block, split
        criteria, val/test perf tables, narrative) is testable off a synthetic config+metrics pair."""
        res, d, mdl, ls = m["results"], (cfg.get("generator") or cfg)["data"], cfg["model"], cfg["loss"]

        parts = [
            f"# Model Card — cardioseg run `{run_name}`",
            "*Auto-generated from `config.json` + `metrics.json` at training end — numbers are never "
            "hand-typed, so this card always matches the shipped model. Deeper analysis lives in the "
            "curated [`cardioseg/MODEL_CARD.md`](../../cardioseg/MODEL_CARD.md).*",
            "",
            "## Model",
            f"- 2D MONAI U-Net — channels {tuple(mdl['channels'])}, strides {tuple(mdl['strides'])}, "
            f"{mdl['res_units']} res-units, {mdl['out_channels']} classes (bg/RV/myo/LV-cav).",
            f"- Loss: `{ls['kind']}`. Optimizer Adam (lr {cfg['lr']}), AMP, early-stop "
            f"(patience {cfg['patience']}, {cfg['epochs']}-epoch ceiling), seed {cfg['seed']}.",
            f"- Trained on {m['n_train']} subjects, validated on {m['n_val']}.",
            "",
            "## Data & split (criteria over the data cloud)",
            f"- Sources: {list(d['sources'])}. Resample in-plane {d['inplane']} mm, N4={d['n4']}.",
            f"- **Held out (test):** datasets={list(d['test_datasets'])}, vendors={list(d['test_vendors'])}. "
            "Train/val = the rest, labelled. The criteria ARE the split (this card + config.json record it).",
            "",
            "## Performance",
        ]
        if "val" in res:
            parts += ["### Validation (in-domain)", ModelCard._perf_table(res["val"]),
                      f"\nEF vs GT: **MAE {res['val'].get('ef_mae', float('nan')):.1f}%** (n={len(res['val'].get('ef_rows', []))})."]
        test_key = next((k for k in res if k.startswith("test")), None)
        if test_key:
            t = res[test_key]
            parts += ["", "### Held-out test", ModelCard._perf_table(t),
                      f"\nEF vs GT: **MAE {t.get('ef_mae', float('nan')):.1f}%** (n={len(t.get('ef_rows', []))})."]
        parts += ref_section or []
        parts += [
            "",
            "## Intended use & limitations",
            "- **Research / benchmark only — NOT clinical, not a device.** Public benchmark data; no "
            "DICOM/PII handling, no prospective or regulatory validation.",
            "- Single modality (cine MRI short-axis), 2D slice model. Public-benchmark performance ≠ "
            "deployment performance.",
            "- EF carries a systematic under-prediction (end-systolic cavity over-segmentation); it is "
            "partly **irreducible** (partial-volume on a small ES cavity) and reported, not patched. See "
            "the curated card + `research/deep_dives/2026-06-21_ef-bias-mechanism-esv-overseg.md`.",
            "- Unseen-vendor evidence (Canon) is thin (n≈9). Demographic fairness (sex/age) is not "
            "audited — the data lacks coverage on the held-out set.",
            "",
            f"**Reproduce:** `runs/{run_name}/config.json` fully specifies this run "
            "(`core.hparams.from_json` → `train_seg`).",
        ]
        return "\n".join(parts) + "\n"

    @staticmethod
    def generate(run_dir: str | Path) -> Path:  # pragma: no cover  (reads config.json/metrics.json + reference store, writes MODEL_CARD.md — render_card is the pure core)
        run = Path(run_dir)
        cfg = json.loads((run / "config.json").read_text())
        m = json.loads((run / "metrics.json").read_text())
        out = run / "MODEL_CARD.md"
        out.write_text(ModelCard.render_card(run.name, cfg, m, ModelCard._reference_section()))
        return out

    @staticmethod
    def add_args(ap):
        ap.add_argument("--run", required=True, help="run dir with config.json + metrics.json")

    @staticmethod
    def run(args):  # pragma: no cover  (CLI: resolve registry ref + generate the card file)
        log.info(f"wrote {ModelCard.generate(Registry.resolve(args.run))}")
