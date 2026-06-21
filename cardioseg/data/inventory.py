"""Build the data cloud: one unified row per subject across all datasets.

The dataset adapters are the metadata-unification layer; this walks them and emits a single
table (dataset, id, vendor, raw + harmonized pathology, field, centre, demographics). null is a
valid value (= unknown). Saved to <data>/processed/inventory.csv (gitignored, regenerable) —
the input for choosing train/val/test roles + the model card. Fast: uses adapter.meta(), no image load.

    python -m cardioseg.data.inventory
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from cardioseg.config import data_root
from cardioseg.data.mri.pathology import harmonize
from cardioseg.data.mri.registry import dataset_names, get_adapter

FIELDS = ["dataset", "id", "vendor", "pathology_raw", "pathology",
          "field_T", "centre", "age", "sex", "height", "weight"]


def build_rows() -> list[dict]:
    rows = []
    for name in dataset_names():
        a = get_adapter(name)
        for c in a.cases():
            try:
                m = a.meta(c)
            except Exception:
                m = {}
            raw = m.get("group")
            f = m.get("field_T")
            rows.append({
                "dataset": name, "id": c.name,
                "vendor": _norm_vendor(m.get("vendor")),
                "pathology_raw": raw, "pathology": harmonize(raw),
                "field_T": f if not isinstance(f, list) else "/".join(map(str, f)),
                "centre": m.get("centre"), "age": m.get("age"), "sex": m.get("sex"),
                "height": m.get("height"), "weight": m.get("weight"),
            })
    return rows


def _norm_vendor(v):
    """Collapse vendor strings to a canonical short name (Siemens/Philips/GE/Canon)."""
    if not v:
        return None
    s = str(v).upper()
    for key, short in (("SIEMENS", "Siemens"), ("PHILIPS", "Philips"),
                       ("GE", "GE"), ("CANON", "Canon")):
        if key in s:
            return short
    return str(v)


def summarize(rows: list[dict]) -> None:
    by = defaultdict(list)
    for r in rows:
        by[r["dataset"]].append(r)
    print(f"=== data cloud: {len(rows)} subjects across {len(by)} datasets ===")
    for ds, rs in by.items():
        cov = {f: sum(1 for r in rs if r[f] not in (None, "")) for f in FIELDS}
        print(f"\n{ds}  (n={len(rs)})")
        print("  vendor:", dict(Counter(r["vendor"] for r in rs)))
        print("  pathology (harmonized):", dict(Counter(r["pathology"] for r in rs)))
        print("  coverage:", {f: f"{cov[f]}/{len(rs)}"
                              for f in ("field_T", "centre", "age", "sex", "height", "weight")})
    print("\n=== pooled harmonized pathology ===")
    print(" ", dict(Counter(r["pathology"] for r in rows)))
    print("=== pooled vendor ===")
    print(" ", dict(Counter(r["vendor"] for r in rows)))


def main() -> None:
    rows = build_rows()
    out = Path(data_root("processed")) / "inventory.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    summarize(rows)
    print(f"\ninventory -> {out}")


if __name__ == "__main__":
    main()
