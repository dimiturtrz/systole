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
          "field_T", "centre", "age", "age_band", "sex", "height", "weight", "bsa"]


def _bsa(height, weight):
    """Body surface area (m^2, Mosteller) from height(cm)+weight(kg) — the clinical volume-indexing
    variable + a fairness axis. None if either missing."""
    try:
        h, w = float(height), float(weight)
        return round((h * w / 3600.0) ** 0.5, 2) if h > 0 and w > 0 else None
    except (TypeError, ValueError):
        return None


def _age_band(age):
    """Coarse age band for fairness stratification. None -> None (unknown)."""
    try:
        a = float(age)
    except (TypeError, ValueError):
        return None
    return "<45" if a < 45 else "45-60" if a < 60 else "60-75" if a < 75 else "75+"


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
                "centre": m.get("centre"), "age": m.get("age"), "age_band": _age_band(m.get("age")),
                "sex": m.get("sex"), "height": m.get("height"), "weight": m.get("weight"),
                "bsa": _bsa(m.get("height"), m.get("weight")),
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
        print("  sex:", dict(Counter(r["sex"] for r in rs if r["sex"])),
              "| age_band:", dict(Counter(r["age_band"] for r in rs if r["age_band"])),
              f"| bsa: {cov['bsa']}/{len(rs)}")
        print("  coverage:", {f: f"{cov[f]}/{len(rs)}"
                              for f in ("field_T", "centre", "age", "sex", "height", "weight", "bsa")})
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
