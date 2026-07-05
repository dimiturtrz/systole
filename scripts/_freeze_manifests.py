"""Freeze the standard cross-domain TEST manifests (one-shot; immutable once written).

seg4 holdouts over the default 4-class cloud (acdc/mnm2/mnms1/cmrxmotion):
  - vendor: siemens / philips / ge / canon   (does the number survive a scanner change)
  - dataset: acdc / mnm2 / mnms1 / cmrxmotion (leave-one-dataset-out; cmrxmotion == continent Asia)
seg_lv external lane:
  - scd: Sunnybrook GE/Canada, LV-only {0,2,3} — a held-out CENTRE we never train on.
Kaggle EF is NOT here — it has no masks / no seg npz, so it can't resolve to validate(); it's a
separate EF-at-scale lane (load_sax -> seg->EF pipeline), tracked apart.
"""
from core.data.static import store, manifest

DATE = "2026-07-05"

meta = store.load()   # default 4-class cloud

VENDORS = {
    "siemens": ("Siemens", "biggest domain (acdc+mnm2+mnms1+cmrx, n=522); holdout leaves a small non-Siemens train"),
    "philips": ("Philips", "n=192 (mnm2+mnms1)"),
    "ge":      ("GE",      "scarce-vendor stress, n=69 (mnm2+mnms1)"),
    "canon":   ("Canon",   "UNSEEN-vendor stress, n=9 (mnms1 only) — small, report with care"),
}
for key, (vendor, note) in VENDORS.items():
    p = manifest.freeze(f"vendor_{key}", meta, test_vendors=[vendor], task="seg4",
                        note=note, created=DATE)
    print("froze", p.name)

for ds, note in [("acdc", "single-centre Siemens 1.5/3T"), ("mnm2", "multi-vendor multi-centre"),
                 ("mnms1", "multi-vendor challenge (incl. Canon)"),
                 ("cmrxmotion", "Asia/Siemens motion cohort (== continent_asia)")]:
    p = manifest.freeze(f"dataset_{ds}", meta, test_datasets=[ds], task="seg4", note=note, created=DATE)
    print("froze", p.name)

# external LV-only lane — pull SCD into a store load (not in SOURCE_DATASETS)
scd_meta = store.load(["scd"])
p = manifest.freeze("scd_lv", scd_meta, test_datasets=["scd"], task="seg_lv",
                    note="Sunnybrook GE/Canada, LV-only {0,2,3}, held-out centre; score classes 2,3 only",
                    created=DATE)
print("froze", p.name)

print("\nmanifests:", manifest.list_manifests())
