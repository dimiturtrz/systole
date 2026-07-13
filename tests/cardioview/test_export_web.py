"""Pure web-asset logic for export_web: the shared union crop, chamber volumes (core.measure),
the manifest dict transform, the held-out set, and the cine heart-bbox. glb/png/onnx writes,
nibabel load, model inference and the run/animate orchestration are shells (pragma'd)."""
from pathlib import Path

import export_onnx
import export_web as E
import numpy as np
from export_web import (
    SIZE,
    ManifestEntry,
    _heart_bbox,
    heldout_set,
    manifest_with,
    shared_crop,
    volumes,
)

# --- shared_crop -----------------------------------------------------------

def test_shared_crop_uses_the_phase_union():
    """Heart-present class: the crop covers BOTH phases' heart (union bbox), so toggling stays aligned.
    ED blob top-left, ES blob bottom-right -> the window spans both; iso = finest spacing."""
    ed = np.zeros((2, 10, 10), np.uint8); ed[:, 1:3, 1:3] = 1
    es = np.zeros((2, 10, 10), np.uint8); es[:, 6:8, 6:8] = 1
    crops, iso = shared_crop({"ED": ed, "ES": es}, (6.0, 1.5, 1.5), margin_mm=0.0)
    assert iso == 1.5
    # both crops share one window -> same shape, and each still holds its own blob
    assert crops["ED"].shape == crops["ES"].shape
    assert crops["ED"].sum() == ed.sum() and crops["ES"].sum() == es.sum()


def test_shared_crop_no_heart_returns_full():
    """Heart-absent class: an all-zero union -> bbox is the full array (empty-mask fallback)."""
    z = np.zeros((2, 6, 6), np.uint8)
    crops, iso = shared_crop({"ED": z, "ES": z.copy()}, (1.0, 1.0, 1.0))
    assert crops["ED"].shape == (2, 6, 6)


# --- volumes (delegates to core.measure.ejection_fraction) -----------------

def test_volumes_per_chamber_ml():
    """Full-pair class: EDV, ESV, EF from LV-cav (label 3) at real spacing. 8 vs 4 cav voxels at
    1 mm^3 -> EDV 8, ESV 4 voxels; voxel=1mm^3=0.001mL -> EDV .008, ESV .004, EF 50%."""
    ed = np.zeros((1, 4, 4), np.uint8); ed[0, :2, :4] = 3   # 8 voxels
    es = np.zeros((1, 4, 4), np.uint8); es[0, 0, :4] = 3    # 4 voxels
    v = volumes({"ED": ed, "ES": es}, (1.0, 1.0, 1.0))
    assert v["ef"] == 50.0                            # EF ratio is spacing-free: (8-4)/8
    assert v["edv"] == 0.0 and v["esv"] == 0.0        # 8/4 mm^3 -> mL rounds to 0.0 at 1dp


def test_volumes_scales_with_spacing():
    """Spacing class: same voxel counts at coarse spacing give larger mL (Riemann sum × voxel vol)."""
    ed = np.zeros((1, 4, 4), np.uint8); ed[0, :2, :4] = 3
    es = np.zeros((1, 4, 4), np.uint8); es[0, 0, :4] = 3
    v = volumes({"ED": ed, "ES": es}, (10.0, 10.0, 10.0))  # 1000 mm^3/voxel = 1 mL
    assert v["edv"] == 8.0 and v["esv"] == 4.0 and v["ef"] == 50.0


def test_volumes_missing_phase_is_empty():
    """Single-phase class: only ED -> no EF pair -> empty dict (nothing measurable)."""
    assert volumes({"ED": np.zeros((1, 4, 4), np.uint8)}, (1.0, 1.0, 1.0)) == {}


# --- manifest_with (pure dict transform behind upsert_manifest) ------------

def test_manifest_with_inserts_into_empty():
    """Empty class: no prior manifest -> a {model, hearts:[entry]} dict."""
    out = manifest_with(None, {"patient": "p1"}, "gen")
    assert out == {"model": "gen", "hearts": [{"patient": "p1"}]}


def test_manifest_with_replaces_same_patient_and_sorts():
    """Replace class: re-inserting a patient overwrites its row (no dup); hearts stay patient-sorted."""
    prior = {"model": "old", "hearts": [{"patient": "p2", "v": 1}, {"patient": "p1"}]}
    out = manifest_with(prior, {"patient": "p2", "v": 2}, "gen")
    pats = [h["patient"] for h in out["hearts"]]
    assert pats == ["p1", "p2"]                       # sorted
    assert next(h for h in out["hearts"] if h["patient"] == "p2")["v"] == 2  # replaced


def test_manifest_with_upgrades_legacy_array():
    """Back-compat class: the old bare-array manifest form is lifted into {hearts:[...]}."""
    out = manifest_with([{"patient": "p1"}], {"patient": "p2"}, "gen")
    assert [h["patient"] for h in out["hearts"]] == ["p1", "p2"]


# --- ManifestEntry.to_dict (the single manifest schema for both export paths) ----

def test_manifest_entry_static_omits_animation_keys():
    """Static class: a static entry carries no cine fields -> to_dict emits exactly the 7 static keys
    (no null frames/slices/sliceD leaking into the JSON the viewer reads)."""
    entry = ManifestEntry(patient="p1", group="DCM", held_out=True, source="pred",
                          pred={"ef": 42.0}, gt={"ef": 40.0}, glb={"ED": "p1_ED_pred.gltf"})
    assert entry.to_dict() == {"patient": "p1", "group": "DCM", "held_out": True, "source": "pred",
                               "pred": {"ef": 42.0}, "gt": {"ef": 40.0}, "glb": {"ED": "p1_ED_pred.gltf"}}


def test_manifest_entry_beating_includes_cine_fields_and_sliced_key():
    """Beating class: an animation entry adds the cine strips + phase indices, and the snake `slice_d`
    field serializes to the viewer's `sliceD` JSON key."""
    entry = ManifestEntry(patient="p2", group="NOR", held_out=False, source="pred",
                          pred={"ef": 55.0}, gt={"ef": 54.0}, glb={"ED": "a.gltf", "ES": "b.gltf"},
                          frames=["f0.gltf", "f1.gltf"], ed_idx=0, es_idx=1,
                          slices=["s0.png"], slice_d=9)
    d = entry.to_dict()
    assert d["frames"] == ["f0.gltf", "f1.gltf"] and d["ed_idx"] == 0 and d["es_idx"] == 1
    assert d["slices"] == ["s0.png"] and d["sliceD"] == 9 and "slice_d" not in d


# --- heldout_set (fallback branch: no config.json -> split_patients) -------

def test_heldout_set_fallback_split(monkeypatch, tmp_path):
    """Legacy-run class: a run dir with no config.json -> the 80/20 ACDC val split (deterministic set)."""
    monkeypatch.setattr(E, "model_dir", lambda ref: tmp_path)            # empty dir, no config.json
    monkeypatch.setattr(E.AcdcAdapter, "cases", lambda self: [Path(f"p{i}") for i in range(10)])
    held = heldout_set("gen")
    assert isinstance(held, set) and len(held) == 2   # 0.2 of 10
    assert held == heldout_set("gen")                 # deterministic


# --- _heart_bbox (whole-cine union in-plane window) ------------------------

def test_heart_bbox_spans_all_frames_with_margin():
    """Multi-frame class: the bbox covers the union of every frame's heart + margin, clamped to SIZE."""
    masks = {0: np.zeros((2, SIZE, SIZE), np.uint8), 1: np.zeros((2, SIZE, SIZE), np.uint8)}
    masks[0][:, 20, 20] = 1
    masks[1][:, 40, 60] = 1
    r0, r1, c0, c1 = _heart_bbox(masks, margin=2)
    assert r0 == 18 and r1 == 43 and c0 == 18 and c1 == 63   # rows 20..40, cols 20..60, ±2 margin


def test_heart_bbox_empty_returns_full_grid():
    """No-heart class: an all-zero cine -> the full SIZE×SIZE window (safe default crop)."""
    masks = {0: np.zeros((2, SIZE, SIZE), np.uint8)}
    assert _heart_bbox(masks) == (0, SIZE, 0, SIZE)


# --- export_onnx: import smoke (whole file is an onnx-build + shutil.copy shell) --------

def test_export_onnx_imports_and_exposes_out():
    """The export_onnx module is a CLI shell (main pragma'd), but its import path runs in production —
    smoke it so a broken import (e.g. a moved core symbol) is caught, not silently shipped."""
    assert export_onnx.OUT.name == "models" and callable(export_onnx.main)
