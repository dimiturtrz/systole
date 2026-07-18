"""Precompute the web viewer's assets: per-patient chamber meshes (.glb) + an EF manifest.

The browser app (cardioview/web) is pure rendering — all inference/geometry happens here,
offline, and ships as glb + JSON. ED and ES share one crop/grid so the chambers align when
you toggle phases. Volumes (EDV/ESV/EF) come from the full-res predicted masks × the
patient's real spacing — the viz crop/resample never touches the numbers.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from common import (
    DEFAULT_MODEL,
    MODELS,
    SIZE,
    load_model,
    log_setup,
    model_dir,
    patient_dir,
    square_stack,
)
from common import (
    masks as build_masks,
)
from geometry import bbox_slices, nearest_index
from PIL import Image

from core.config import Config
from core.data.static import splits, store
from core.data.static.mri.acdc import AcdcAdapter
from core.data.static.splits import Splits
from core.hparams import Hparams
from core.inference import Inference
from core.measure import Measure
from core.mesh import Mesh  # reusable chamber-mesh tool (bd 7c9.1)
from core.postprocess import Postprocess
from core.preprocessing.preprocess import Preprocess

log = logging.getLogger("cardioview.export_web")

MARGIN_MM = 12.0            # heart-bbox crop margin (shared-crop, keeps chambers aligned)
INPLANE_MM = 1.5           # in-plane resample step the 2D model runs at
CINE_BBOX_MARGIN = 14      # in-plane crop margin (voxels) around the whole-cine heart union

# Web assets live OUT of the repo, under the data root (<data>/meshes/cardioview/) — never committed.
# This is the single external home: glb/manifest/slices here + the exact model in models/<name>.onnx.
# The viewer serves it via web/scripts/sync-assets.mjs (copies here -> gitignored public/{data,models}
# on predev/prebuild), not from a committed public dir. (bd cardiac-seg-ra3)
OUT = Path(Config.data_root("meshes")) / "cardioview"


def publish_model(model_name: str) -> None:  # pragma: no cover  (file-copy shell)
    """Copy the exact ONNX that produced these assets into the external home (models/<name>.onnx),
    so the web bundle's model travels with its manifest. The registry artifact dir holds model.onnx
    (built by core.export_onnx at train time)."""
    src = Path(model_dir(MODELS[model_name])) / "model.onnx"
    if not src.exists():
        log.warning("no model.onnx in %s — run `python -m core.export_onnx` to bundle the web model", src.parent)
        return
    dst = OUT / "models" / f"{model_name}.onnx"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    log.info("  model  -> %s", dst)


def heldout_set(model_name: str) -> set[str]:
    """Subject IDs the model did NOT train on — derived from the run's saved config
    (the flagship holds out ALL ACDC, not an 80/20 val slice). Falls back to the legacy
    80/20 ACDC val split for older runs without a config.json."""
    run = model_dir(MODELS[model_name])
    cfg_path = run / "config.json"
    if cfg_path.exists():  # pragma: no cover  (registry/data dependency)
        dc = Hparams.from_json(cfg_path).generator.data
        meta = store.load(list(dc.sources))
        _, val, test = splits.Splits.make_split(meta, dc.test_datasets, dc.test_vendors, dc.val_frac,
                                         val_datasets=dc.val_datasets, val_vendors=dc.val_vendors)
        # "held out" = anything the model did NOT train on (val OR test) — ACDC is now val, still unseen.
        return set(val.get_column("subject_id").to_list()) | set(test.get_column("subject_id").to_list())
    _, val = Splits.split_patients(list(AcdcAdapter().cases()), 0.2, 0)
    return {c.name for c in val}


@dataclass
class ExportCtx:
    """Plumbing shared across an export run: the loaded model, its device, and registry name."""
    model: object
    device: str
    model_name: str


def shared_crop(masks: dict[str, Any], spacing: Sequence[float], margin_mm: float = MARGIN_MM) -> tuple[dict[str, Any], float]:
    """Crop every phase to the union heart bbox + margin (kept anisotropic — the mesher
    resamples per-chamber with linear interp for smooth surfaces). Returns crops + iso step."""
    union = np.zeros_like(next(iter(masks.values())), dtype=bool)
    for m in masks.values():
        union |= m > 0
    crop = bbox_slices(union, spacing, margin_mm)
    return {t: m[crop] for t, m in masks.items()}, float(min(spacing))




def volumes(masks: dict[str, Any], spacing: Sequence[float]) -> dict[str, Any]:
    """EDV (ml full), ESV (ml empty), EF (%) from the LV cavity — full-res, real spacing."""
    if "ED" in masks and "ES" in masks:
        ef, edv, esv = Measure.ejection_fraction(masks["ED"], masks["ES"], spacing, lv_label=3)
        return {"ef": round(ef, 1), "edv": round(edv, 1), "esv": round(esv, 1)}
    return {}


@dataclass(kw_only=True)
class ManifestEntry:
    """One patient row in the web manifest — the single schema both exports build, so the on-disk
    shape can't drift between the static and beating paths. `to_dict` IS the JSON form; the
    animation-only fields are dropped when unset, so a static entry serializes to exactly the keys
    the viewer expects (no null frames/slices)."""
    patient: str
    group: str | None
    held_out: bool
    source: str
    pred: dict[str, Any]
    gt: dict[str, Any]
    glb: dict[str, Any]
    frames: list[str] | None = None      # beating-cycle only (None on a static entry)
    ed_idx: int | None = None
    es_idx: int | None = None
    slices: list[str] | None = None
    slice_d: int | None = None           # serialized as the viewer's `sliceD` key

    def to_dict(self) -> dict[str, Any]:
        entry: dict[str, Any] = {"patient": self.patient, "group": self.group, "held_out": self.held_out,
                 "source": self.source, "pred": self.pred, "gt": self.gt, "glb": self.glb}
        if self.frames is not None:      # a beating entry carries the cine strips + phase indices
            entry.update(frames=self.frames, ed_idx=self.ed_idx, es_idx=self.es_idx,
                         slices=self.slices, sliceD=self.slice_d)
        return entry


def manifest_with(data: dict[str, Any] | None, entry: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Pure insert/replace of one patient into a manifest dict -> the new {model, hearts} dict.
    `data` is the prior manifest (dict, the old bare-array form, or None); the entry replaces any
    same-patient row and the hearts list stays sorted by patient. (write is the shell around this.)"""
    data = data or {}
    if isinstance(data, list):  # back-compat with the old array form
        data = {"hearts": data}
    hearts = [e for e in data.get("hearts", []) if e["patient"] != entry["patient"]]
    hearts.append(entry)
    hearts.sort(key=lambda e: e["patient"])
    return {"model": model_name, "hearts": hearts}


def upsert_manifest(entry: ManifestEntry, model_name: str) -> None:  # pragma: no cover  (file-IO shell)
    """Insert/replace one patient in manifest.json. Manifest = {model, hearts}; the web
    reads `model` to know which bundled .onnx to load (so it follows what was exported)."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "manifest.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps(manifest_with(data, entry.to_dict(), model_name), indent=2))


def load_4d(pdir: Path, name: str) -> tuple[Any, tuple[Any, ...]]:  # pragma: no cover  (nibabel NIfTI disk load — IO shell)
    """Load the cine as [t, z, y, x] + spacing (z, y, x) mm."""
    img = nib.load(str(pdir / f"{name}_4d.nii.gz"))
    arr = np.transpose(np.asanyarray(img.dataobj), (3, 2, 1, 0))  # x,y,z,t -> t,z,y,x
    xs, ys, zs = img.header.get_zooms()[:3]
    return arr, (zs, ys, xs)


def frame_indices(pdir: Path) -> tuple[int, int]:  # pragma: no cover  (Info.cfg disk parse — IO shell)
    """0-based ED, ES frame indices (Info.cfg parsing reused from core)."""
    cfg = AcdcAdapter.parse_info_cfg(pdir)
    return int(cfg["ED"]) - 1, int(cfg["ES"]) - 1


def run(patients: Sequence[str], source: str, ctx: ExportCtx) -> None:  # pragma: no cover  (export orchestration shell)
    held = heldout_set(ctx.model_name)
    for p in patients:
        pdir = patient_dir(p)  # p may be an ID or a full path
        name = pdir.name
        case = Preprocess.preprocess_case(pdir, loader=AcdcAdapter().load_ed_es)
        spacing = tuple(float(s) for s in case["spacing"])
        masks = build_masks(case, source, ctx.model, ctx.device)
        crop_masks, iso = shared_crop(masks, spacing)
        glb = {}
        for tag, m in crop_masks.items():
            fn = f"{name}_{tag}_{source}.gltf"
            Mesh.export_glb(m, spacing, OUT / fn)
            glb[tag] = fn
        entry = ManifestEntry(patient=name, group=case.get("group"), held_out=(name in held), source=source,
                              pred=volumes(masks, spacing), gt=volumes(build_masks(case, "gt"), spacing), glb=glb)
        upsert_manifest(entry, ctx.model_name)
        log.info("  %-11s %-5s static  EF %s%% (GT %s%%)", name, str(case.get("group")),
                 entry.pred.get("ef"), entry.gt.get("ef"))


def run_animate(patients: Sequence[str], ctx: ExportCtx, stride: int = 1) -> None:  # pragma: no cover  (export orchestration shell)
    """Segment every cine frame -> per-frame chamber glb -> a beating-cycle entry, per patient."""
    held = heldout_set(ctx.model_name)
    for p in patients:
        _animate_patient(p, ctx, held, stride)


def _segment_cine(pdir: Path, name: str, ctx: ExportCtx, stride: int) -> tuple[list[int], dict[int, Any], dict[int, Any], tuple[Any, ...]]:  # pragma: no cover  (inference shell)
    """Segment every strided cine frame -> (frames_t, masks{k}, grays{k} aligned, rspacing)."""
    vol, spacing = load_4d(pdir, name)
    rspacing = (spacing[0], INPLANE_MM, INPLANE_MM)
    frames_t = list(range(0, vol.shape[0], stride))
    masks: dict[int, Any] = {}
    grays: dict[int, Any] = {}
    for k, t in enumerate(frames_t):
        img = Preprocess.zscore(Preprocess.resample_inplane(vol[t].astype(np.float32), spacing, INPLANE_MM)[0])
        masks[k] = Postprocess.largest_cc_per_class(
            Inference(ctx.model, SIZE, ctx.device).predict_volume(img, tta=True))
        grays[k] = square_stack(img)
    return frames_t, masks, grays, rspacing


def _heart_bbox(masks: dict[int, Any], margin: int = CINE_BBOX_MARGIN) -> tuple[int, int, int, int]:
    """In-plane (r0,r1,c0,c1) of the heart union over the whole cine (+margin) — one stable crop."""
    union = np.zeros((SIZE, SIZE), bool)
    for k in masks:
        union |= (masks[k] > 0).any(axis=0)
    ys, xs = np.where(union)
    if not ys.size:
        return 0, SIZE, 0, SIZE
    return (max(0, int(ys.min()) - margin), min(SIZE, int(ys.max()) + margin + 1),
            max(0, int(xs.min()) - margin), min(SIZE, int(xs.max()) + margin + 1))


def _animate_patient(p: str, ctx: ExportCtx, held: set[str], stride: int) -> None:  # pragma: no cover  (export orchestration shell)
    """One patient: segment the cine, write per-frame slice strips + chamber glbs, upsert the entry."""
    pdir = patient_dir(p)
    name = pdir.name
    frames_t, masks, grays, rspacing = _segment_cine(pdir, name, ctx, stride)
    r0, r1, c0, c1 = _heart_bbox(masks)
    slice_files = []
    for k in range(len(frames_t)):
        sfn = f"{name}_f{k:02d}_slices.png"
        save_strip(grays[k][:, r0:r1, c0:c1], masks[k][:, r0:r1, c0:c1], OUT / sfn)
        slice_files.append(sfn)
    crop_masks, iso = shared_crop(masks, rspacing)
    files = []
    for k in range(len(frames_t)):
        fn = f"{name}_f{k:02d}_pred.gltf"
        Mesh.export_glb(crop_masks[k], rspacing, OUT / fn)
        files.append(fn)
    edi, esi = frame_indices(pdir)
    ed_k = nearest_index(frames_t, edi)
    es_k = nearest_index(frames_t, esi)
    ef, edv, esv = Measure.ejection_fraction(masks[ed_k], masks[es_k], rspacing, lv_label=3)
    case = Preprocess.preprocess_case(pdir, loader=AcdcAdapter().load_ed_es)
    gt = volumes(build_masks(case, "gt"), tuple(float(s) for s in case["spacing"]))
    entry = ManifestEntry(patient=name, group=case.get("group"), held_out=(name in held), source="pred",
                          pred={"ef": round(ef, 1), "edv": round(edv, 1), "esv": round(esv, 1)}, gt=gt,
                          glb={"ED": files[ed_k], "ES": files[es_k]},
                          frames=files, ed_idx=ed_k, es_idx=es_k,
                          slices=slice_files, slice_d=int(masks[0].shape[0]))
    upsert_manifest(entry, ctx.model_name)
    log.info("  %-11s %-5s BEATING %d frames  EF %s%% (GT %s%%)", name, str(case.get("group")),
             len(files), entry.pred["ef"], gt.get("ef"))




def save_strip(gray: np.ndarray, mask: np.ndarray, path: Path) -> None:  # pragma: no cover  (file-write shell)
    """One cine frame's (already heart-cropped) slices -> a vertical RGBA PNG strip [D*H, W]:
    R = grayscale (percentile-windowed for real MRI contrast), G = label (0..3). The web decodes it
    directly (W from image width, H from height/D)."""
    nz, h, w = gray.shape
    lo, hi = np.percentile(gray, [1, 99])  # window: z-scored MRI has outliers; min/max washes out
    g8 = np.clip((gray - lo) / ((hi - lo) or 1) * 255, 0, 255).astype(np.uint8)
    rgba = np.zeros((nz * h, w, 4), np.uint8)
    for z in range(nz):
        rgba[z * h:(z + 1) * h, :, 0] = g8[z]
        rgba[z * h:(z + 1) * h, :, 1] = mask[z]
        rgba[z * h:(z + 1) * h, :, 3] = 255
    Image.fromarray(rgba, "RGBA").save(path)


def main():  # pragma: no cover  (argparse CLI entry + load_model GPU + export orchestration — shell)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["static", "animate"], default="static")
    # Default hearts come from paths.yaml (cardioview.hearts); fallback = one per condition.
    ap.add_argument("--patients", nargs="*", default=None)
    ap.add_argument("--source", default="pred", choices=["pred", "gt"])
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--stride", type=int, default=1, help="cine frame stride (animate)")
    a = ap.parse_args()
    log_setup()
    OUT.mkdir(parents=True, exist_ok=True)   # first writer (slice PNGs) runs before upsert_manifest
    # canned demo hearts (one per pathology); override with --patients (IDs or full paths).
    patients = a.patients or ["patient073", "patient006", "patient021", "patient053"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(MODELS[a.model], device) if a.source == "pred" or a.mode == "animate" else None
    ctx = ExportCtx(model=model, device=device, model_name=a.model)
    if a.mode == "animate":
        run_animate(patients, ctx, a.stride)
    else:
        run(patients, a.source, ctx)
    if model is not None:
        publish_model(a.model)          # bundle the exact ONNX next to its manifest (bd ra3)
    log.info("manifest: %s", OUT / "manifest.json")


if __name__ == "__main__":
    main()
