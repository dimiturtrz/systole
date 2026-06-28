"""Convert the cardioseg data store -> nnU-Net v2 raw format, so nnU-Net trains on the SAME split
as our flagship (the battery) for an apples-to-apples SOTA baseline.

nnU-Net owns its own data layout (nnUNet_raw / dataset.json) and does its OWN preprocessing
(resample + normalize), so we feed it RAW volumes (via the dataset adapters' load_ed_es — labels
already remapped to canonical 0=bg/1=RV/2=myo/3=LV, which nnU-Net wants 0-based consecutive). Each
ED/ES frame = one nnU-Net case.

Battery parity: imagesTr/labelsTr = the battery TRAIN+VAL pool (M&M-2 + M&Ms-1 ex-Canon, labelled);
imagesTs/labelsTs = the held-out battery TEST (ACDC-150 + Canon-9). A ts_manifest.json records each
test case's axis (acdc vs canon) so scoring can report the two generalization axes separately.

    python -m baselines.nnunet.convert --id 29     # --out defaults to <data>/nnunet/raw (config-derived)
    # then (same env; nnUNet_raw/_preprocessed/_results set by `source baselines/nnunet/env.sh`):
    nnUNetv2_plan_and_preprocess -d 29 --verify_dataset_integrity
    nnUNetv2_train 29 2d 0 -tr nnUNetTrainer_50epochs
    nnUNetv2_predict -i .../imagesTs -o <pred> -d 29 -c 2d -f 0 -tr nnUNetTrainer_50epochs
    python -m baselines.nnunet.score ...
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def _save_nifti(arr_zyx: np.ndarray, spacing_zyx, path: Path) -> None:
    """Write a [z,y,x] array as NIfTI (x,y,z order + diagonal affine from spacing)."""
    arr_xyz = np.transpose(arr_zyx, (2, 1, 0))
    aff = np.diag([spacing_zyx[2], spacing_zyx[1], spacing_zyx[0], 1.0])
    nib.save(nib.Nifti1Image(arr_xyz, aff), str(path))


def _write_cases(rows, img_dir: Path, lbl_dir: Path, get_adapter) -> int:
    """Load each subject's raw ED/ES via its adapter, write nnU-Net case files. Returns case count."""
    from pathlib import Path as P
    n = 0
    for r in rows:
        a = get_adapter(r["dataset"])
        d = a.load_ed_es(P(r["raw_path"]))
        sp = tuple(float(s) for s in d["spacing"])
        for tag in ("ED", "ES"):
            if tag not in d:
                continue
            case = f"{r['dataset']}_{r['subject_id']}_{tag}"
            _save_nifti(d[tag]["img"], sp, img_dir / f"{case}_0000.nii.gz")  # _0000 = channel 0
            _save_nifti(d[tag]["gt"], sp, lbl_dir / f"{case}.nii.gz")
            n += 1
    return n


def convert_battery(out_root: str, dataset_id: int = 29, n_patients: int = 0) -> Path:
    """Export the battery split to nnU-Net raw: Tr = train+val pool, Ts = acdc+canon held-out."""
    import polars as pl
    from cardioseg.data import store, splits
    from cardioseg.data.mri.registry import get_adapter
    from core.hparams import DataCfg

    dc = DataCfg()                                           # the generalization criteria (Canon+GE test, ACDC val)
    meta = store.load(list(dc.sources))
    train_df, val_df, test_df = splits.make_split(meta, dc.test_datasets, dc.test_vendors, dc.val_frac,
                                                  val_datasets=dc.val_datasets, val_vendors=dc.val_vendors)
    tr = train_df                                            # nnU-Net trains on our TRAIN (Si+Ph); does its
    #                                                          own internal CV. ACDC=val is NOT given to it
    #                                                          (apples-to-apples: our model doesn't train on it).
    if n_patients:
        tr, test_df = tr.head(n_patients), test_df.head(n_patients)

    ds = Path(out_root) / f"Dataset{dataset_id:03d}_BATTERY"
    import shutil
    if ds.exists():
        shutil.rmtree(ds)              # wipe stale cases — a prior split's files must NOT linger
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (ds / sub).mkdir(parents=True, exist_ok=True)

    n_tr = _write_cases(tr.iter_rows(named=True), ds / "imagesTr", ds / "labelsTr", get_adapter)
    n_ts = _write_cases(test_df.iter_rows(named=True), ds / "imagesTs", ds / "labelsTs", get_adapter)

    # axis manifest: which held-out test cases are the centre-shift (acdc) vs unseen-vendor (canon)
    manifest = {}
    for r in test_df.iter_rows(named=True):
        axis = (r.get("vendor") or r["dataset"]).lower()   # per-vendor axis (canon / ge) — the held-out unit
        for tag in ("ED", "ES"):
            manifest[f"{r['dataset']}_{r['subject_id']}_{tag}"] = {"axis": axis, "vendor": r.get("vendor")}
    (ds / "ts_manifest.json").write_text(json.dumps(manifest, indent=2))

    (ds / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "MRI"},                       # MRI -> nnU-Net per-image z-score norm
        "labels": {"background": 0, "RV": 1, "MYO": 2, "LV": 3},
        "numTraining": n_tr,
        "file_ending": ".nii.gz",
    }, indent=2))
    print(f"battery -> {ds}\n  train+val: {len(tr)} subjects -> {n_tr} cases\n  test (acdc+canon): "
          f"{len(test_df)} subjects -> {n_ts} cases")
    return ds


def main():
    from core.config import data_root

    ap = argparse.ArgumentParser(description=__doc__)
    # Default to the data-namespaced raw dir (<data>/nnunet/raw) derived from cardioseg's path
    # config — never the D:/data root, and machine-independent (no hardcoded absolute path).
    ap.add_argument("--out", default=None, help="nnU-Net raw root (default: <data>/nnunet/raw)")
    ap.add_argument("--id", type=int, default=29, help="nnU-Net dataset id")
    ap.add_argument("--n-patients", type=int, default=0, help="0 = all (debug cap)")
    a = ap.parse_args()
    out = a.out or str(Path(data_root("nnunet")) / "raw")
    convert_battery(out, a.id, a.n_patients)


if __name__ == "__main__":
    main()
