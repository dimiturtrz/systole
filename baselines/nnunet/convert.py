"""Convert ACDC / M&M-2 -> nnU-Net v2 raw format, so nnU-Net can train on it as a
SOTA baseline.

nnU-Net owns its own data layout (nnUNet_raw / dataset.json) and won't read our
pipeline directly — this is the one-way bridge. Each ED/ES frame becomes one
nnU-Net case. It reuses cardioseg's dataset-agnostic loaders (the same `load_ed_es`
the pipeline uses), so labels are already remapped to the ACDC convention
(0=bg, 1=RV, 2=myo, 3=LV — nnU-Net wants 0-based consecutive, which this is).

    python -m baselines.nnunet.convert --dataset acdc --out D:/data/nnUNet_raw --id 27

Then, in a SEPARATE nnU-Net env (kept out of cardioseg's deps):
    export nnUNet_raw=D:/data/nnUNet_raw nnUNet_preprocessed=... nnUNet_results=...
    nnUNetv2_plan_and_preprocess -d 27 --verify_dataset_integrity
    nnUNetv2_train 27 2d 0          # folds 0..4 for the 5-fold ensemble
    nnUNetv2_predict -i imagesTs -o <pred> -d 27 -c 2d
Then score the predictions with OUR eval layer:  python -m baselines.nnunet.score ...
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def _loaders():
    from cardioseg.data.mri.data import acdc_cases, load_ed_es
    from cardioseg.data.mri.mnm2 import mnm2_cases, load_ed_es as mnm2_load
    return {
        "acdc": (acdc_cases, load_ed_es, "ACDC"),
        "mnm2": (mnm2_cases, mnm2_load, "MNM2"),
    }


def _save_nifti(arr_zyx: np.ndarray, spacing_zyx, path: Path) -> None:
    """Write a [z,y,x] array as NIfTI (x,y,z order + diagonal affine from spacing)."""
    arr_xyz = np.transpose(arr_zyx, (2, 1, 0))
    aff = np.diag([spacing_zyx[2], spacing_zyx[1], spacing_zyx[0], 1.0])
    nib.save(nib.Nifti1Image(arr_xyz, aff), str(path))


def convert(dataset: str, out_root: str, dataset_id: int = 27, n_patients: int = 0) -> Path:
    cases_fn, loader, name = _loaders()[dataset]
    ds = Path(out_root) / f"Dataset{dataset_id:03d}_{name}"
    img_dir, lbl_dir = ds / "imagesTr", ds / "labelsTr"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    cases = cases_fn()
    if n_patients:
        cases = cases[:n_patients]
    n = 0
    for pd in cases:
        d = loader(pd)
        sp = tuple(float(s) for s in d["spacing"])
        for tag in ("ED", "ES"):
            if tag not in d:
                continue
            case = f"{pd.name}_{tag}"
            _save_nifti(d[tag]["img"], sp, img_dir / f"{case}_0000.nii.gz")  # _0000 = channel 0
            _save_nifti(d[tag]["gt"], sp, lbl_dir / f"{case}.nii.gz")
            n += 1

    # nnU-Net v2 dataset.json. "MRI" (non-"CT") channel -> per-image z-score normalization.
    (ds / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "MRI"},
        "labels": {"background": 0, "RV": 1, "MYO": 2, "LV": 3},
        "numTraining": n,
        "file_ending": ".nii.gz",
    }, indent=2))
    print(f"wrote {n} cases ({len(cases)} patients x ED/ES) -> {ds}")
    return ds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["acdc", "mnm2"], default="acdc")
    ap.add_argument("--out", default="D:/data/nnUNet_raw")
    ap.add_argument("--id", type=int, default=27, help="nnU-Net dataset id")
    ap.add_argument("--n-patients", type=int, default=0, help="0 = all")
    a = ap.parse_args()
    convert(a.dataset, a.out, a.id, a.n_patients)


if __name__ == "__main__":
    main()
