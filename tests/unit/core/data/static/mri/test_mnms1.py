"""M&Ms-1 adapter pure logic (core.data.static.mri.mnms1) — the M&Ms-1 marker detection, SA path
resolution, CSV frame-index parsing, and centre-code -> (site, country) meta mapping. All I/O-free
except the marker/path tests which touch tmp_path dirs (cheap, no dataset)."""
import nibabel as nib
import numpy as np

from core.data.static.mri.base import MNM_LABEL_MAP, Base
from core.data.static.mri.mnms1 import CENTRES, Mnms1Adapter


# --- _frame_idx: blank/None -> None; float-string -> int; int-string ---
def test_frame_idx_classes():
    assert Mnms1Adapter._frame_idx("") is None                     # blank cell
    assert Mnms1Adapter._frame_idx(None) is None                   # missing
    assert Mnms1Adapter._frame_idx("0") == 0                        # int-string
    assert Mnms1Adapter._frame_idx("12.0") == 12                    # float-string (CSV formatting) -> int


# --- meta_from_row: known centre -> (site, country); unknown -> (code, None); vendor dual-spelling ---
def test_meta_from_row_known_centre():
    r = {"Centre": "1", "Vendor": "Siemens", "Pathology": "DCM", "Age": "55",
         "Sex": "M", "Height": "175", "Weight": "70"}
    m = Mnms1Adapter.meta_from_row(r)
    assert m["centre"] == CENTRES["1"][0] and m["country"] == "Spain"
    assert m["vendor"] == "Siemens" and m["group"] == "DCM"
    assert m["age"] == 55.0 and m["height"] == 175.0 and m["weight"] == 70.0
    assert m["field_T"] is None and m["scanner"] is None      # not in CSV


def test_meta_from_row_unknown_centre():
    """Centre code absent from the paper map -> raw code kept, country None (never fabricated)."""
    m = Mnms1Adapter.meta_from_row({"Centre": "99", "Vendor": "GE"})
    assert m["centre"] == "99" and m["country"] is None


def test_meta_from_row_vendor_dual_spelling():
    """VendorName preferred over Vendor when present (both spellings ship)."""
    assert Mnms1Adapter.meta_from_row({"VendorName": "Canon", "Vendor": "X"})["vendor"] == "Canon"
    assert Mnms1Adapter.meta_from_row({"Vendor": "Philips"})["vendor"] == "Philips"   # fallback to Vendor


def test_meta_from_row_bad_demographics_none():
    """Non-numeric age/height -> None via to_float, not a crash."""
    m = Mnms1Adapter.meta_from_row({"Centre": "1", "Age": "n/a", "Height": ""})
    assert m["age"] is None and m["height"] is None


# --- _is_mnms1: nested Training/Labeled marker / CSV column marker / negative ---
def test_is_mnms1_labeled_dir(tmp_path):
    (tmp_path / "Training" / "Labeled").mkdir(parents=True)
    assert Mnms1Adapter._is_mnms1(tmp_path) is True                         # nested Labeled dir marker


def test_is_mnms1_csv_column(tmp_path):
    (tmp_path / "info.csv").write_text("External code,Vendor\nA1,Canon\n")
    assert Mnms1Adapter._is_mnms1(tmp_path) is True                         # External-code CSV marker


def test_is_mnms1_negative(tmp_path):
    (tmp_path / "training").mkdir()                            # lowercase (ACDC), no Labeled/CSV
    assert Mnms1Adapter._is_mnms1(tmp_path) is False


# --- _sa: prefers .nii.gz, falls back to .nii, else the .nii.gz pair ---
def test_sa_prefers_nii_gz(tmp_path):
    case = tmp_path / "code1"; case.mkdir()
    (case / "code1_sa.nii.gz").write_text("x")
    img, gt = Mnms1Adapter._sa(case)
    assert img.name == "code1_sa.nii.gz" and gt.name == "code1_sa_gt.nii.gz"


def test_sa_falls_back_to_nii(tmp_path):
    case = tmp_path / "code2"; case.mkdir()
    (case / "code2_sa.nii").write_text("x")
    img, gt = Mnms1Adapter._sa(case)
    assert img.suffix == ".nii" and gt.name == "code2_sa_gt.nii"


def test_sa_default_pair_when_absent(tmp_path):
    """No file present -> default .nii.gz pair (caller's load_frames then skips the missing file)."""
    case = tmp_path / "code3"; case.mkdir()
    img, gt = Mnms1Adapter._sa(case)
    assert img.name == "code3_sa.nii.gz" and gt.name == "code3_sa_gt.nii.gz"


# --- load_ed_es resolve wiring: ES blank -> skipped, ED index used + label-remapped ---
def test_load_ed_es_skips_blank_es(tmp_path):
    case = tmp_path / "codeR"; case.mkdir()
    img4 = np.zeros((2, 2, 1, 3), np.float32)                 # x,y,z,t (3 frames)
    gt4 = np.zeros((2, 2, 1, 3), np.float32); gt4[..., 1] = 1  # frame 1 raw label 1 (LV-cav)
    for name, arr in (("codeR_sa.nii.gz", img4), ("codeR_sa_gt.nii.gz", gt4)):
        im = nib.Nifti1Image(arr, np.eye(4)); im.header.set_zooms((1, 1, 1, 1)); nib.save(im, case / name)
    row = {"ED": "1", "ES": "", "Pathology": "DCM"}

    def resolve(tag):
        idx = Mnms1Adapter._frame_idx(row.get(tag))
        return None if idx is None else (case / "codeR_sa.nii.gz", case / "codeR_sa_gt.nii.gz", idx)

    out = Base.load_frames(row["Pathology"], resolve, MNM_LABEL_MAP)
    assert "ED" in out and "ES" not in out                    # blank ES skipped
    assert 3 in np.unique(out["ED"]["gt"])                    # raw 1 -> canonical 3 (remapped)


# --- _is_mnms1: unreadable CSV path -> OSError swallowed, not a crash ---
def test_is_mnms1_unreadable_csv_swallowed(tmp_path, monkeypatch):
    """A CSV that raises on read (OSError) is skipped by the marker probe (no crash), returns False."""
    (tmp_path / "x.csv").write_text("External_code,Vendor\nA1,GE\n")  # no 'External code' marker string

    def _boom(self, *a, **k):
        raise OSError("unreadable")
    monkeypatch.setattr("pathlib.Path.read_text", _boom)
    assert Mnms1Adapter._is_mnms1(tmp_path) is False                        # OSError swallowed -> no marker -> False


# --- _root: resolves a valid MnM tree via CARDIAC_DATA; env override wins; fallback when absent ---
def _fake_mnm(base):
    """A minimal M&Ms-1 tree: MnM/ with the Training/Labeled marker + a subject dir with a SA cine."""
    root = base / "MnM"
    (root / "Training" / "Labeled").mkdir(parents=True)
    return root


def test_root_finds_mnm_under_raw_parent(tmp_path, monkeypatch):
    monkeypatch.setenv("CARDIAC_DATA", str(tmp_path / "data"))
    _fake_mnm(tmp_path / "data" / "raw")                       # data_root('raw').parent? -> raw.parent/MnM etc
    # _root scans raw.parent/MnM, raw/MnM ... the marker (Training/Labeled) validates it
    assert Mnms1Adapter._root().name == "MnM" and (Mnms1Adapter._root() / "Training" / "Labeled").is_dir()


def test_root_env_override(tmp_path, monkeypatch):
    root = _fake_mnm(tmp_path)
    monkeypatch.setenv("CARDIAC_MNMS1_ROOT", str(root))
    assert Mnms1Adapter._root() == root                                     # explicit env root wins


def test_root_fallback_when_no_tree(tmp_path, monkeypatch):
    """No valid MnM tree anywhere -> falls back to raw.parent/MnM (the default guess), never crashes."""
    monkeypatch.setenv("CARDIAC_DATA", str(tmp_path / "data"))
    assert Mnms1Adapter._root().name == "MnM"                               # default fallback path


# --- mnms1_info: reads the first CSV (External-code keyed); {} when no CSV present ---
def test_mnms1_info_reads_csv(tmp_path, monkeypatch):
    root = _fake_mnm(tmp_path)
    (root / "info.csv").write_text("External code,Vendor,Pathology\nA1,Canon,DCM\n")
    monkeypatch.setenv("CARDIAC_MNMS1_ROOT", str(root))
    info = Mnms1Adapter.mnms1_info()
    assert info["A1"]["Vendor"] == "Canon" and info["A1"]["Pathology"] == "DCM"


def test_mnms1_info_no_csv_empty(tmp_path, monkeypatch):
    root = _fake_mnm(tmp_path)
    monkeypatch.setenv("CARDIAC_MNMS1_ROOT", str(root))
    assert Mnms1Adapter.mnms1_info() == {}                                  # no CSV -> {}


# --- mnms1_cases: subject dirs with a SA-gt across Labeled/Validation/Testing, sorted ---
def test_mnms1_cases_lists_subjects_with_gt(tmp_path, monkeypatch):
    root = _fake_mnm(tmp_path)
    for split, code in (("Training/Labeled", "b1"), ("Validation", "v1"), ("Testing", "t1")):
        d = root / split / code; d.mkdir(parents=True)
        (d / f"{code}_sa.nii.gz").write_text("x")
        (d / f"{code}_sa_gt.nii.gz").write_text("x")           # gt present -> included
    (root / "Testing" / "nogt").mkdir()                        # no SA-gt -> excluded
    monkeypatch.setenv("CARDIAC_MNMS1_ROOT", str(root))
    names = [c.name for c in Mnms1Adapter.mnms1_cases()]
    assert names == ["b1", "v1", "t1"] and "nogt" not in names


# --- load_ed_es: end-to-end via CSV frame indices + 4D nifti, canonical remap, spacing carried ---
def test_load_ed_es_reads_frames(tmp_path, monkeypatch):
    root = _fake_mnm(tmp_path)
    case = root / "Training" / "Labeled" / "c1"; case.mkdir(parents=True)
    img4 = np.zeros((2, 2, 1, 3), np.float32)
    gt4 = np.zeros((2, 2, 1, 3), np.float32); gt4[..., 0] = 1; gt4[..., 2] = 3   # ED raw1, ES raw3
    for name, arr in (("c1_sa.nii.gz", img4), ("c1_sa_gt.nii.gz", gt4)):
        im = nib.Nifti1Image(arr, np.eye(4)); im.header.set_zooms((1, 1, 1, 1)); nib.save(im, case / name)
    (root / "info.csv").write_text("External code,ED,ES,Pathology\nc1,0,2,HCM\n")
    monkeypatch.setenv("CARDIAC_MNMS1_ROOT", str(root))
    out = Mnms1Adapter().load_ed_es(case)
    assert out["group"] == "HCM" and "ED" in out and "ES" in out
    assert 3 in np.unique(out["ED"]["gt"])                     # ED raw 1 -> canonical 3
    assert 1 in np.unique(out["ES"]["gt"])                     # ES raw 3 -> canonical 1


# --- Mnms1Adapter: thin delegators ---
def test_mnms1_adapter_delegates(tmp_path, monkeypatch):
    root = _fake_mnm(tmp_path)
    case = root / "Validation" / "s1"; case.mkdir(parents=True)
    (case / "s1_sa.nii.gz").write_text("x"); (case / "s1_sa_gt.nii.gz").write_text("x")
    (root / "info.csv").write_text("External code,Vendor,Centre\ns1,GE,3\n")
    monkeypatch.setenv("CARDIAC_MNMS1_ROOT", str(root))
    a = Mnms1Adapter()
    assert a.name == "mnms1" and a.label_map == MNM_LABEL_MAP
    assert [c.name for c in a.cases()] == ["s1"]               # delegates to mnms1_cases
    assert a.meta(case)["country"] == "Germany"               # delegates to meta_from_row (centre 3 -> Germany)
    led = a.load_ed_es(case)                                   # delegates; SA files are stubs (no ED/ES idx) -> empty
    assert "ED" not in led and "ES" not in led
