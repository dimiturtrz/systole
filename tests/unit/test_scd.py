"""SCD adapter pure logic (core.data.static.mri.scd) — the contour-dir renaming, polygon rasterization
to canonical {0,2,3}, and ED/ES-by-endo-area selection. All I/O-free (DICOM reads + patient CSV globbing
are the integration shell, skipped without the sunnybrook data)."""
import numpy as np

from core.data.static.mri.scd import (
    _IRCCI,
    ScdAdapter,
    _contour_dir_name,
    _fill,
    _rasterize,
    _read_contour,
    _root,
    _sax_series_dir,
    load_ed_es,
    scd_cases,
    scd_meta,
    select_ed_es,
)


# --- _contour_dir_name: trailing number zero-padded to 2; interior digits untouched ---
def test_contour_dir_name_pads_trailing():
    assert _contour_dir_name("SC-HF-I-1") == "SC-HF-I-01"     # single digit -> 2
    assert _contour_dir_name("SC-HF-I-12") == "SC-HF-I-12"    # already 2 -> unchanged
    assert _contour_dir_name("SC-HF-I-5") == "SC-HF-I-05"


def test_contour_dir_name_only_trailing_run():
    """Interior digit is left alone; only the FINAL run pads (regex anchored at end)."""
    assert _contour_dir_name("SC-N-9") == "SC-N-09"


# --- _fill: rasterize a closed polygon (x=col,y=row) to a boolean mask ---
def test_fill_square_polygon():
    # a 4x4 square (cols 1-4, rows 1-4) into a 6x6 grid
    pts = np.array([[1, 1], [4, 1], [4, 4], [1, 4]], float)   # (x, y)
    m = _fill(pts, (6, 6))
    assert m.dtype == bool and m[2, 2]                         # interior filled
    assert not m[0, 0]                                         # outside corner empty
    assert m.sum() >= 9                                        # at least the interior block


# --- _rasterize: canonical {0,2,3}; endo overrides myo; either-missing cases ---
def test_rasterize_endo_overrides_myo():
    """epi ring -> 2 (myo), endo interior -> 3 (LV-cav), cavity paints OVER the myo fill it sits in."""
    epi = np.array([[1, 1], [8, 1], [8, 8], [1, 8]], float)   # big square
    endo = np.array([[3, 3], [6, 3], [6, 6], [3, 6]], float)  # inner square
    m = _rasterize(endo, epi, (10, 10))
    assert set(np.unique(m)) <= {0, 2, 3}
    assert (m == 3).any() and (m == 2).any()                  # both classes present
    assert m[4, 4] == 3                                        # inner point is cavity, not myo


def test_rasterize_only_endo():
    """No epi contour -> only cavity (3), no myo — some SCD slices lack the ocontour."""
    endo = np.array([[2, 2], [5, 2], [5, 5], [2, 5]], float)
    m = _rasterize(endo, None, (8, 8))
    assert set(np.unique(m)) <= {0, 3} and (m == 3).any() and not (m == 2).any()


def test_rasterize_neither():
    m = _rasterize(None, None, (4, 4))
    assert not m.any()                                        # all background


# --- select_ed_es: larger endo area = ED; single-rec slice ED==ES; spacing from slice steps ---
def _rec(loc, area, px=(1.3, 1.3)):
    return {"loc": loc, "area": area, "img": np.zeros((4, 4)), "mask": np.zeros((4, 4), np.uint8), "px": px}


def test_select_ed_es_area_assignment():
    """Per slice: max-endo-area rec -> ED, min -> ES."""
    by_slice = {10.0: [_rec(10.0, 5), _rec(10.0, 20)], 15.0: [_rec(15.0, 8), _rec(15.0, 30)]}
    ed, es, sp = select_ed_es(by_slice)
    assert [r["area"] for r in ed] == [20, 30]                # larger = ED
    assert [r["area"] for r in es] == [5, 8]                  # smaller = ES
    assert sp[0] == 5.0                                        # z = median inter-slice step (15-10)


def test_select_ed_es_single_rec_ed_equals_es():
    """One rec at a slice -> that rec is both ED and ES for it."""
    by_slice = {10.0: [_rec(10.0, 7)]}
    ed, es, sp = select_ed_es(by_slice)
    assert ed[0]["area"] == es[0]["area"] == 7
    assert sp[0] == 1.3                                        # single slice -> z falls back to px[0]


def test_select_ed_es_spacing_never_negative():
    """Descending slice locations still yield a positive z (abs of the median step)."""
    by_slice = {20.0: [_rec(20.0, 3)], 10.0: [_rec(10.0, 4)]}
    _, _, sp = select_ed_es(by_slice)
    assert sp[0] == 10.0 and sp[1:] == (1.3, 1.3)             # sorted asc -> +10, in-plane passthrough


# --- _root: default sunnybrook path vs explicit override ---
def test_root_default_and_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CARDIAC_DATA", str(tmp_path))
    assert _root().name == "sunnybrook"                       # default under raw/
    assert _root(tmp_path / "x") == tmp_path / "x"            # explicit override


# --- _sax_series_dir: finds CINESAX* dir / None when absent ---
def test_sax_series_dir(tmp_path):
    (tmp_path / "CINESAX_300").mkdir()
    (tmp_path / "CINELAX_5").mkdir()                          # decoy non-SAX series
    assert _sax_series_dir(tmp_path).name == "CINESAX_300"
    assert _sax_series_dir(tmp_path / "empty") is None        # no glob match -> None


# --- scd_cases: patient dir + matching contour dir required (fake tree, no DICOM) ---
def _fake_sunnybrook(base):
    base.mkdir(parents=True, exist_ok=True)
    (base / "scd_patientdata.csv").write_text(
        "PatientID,OriginalID,Pathology,Gender,Age\n"
        "SCD0000101,SC-HF-I-1,HF,M,60\n"
        "SCD0000201,SC-HF-I-2,HF,F,55\n")                     # 2nd lacks a contour dir -> excluded
    (base / "SCD0000101").mkdir()
    (base / "SCD_ManualContours" / "SC-HF-I-01" / _IRCCI).mkdir(parents=True)
    (base / "SCD0000201").mkdir()                             # no contour folder for this one
    return base


def test_scd_cases_requires_contour_dir(tmp_path):
    base = _fake_sunnybrook(tmp_path / "sunnybrook")
    cases = scd_cases(base)
    assert [c.name for c in cases] == ["SCD0000101"]          # only the one with a contour dir


# --- scd_meta: demographics from CSV; no SAX dir -> no DICOM acquisition fields ---
def test_scd_meta_demographics_no_sax(tmp_path):
    base = _fake_sunnybrook(tmp_path / "sunnybrook")
    m = scd_meta(base / "SCD0000101", root=base)
    assert m["group"] == "HF" and m["sex"] == "M" and m["age"] == "60"
    assert m["vendor"] == "GE" and m["country"] == "Canada"   # single-vendor / single-centre constants
    assert "tr_ms" not in m                                    # no CINESAX dir -> DICOM read skipped


# --- _read_contour: [N,2] float x/y from a whitespace txt (skimage polygon coords) ---
def test_read_contour_parses_xy(tmp_path):
    p = tmp_path / "IM-0001-0020-icontour-manual.txt"
    p.write_text("1.0 2.0\n3.5 4.5\n")
    pts = _read_contour(p)
    assert pts.shape == (2, 2) and pts[1].tolist() == [3.5, 4.5]   # x y pixel coords


# --- load_ed_es early returns: missing SAX/contour dir, and icontour-without-DICOM (no match) ---
def test_load_ed_es_no_sax_returns_group_only(tmp_path):
    """No CINESAX dir -> early return with just group/spacing (no ED/ES); DICOM never touched."""
    base = _fake_sunnybrook(tmp_path / "sunnybrook")
    out = load_ed_es(base / "SCD0000101", root=base)          # patient dir has no CINESAX* subdir
    assert out["group"] == "HF" and "ED" not in out and out["spacing"] is None


def test_load_ed_es_icontour_no_dcm_match(tmp_path):
    """An icontour whose instance has no matching .dcm in the SAX series -> skipped -> empty by_slice
    -> early return (the dcm-is-None continue + no-by_slice paths, no real DICOM read)."""
    base = _fake_sunnybrook(tmp_path / "sunnybrook")
    (base / "SCD0000101" / "CINESAX_1").mkdir()               # SAX dir present but holds no .dcm
    cdir = base / "SCD_ManualContours" / "SC-HF-I-01" / _IRCCI
    (cdir / "IM-0001-0020-icontour-manual.txt").write_text("1 1\n2 2\n3 1\n")
    out = load_ed_es(base / "SCD0000101", root=base)
    assert "ED" not in out and out["spacing"] is None         # no dcm match -> nothing gathered


# --- ScdAdapter: thin delegators to the module funcs (default-root, driven via CARDIAC_DATA) ---
def test_scd_adapter_delegates(tmp_path, monkeypatch):
    monkeypatch.setenv("CARDIAC_DATA", str(tmp_path))
    _fake_sunnybrook(tmp_path / "raw" / "sunnybrook")         # default _root -> raw/sunnybrook
    a = ScdAdapter()
    assert a.name == "scd" and a.label_map == {}              # masks built canonical from contours
    assert [c.name for c in a.cases()] == ["SCD0000101"]      # delegates to scd_cases (default root)
    patient = tmp_path / "raw" / "sunnybrook" / "SCD0000101"
    assert a.load_ed_es(patient)["group"] == "HF"            # delegates to load_ed_es (no CINESAX -> early)
    assert a.meta(patient)["vendor"] == "GE"                 # delegates to scd_meta
