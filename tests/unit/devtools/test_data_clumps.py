"""Unit tests for devtools/data_clumps.py — Fowler data clumps (maximal travelling param sets)."""

import sys

import pytest

from devtools.data_clumps import DataClumps


def test_data_clumps_finds_maximal_travelling_set(write_pkg, tmp_path):
    # {a,b,c} carried whole by 4 functions -> a clump at support 4 (>= _MIN_SUPPORT)
    src = "def f1(a, b, c): pass\ndef f2(a, b, c, d): pass\ndef f3(a, b, c, e): pass\ndef f4(a, b, c, g): pass\n"
    pkg = write_pkg(tmp_path, "clump_pos", src)
    rows = DataClumps([pkg]).clumps()
    assert rows, "a param set carried by >=4 functions must surface as a clump"
    support, params, size, _ = rows[0]
    assert set(params) == {"a", "b", "c"}
    assert support == 4
    assert size == 3


def test_data_clumps_below_support_is_silent(write_pkg, tmp_path):
    # only 3 functions carry {a,b,c} -> support 3 < 4 -> nothing
    src = "def f1(a, b, c): pass\ndef f2(a, b, c): pass\ndef f3(a, b, c): pass\n"
    pkg = write_pkg(tmp_path, "clump_neg", src)
    assert DataClumps([pkg]).clumps() == []


def test_data_clumps_reports_maximal_not_subset(write_pkg, tmp_path):
    # 4 functions all carry {a,b,c,d}; the maximal set must win, its {a,b,c} subset must be dropped
    src = "".join(f"def f{i}(a, b, c, d): pass\n" for i in range(4))
    pkg = write_pkg(tmp_path, "clump_max", src)
    rows = DataClumps([pkg]).clumps()
    param_sets = {frozenset(params) for _, params, _, _ in rows}
    assert frozenset({"a", "b", "c", "d"}) in param_sets
    assert frozenset({"a", "b", "c"}) not in param_sets, "a subset at the same support must be suppressed"


def test_data_clumps_main_requires_packages(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["devtools.data_clumps"])
    with pytest.raises(SystemExit) as exc:
        from devtools import data_clumps

        data_clumps.main()
    assert exc.value.code == 2, "no-arg invocation must be an argparse usage error"
