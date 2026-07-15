"""Unit tests for devtools/magic_literals.py — recurring string vocab + repeated dict key-sets + ratchet."""

import sys

import pytest

from devtools.magic_literals import MagicLiterals


def test_magic_literals_flags_recurring_token(write_pkg, tmp_path):
    # a value-position token appearing >= 4x is vocabulary; 3x is incidental
    hot = "".join(f"def f{i}():\n    return g('widget')\n" for i in range(4))
    cold = "".join(f"def c{i}():\n    return g('gadget')\n" for i in range(3))
    pkg = write_pkg(tmp_path, "ml_tokens", hot + cold)
    strings = dict(MagicLiterals([pkg]).scan_strings())
    assert strings == {"widget": 4}, f"only the >=4x token is vocabulary, got {strings}"


def test_magic_literals_defers_comparison_key_and_subscript(write_pkg, tmp_path):
    # the SAME token 4x but all in contexts owned elsewhere (comparison=ruff, dict key + subscript=schema)
    src = (
        "def a(x, d):\n"
        "    if x == 'kind':\n"  # comparison operand -> ruff PLR2004
        "        return d['kind']\n"  # subscript -> field ref
        "    return {'kind': 1}\n"  # dict key -> key-set smell, not a value token
        "def b(x):\n"
        "    return x == 'kind'\n"  # comparison operand again
    )
    pkg = write_pkg(tmp_path, "ml_excluded", src)
    assert MagicLiterals([pkg]).scan_strings() == [], "tokens only in comparison/key/subscript are deferred"


def test_magic_literals_finds_repeated_key_set(write_pkg, tmp_path):
    # the same constant-string key-set built in 2 sites = an implicit record schema
    src = "def a():\n    return {'x': 1, 'y': 2}\ndef b():\n    return {'x': 3, 'y': 4}\n"
    pkg = write_pkg(tmp_path, "ml_keysets", src)
    rows = MagicLiterals([pkg]).scan_key_sets()
    assert len(rows) == 1
    n_sites, keys, _ = rows[0]
    assert n_sites == 2
    assert keys == ("x", "y")
    # a single construction site is not a reused schema
    solo = write_pkg(tmp_path, "ml_solo", "def a():\n    return {'x': 1, 'y': 2}\n")
    assert MagicLiterals([solo]).scan_key_sets() == []


def test_magic_literals_ratchet_bites_over_ceiling():
    check = MagicLiterals.check_ratchet
    assert check(5, 2, 4, 9) == ["strings 5 > 4"], "over the string ceiling must report"
    assert check(2, 12, 9, 11) == ["key-sets 12 > 11"], "over the key-set ceiling must report"
    assert check(5, 12, 9, 20) == [], "under both ceilings is silent"
    assert check(999, 999, None, None) == [], "no ceilings = advisory (report-only), never bites"


def test_magic_literals_ratchet_ceilings_from_pyproject(tmp_path):
    pp = tmp_path / "pyproject.toml"
    pp.write_text("[tool.magic_literals]\nmax_strings = 12\nmax_key_sets = 3\n")
    assert MagicLiterals.ratchet_ceilings(str(pp)) == (12, 3), "the FACT slot drives the enforced ceiling (2cj)"
    # a fresh base ships 0/0 -> enforced-at-zero
    pp.write_text("[tool.magic_literals]\nmax_strings = 0\nmax_key_sets = 0\n")
    assert MagicLiterals.ratchet_ceilings(str(pp)) == (0, 0)
    # absent section/file -> (None, None) = advisory fallback, never bites
    assert MagicLiterals.ratchet_ceilings(str(tmp_path / "none.toml")) == (None, None)
    pp.write_text("[tool.ruff]\nline-length = 120\n")
    assert MagicLiterals.ratchet_ceilings(str(pp)) == (None, None), "no [tool.magic_literals] -> advisory"


def test_magic_literals_main_requires_packages(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["devtools.magic_literals"])
    with pytest.raises(SystemExit) as exc:
        from devtools import magic_literals

        magic_literals.main()
    assert exc.value.code == 2, "no-arg invocation must be an argparse usage error"
