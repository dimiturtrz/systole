"""Unit tests for devtools/omit.py — the coverage-omit glob reader + matcher (the 'non-logic shell' set)."""

from devtools.omit import Omit


def test_matches_omit_glob_semantics():
    assert Omit.matches_omit("pkg/runner.py", ["pkg/*.py"]), "* matches one segment"
    assert not Omit.matches_omit("pkg/sub/runner.py", ["pkg/*.py"]), "* must NOT cross a segment"
    assert Omit.matches_omit("pkg/sub/deep.py", ["pkg/**"]), "** crosses segments"
    assert not Omit.matches_omit("pkg/keep.py", ["other/*.py"]), "non-matching glob is silent"


def test_coverage_omit_reads_pyproject(tmp_path):
    pp = tmp_path / "pyproject.toml"
    pp.write_text('[tool.coverage.run]\nomit = ["a/*.py", "b/**"]\n')
    assert Omit.coverage_omit(str(pp)) == ["a/*.py", "b/**"]
    assert Omit.coverage_omit(str(tmp_path / "absent.toml")) == [], "absent file -> empty omit"
