"""Unit tests for devtools/analytics.py — the code-size / complexity explorer's counting logic."""

from devtools.analytics import Analytics


def test_analyze_file_counts_defs_branches_and_code(tmp_path):
    src = (
        "# a comment (not counted)\n"
        "\n"  # blank (not counted)
        "def f(x):\n"  # def 1
        "    if x:\n"  # branch: If
        "        return [i for i in x]\n"  # branch: comprehension
        "    return x and 1\n"  # branch: BoolOp
        "\n"
        "def g():\n"  # def 2
        "    for _ in range(3):\n"  # branch: For
        "        pass\n"
    )
    p = tmp_path / "snippet.py"
    p.write_text(src)
    stat = Analytics.analyze_file(p)
    assert stat.defs == 2, "f, g"
    assert stat.branches == 4, "If, comprehension, BoolOp, For"
    assert stat.code_lines == 7, "10 lines - 1 comment - 2 blank"


def test_code_lines_excludes_blank_and_comment():
    assert Analytics._code_lines("a = 1\n# c\n\n   \nb = 2\n") == 2, "only the two assignments count"
