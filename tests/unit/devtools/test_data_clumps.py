"""Unit tests for the data-clump detector (bd cardiac-seg-o07n): a param SET carried whole by enough
functions surfaces as a clump; subsets are suppressed by the maximal filter; thin support is ignored."""
import textwrap

from devtools.data_clumps import _candidates, _functions, clumps


def _write(tmp_path, name, src):
    (tmp_path / name).write_text(textwrap.dedent(src))


def test_recurring_tuple_surfaces_as_a_clump(tmp_path):
    """A 4-param tuple threaded through 4 functions is one clump with support 4."""
    _write(tmp_path, "m.py", """
        def a(inplane, n4, nyul, norm): ...
        def b(inplane, n4, nyul, norm, extra): ...
        def c(inplane, n4, nyul, norm): ...
        def d(self, inplane, n4, nyul, norm): ...
    """)
    rows = clumps([str(tmp_path)], min_support=4, min_clump=3)
    assert len(rows) == 1
    support, params, size, _files = rows[0]
    assert support == 4
    assert params == ("inplane", "n4", "norm", "nyul")
    assert size == 4


def test_maximal_filter_suppresses_the_subset(tmp_path):
    """The 4-tuple's 3-subsets have the SAME support -> only the maximal 4-set is reported, not its subsets."""
    _write(tmp_path, "m.py", """
        def a(inplane, n4, nyul, norm): ...
        def b(inplane, n4, nyul, norm): ...
        def c(inplane, n4, nyul, norm): ...
        def d(inplane, n4, nyul, norm): ...
    """)
    rows = clumps([str(tmp_path)], min_support=4, min_clump=3)
    assert [r[1] for r in rows] == [("inplane", "n4", "norm", "nyul")]   # no 3-subset rows


def test_below_support_is_not_a_clump(tmp_path):
    """A tuple in only 2 functions (< min_support=4) is not reported."""
    _write(tmp_path, "m.py", """
        def a(x, y, z): ...
        def b(x, y, z): ...
    """)
    assert clumps([str(tmp_path)], min_support=4, min_clump=3) == []


def test_no_transitive_merge_via_hub_param(tmp_path):
    """Two distinct triples sharing one hub param must NOT merge into a 5-blob (the components bug)."""
    _write(tmp_path, "m.py", """
        def a(hub, aa, bb): ...
        def b(hub, aa, bb): ...
        def c(hub, aa, bb): ...
        def d(hub, aa, bb): ...
        def e(hub, cc, dd): ...
        def f(hub, cc, dd): ...
        def g(hub, cc, dd): ...
        def h(hub, cc, dd): ...
    """)
    rows = clumps([str(tmp_path)], min_support=4, min_clump=3)
    sets = {r[1] for r in rows}
    assert ("aa", "bb", "hub") in sets and ("cc", "dd", "hub") in sets
    assert all(size <= 3 for _s, _p, size, _f in rows)          # never a merged 5-set


def test_candidates_respect_min_clump(tmp_path):
    """No candidate subset is smaller than min_clump."""
    _write(tmp_path, "m.py", "def a(p, q, r, s): ...\n")
    funcs = _functions([str(tmp_path)])
    assert all(len(c) >= 3 for c in _candidates(funcs, 3))


def test_self_and_cls_excluded(tmp_path):
    """self/cls never count toward a clump."""
    _write(tmp_path, "m.py", """
        class K:
            def a(self, x, y, z): ...
            def b(self, x, y, z): ...
            def c(self, x, y, z): ...
            def d(self, x, y, z): ...
    """)
    rows = clumps([str(tmp_path)], min_support=4, min_clump=3)
    assert rows and "self" not in rows[0][1]
