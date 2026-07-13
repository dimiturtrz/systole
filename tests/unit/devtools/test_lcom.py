"""Unit tests for the LCOM4 cohesion detector (bd cardiac-seg-2r6f): a class whose methods split into
disjoint-state groups is flagged; cohesive classes and the interface/abstract/stub exclusions are not."""
import ast
import textwrap

from devtools.lcom import _is_split_candidate, lcom4, scan


def _cls(src: str) -> ast.ClassDef:
    return next(n for n in ast.walk(ast.parse(textwrap.dedent(src))) if isinstance(n, ast.ClassDef))


def test_cohesive_class_is_one_component():
    """Two methods both touching self.x -> LCOM4 == 1 (cohesive), not a split candidate."""
    c = _cls("""
        class K:
            def __init__(self): self.x = 0
            def a(self): return self.x + 1
            def b(self): return self.x * 2
    """)
    assert lcom4(c)[0] == 1
    assert _is_split_candidate(c) is None


def test_disjoint_state_groups_flagged():
    """One method uses self.a, another self.b, no link -> LCOM4 == 2 = split candidate."""
    c = _cls("""
        class K:
            def __init__(self): self.a = 1; self.b = 2
            def uses_a(self): return self.a
            def uses_b(self): return self.b
    """)
    score, comps = lcom4(c)
    assert score == 2
    assert _is_split_candidate(c) == (2, comps)


def test_method_call_links_components():
    """A calling self.b() connects them even without a shared field -> one component."""
    c = _cls("""
        class K:
            def __init__(self): self.x = 1
            def a(self): return self.b()
            def b(self): return self.x
    """)
    assert lcom4(c)[0] == 1


def test_abstract_base_excluded():
    """An ABC interface (disjoint methods by design) is not a split candidate."""
    c = _cls("""
        class Iface(ABC):
            def extend(self, m): ...
            def compose(self, i): ...
    """)
    assert _is_split_candidate(c) is None


def test_abstractmethod_excluded():
    """A base with an @abstractmethod is an interface -> excluded even without an ABC base name."""
    c = _cls("""
        class Iface:
            @abstractmethod
            def extend(self, m): ...
            def compose(self, i): return i
    """)
    assert _is_split_candidate(c) is None


def test_interface_impl_excluded():
    """A subclass of a domain base is a polymorphic impl; its split mirrors the contract -> excluded."""
    c = _cls("""
        class AcdcAdapter(DatasetAdapter):
            def cases(self): return self.root
            def meta(self, case): return case
    """)
    assert _is_split_candidate(c) is None


def test_stub_class_excluded():
    """A null-object stub (all trivial bodies) is not a real split candidate."""
    c = _cls("""
        class Noop:
            def metric(self, *a): ...
            def summary(self, *a): pass
    """)
    assert _is_split_candidate(c) is None


def test_static_and_init_excluded_from_graph():
    """__init__ (wires all fields) and staticmethods don't count toward the method graph."""
    c = _cls("""
        class K:
            def __init__(self): self.a = 1; self.b = 2
            @staticmethod
            def helper(x): return x
            def uses_a(self): return self.a
            def uses_b(self): return self.b
    """)
    score, comps = lcom4(c)          # only uses_a / uses_b are graphed -> 2 components
    assert score == 2
    assert sorted(sum(comps, [])) == ["uses_a", "uses_b"]


def test_scan_returns_empty_on_cohesive_package(tmp_path):
    """A file of only cohesive/interface classes yields no rows."""
    (tmp_path / "m.py").write_text(textwrap.dedent("""
        class K:
            def __init__(self): self.x = 0
            def a(self): return self.x
            def b(self): return self.x + 1
    """))
    assert scan([str(tmp_path)]) == []
