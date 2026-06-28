"""Shared kernel package — modality-agnostic primitives reused by cardioseg (training),
cardioview (viewer), and future modality lanes (CT/echo).

Dependency rule: core <- cardioseg, core <- cardioview; core imports NEITHER.
See bd cardiac-seg-t8p.
"""
