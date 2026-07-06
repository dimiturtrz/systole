"""Ingestion layer — the unifying data primitives above static/ (real) and dynamic/ (synth).

Source (StaticSource/DynamicSource) + Split families + TestSet: coded, versioned, hash-frozen. Both
real and synth flow through here as peer sources; cardioseg consumes the resolved train/val/test.
"""
