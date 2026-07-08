# Vulture whitelist: symbols intentionally unreferenced *within* core+cardioseg — a library
# side-effect setter (vulture can't see torch's semantics) or offline-tooling entrypoints invoked
# out-of-tree (manual CLI + tests, never the running product pipeline). Passed as a vulture scan path
# in [tool.vulture]; ruff-excluded (it's name references, not runnable code). bd cardiac-seg-rta8.
_.benchmark            # torch.backends.cudnn.benchmark setter (train) — side-effect, not a dead attr
build_pool             # offline anatomy/mrxcat pool builder (manual CLI + tests; never product-called)
build_pathology_pool   # offline pathology-pool builder
convert_binary         # one-time ASCII -> binary .vtu mesh converter (offline)
