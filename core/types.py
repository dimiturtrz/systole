"""Shared type aliases — the shape & units vocabulary for the whole pipeline.

These aliases don't *enforce* shapes at runtime (NumPy arrays are duck-typed); they
*document* them, so a signature tells you the dimensionality and units at a glance.
The axis order convention is fixed everywhere:

    Volume   [D, H, W]   D = number of short-axis slices (through-plane, ~10 mm apart)
                         H, W = in-plane pixel grid (~1.5 mm)
    Slice2D  [H, W]      one short-axis slice
    Batch    [B, C, H, W]  B = batch, C = channels (1 grayscale in, 4 class scores out)

    Spacing  (z, y, x)   physical voxel size in mm, matching Volume's [D, H, W] axes
                         -> z is the slice gap (big), y/x are in-plane (small)

A Mask is an integer label map (values 0/1/2/3 = bg/RV/myo/LV-cav) with the same
shape as its image. An Image is float intensities of the same shape.

`@shapecheck` lives here too — it makes a boundary's jaxtyping annotations (`Float[Tensor, "b c h w"]`)
LIVE via beartype, so a wrong-shape / wrong-dtype array raises at the call. It's O(1) per call (reads
.shape/.dtype), independent of tensor size — negligible at the coarse seams this codebase uses (no tiny
per-element functions). `@shapecheck_off` (checker None) is the escape for a genuinely hot call: the
annotation stays as documentation but is never checked.
"""
# stdlib numeric tower: numpy scalars register with the numbers ABCs, so `numbers.Real` admits
# float/np.float32/np.float64 (and `numbers.Integral` int/np.int*) — the numpy-scalar-tolerant annotation
# for a param that can receive a numpy scalar. Import Real/Integral straight from `numbers` at the sites.
from numbers import Real

import numpy as np
from beartype import BeartypeConf
from beartype import beartype as _beartype
from jaxtyping import jaxtyped

# Geometry / units
Spacing = tuple[Real, Real, Real]         # (z, y, x) mm, matches [D, H, W]; numpy-scalar-tolerant

# Arrays (shape documented in the alias name; not runtime-enforced)
Volume = np.ndarray                       # [D, H, W]   image or mask, 3D
Slice2D = np.ndarray                      # [H, W]      single slice
Image = np.ndarray                        # float intensities
Mask = np.ndarray                         # integer label map (0/1/2/3)
Batch = np.ndarray                        # [B, C, H, W]

# is_pep484_tower: an `int` also satisfies a bare `float` annotation (and np.float64, a real float subclass).
shapecheck = jaxtyped(typechecker=_beartype(conf=BeartypeConf(is_pep484_tower=True)))
shapecheck_off = jaxtyped(typechecker=None)   # hot-path escape: annotation kept as docs, never checked
