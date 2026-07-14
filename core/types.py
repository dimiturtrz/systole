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
"""
import numpy as np

# Numpy-tolerant scalar aliases (bd cardiac-seg-dnx6): a runtime-checked (jaxtyping/beartype) boundary
# often receives a numpy scalar (np.float32 off a stored header, np.int64 from a reduction), which is NOT
# a python float/int subclass — so a param that can take one is annotated `Real`/`Integral`, not bare
# float/int. (np.float64 IS a float subclass and needs no help; these cover np.float32 / np.int*.)
Real = float | np.floating
Integral = int | np.integer

# Geometry / units
Spacing = tuple[Real, Real, Real]         # (z, y, x) mm, matches [D, H, W]; numpy-scalar-tolerant

# Arrays (shape documented in the alias name; not runtime-enforced)
Volume = np.ndarray                       # [D, H, W]   image or mask, 3D
Slice2D = np.ndarray                      # [H, W]      single slice
Image = np.ndarray                        # float intensities
Mask = np.ndarray                         # integer label map (0/1/2/3)
Batch = np.ndarray                        # [B, C, H, W]
