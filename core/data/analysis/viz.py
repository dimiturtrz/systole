"""Marching-cubes surface mesh per chamber (saves ASCII STL; VTK optional later)."""
from skimage import measure


def mesh_label(mask, label, spacing=(1, 1, 1), out="chamber.stl"):
    """Extract a surface mesh for one label and write an ASCII STL. Returns (verts, faces)."""
    verts, faces, _, _ = measure.marching_cubes(
        (mask == label).astype(float), level=0.5, spacing=tuple(spacing))
    _save_stl(verts, faces, out)
    return verts, faces


def _save_stl(verts, faces, out):
    with open(out, "w") as f:
        f.write("solid chamber\n")
        for tri in faces:
            f.write("facet normal 0 0 0\n outer loop\n")
            for idx in tri:
                v = verts[idx]
                f.write(f"  vertex {v[0]} {v[1]} {v[2]}\n")
            f.write(" endloop\nendfacet\n")
        f.write("endsolid chamber\n")
