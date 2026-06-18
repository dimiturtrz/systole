# F1 · Linear algebra (for this pipeline)

Working fluency, tied to where it appears. Reference: Deisenroth ch. 2–4;
3Blue1Brown *Essence of Linear Algebra* for intuition.

## Vectors & matrices
- A **vector** = a point/direction in N-D (a voxel position; a pixel's feature
  vector). A **matrix** = a linear map (rotate/scale/shear) *or* a table of data.
- **Matrix × vector** = apply a linear map to a point. **Matrix × matrix** = compose
  maps. This is the one operation everything below reduces to.

## Affine transforms — the NIfTI affine
- A **linear** map fixes the origin (rotation, scale). An **affine** map = linear +
  translation: `p_mm = A · p_voxel + t`, packed as one 4×4 matrix on homogeneous
  coords (the extra `1`). 
- This *is* the NIfTI **affine** (`data.py` reads it): it sends voxel index (i,j,k)
  → physical position (x,y,z) in mm. The diagonal carries **spacing**
  (1.56, 1.56, 10 mm); off-diagonal carries orientation; the sign flips are the
  radiological convention. Lose/misread it → every mm³ and EF is wrong.
- **Resampling** (preprocessing) = choosing a new grid and resolving each new
  voxel's value from the old one through these transforms — `zoom` is a restricted
  affine (pure scale) on the in-plane axes.

## Convolution as a linear operator
- A conv layer is a **linear map** (then a nonlinearity). The filter slid across the
  image = a big, **sparse, weight-shared** matrix: each output voxel is a weighted
  sum of a local input window, same weights everywhere (translation equivariance).
- "Shared + sparse" is why a U-Net has far fewer parameters than a dense per-pixel
  net and why it generalizes across positions. Receptive field grows because
  composing convs = multiplying these maps.

## Eigen / SVD (awareness)
- **Eigenvectors/values**: directions a map only stretches. **SVD**: any matrix =
  rotate · scale · rotate; the basis of PCA. Shows up later in **statistical shape
  models** (PCA on meshes) and in understanding what feature directions a layer
  emphasizes. Not load-bearing for the current EF pipeline — know it exists.

## Why an engineer needs it here
Spacing/affine bugs are *the* classic medical-imaging error (we already hit the
label-convention one); they're linear-algebra bugs. And "what is a conv, really" =
linear map + shared weights. That's the whole foundation for reading the model.

### (deepen via teaching / quiz on demand)
