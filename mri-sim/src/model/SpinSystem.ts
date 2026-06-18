import type { Vec3 } from './types';

/**
 * A grid of net-magnetization vectors — the MODEL (no rendering).
 * 3D grid (nx × ny × nz) centered on the origin; every spin at equilibrium (+z = B0).
 * nz defaults to 1 (a single z=0 plane) for backward compatibility.
 */
export class SpinSystem {
  readonly positions: Vec3[] = [];
  readonly magnetization: Vec3[] = [];

  constructor(
    readonly nx: number,
    readonly ny: number,
    readonly spacing: number = 1,
    readonly nz: number = 1,
    readonly zSpacing: number = spacing,
  ) {
    const ox = ((nx - 1) * spacing) / 2;
    const oy = ((ny - 1) * spacing) / 2;
    const oz = ((nz - 1) * zSpacing) / 2;
    for (let k = 0; k < nz; k++) {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          this.positions.push([i * spacing - ox, j * spacing - oy, k * zSpacing - oz]);
          this.magnetization.push([0, 0, 1]); // equilibrium: aligned with B0 (z)
        }
      }
    }
  }

  get count(): number {
    return this.positions.length;
  }
}
