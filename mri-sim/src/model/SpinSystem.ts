import type { Vec3 } from './types';
import { directionFromAngles } from './physics';

const GOLDEN = Math.PI * (3 - Math.sqrt(5)); // golden angle → well-spread phases

/**
 * A 3D grid of PROTONS (individual-spin view). Each proton precesses around B0 on a
 * cone of tilt `theta` (from +z) with azimuth `phase`. At rest every proton has a small
 * tilt and an offset phase → they precess incoherently, so their *sum* points along B0
 * (which is what MRI measures), while each one is still visibly precessing.
 */
export class SpinSystem {
  readonly positions: Vec3[] = [];
  readonly theta: number[] = []; // tilt from +z
  readonly phase: number[] = []; // azimuth

  constructor(
    readonly nx: number,
    readonly ny: number,
    readonly spacing: number = 1,
    readonly nz: number = 1,
    readonly zSpacing: number = spacing,
    readonly restTilt: number = 0.3,
  ) {
    const ox = ((nx - 1) * spacing) / 2;
    const oy = ((ny - 1) * spacing) / 2;
    const oz = ((nz - 1) * zSpacing) / 2;
    let idx = 0;
    for (let k = 0; k < nz; k++) {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          this.positions.push([i * spacing - ox, j * spacing - oy, k * zSpacing - oz]);
          this.theta.push(restTilt);
          this.phase.push((idx * GOLDEN) % (2 * Math.PI)); // incoherent but deterministic
          idx++;
        }
      }
    }
  }

  get count(): number {
    return this.positions.length;
  }

  /** Unit magnetization direction of each proton, from its (theta, phase). */
  static directions(theta: number[], phase: number[]): Vec3[] {
    return theta.map((t, i) => directionFromAngles(t, phase[i]));
  }
}
