import type { Vec3 } from './types';

/**
 * A grid of net-magnetization vectors — the MODEL (no rendering).
 * M0: a 2D grid in the xy-plane, every spin at equilibrium (along +z = B0).
 * Later milestones evolve `magnetization` over time (precession, RF tip, gradients).
 */
export class SpinSystem {
  readonly positions: Vec3[] = [];
  readonly magnetization: Vec3[] = [];

  constructor(
    readonly nx: number,
    readonly ny: number,
    readonly spacing: number = 1,
  ) {
    const ox = ((nx - 1) * spacing) / 2;
    const oy = ((ny - 1) * spacing) / 2;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        this.positions.push([i * spacing - ox, j * spacing - oy, 0]);
        this.magnetization.push([0, 0, 1]); // equilibrium: aligned with B0 (z)
      }
    }
  }

  get count(): number {
    return this.positions.length;
  }
}
