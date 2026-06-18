import type { Vec3 } from './types';

/**
 * Evolves PROTONS over time — the engine (pure, headless-testable).
 * Each proton precesses around B0 (phase advances at the Larmor rate) on a cone of
 * tilt theta. An RF pulse tips the selected slab toward the transverse plane and aligns
 * its phase (coherence); tilt then relaxes back toward `restTilt` (T1-like).
 * `larmorHz` is a *visual* precession rate, not a real Larmor frequency.
 */
export class Simulator {
  constructor(
    public larmorHz = 0.5,
    public restTilt = 0.3,
    public t1 = 2.5,
  ) {}

  /** Advance time dt: every proton precesses; tilt relaxes toward rest. */
  step(theta: number[], phase: number[], dt: number): void {
    const dphi = 2 * Math.PI * this.larmorHz * dt;
    const e1 = Math.exp(-dt / this.t1);
    for (let i = 0; i < theta.length; i++) {
      phase[i] += dphi;
      theta[i] = this.restTilt + (theta[i] - this.restTilt) * e1;
    }
  }

  /**
   * Slice-selective RF: tip protons inside the z-slab toward the transverse plane
   * (theta → flipRad) and align their phase (coherent precession). Others untouched.
   */
  exciteSlab(
    theta: number[],
    phase: number[],
    positions: Vec3[],
    zCenter: number,
    zHalfThickness: number,
    flipRad: number = Math.PI / 2,
  ): void {
    for (let i = 0; i < theta.length; i++) {
      if (Math.abs(positions[i][2] - zCenter) <= zHalfThickness) {
        theta[i] = flipRad;
        phase[i] = 0; // coherent: the slab precesses together → a real net signal
      }
    }
  }
}
