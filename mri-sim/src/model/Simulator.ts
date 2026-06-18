import type { Vec3 } from './types';
import { rotateAboutX, rotateAboutZ } from './physics';

/**
 * Evolves magnetization over time — the engine (pure, headless-testable).
 * Free precession (about z) + Bloch relaxation: transverse decays with T2,
 * longitudinal recovers toward m0 with T1.
 *
 * Relaxation defaults OFF (T1=T2=Infinity → exp(0)=1 → no change), so pure-precession
 * behaviour is the default; the app sets finite T1/T2 for the realistic 3D spiral.
 * `larmorHz` is a *visual* precession rate, not a real Larmor frequency.
 */
export class Simulator {
  constructor(
    public larmorHz = 0.4,
    public t1 = Infinity,
    public t2 = Infinity,
    public m0 = 1,
  ) {}

  /** Apply an RF pulse: tip every spin by `flipDeg` about the x-axis. */
  rfTip(M: Vec3[], flipDeg: number): void {
    const a = (flipDeg * Math.PI) / 180;
    for (let i = 0; i < M.length; i++) M[i] = rotateAboutX(M[i], a);
  }

  /** Advance time by `dt` s: precess about z, then relax (T2 transverse, T1 longitudinal). */
  step(M: Vec3[], dt: number): void {
    const theta = 2 * Math.PI * this.larmorHz * dt;
    const e1 = Math.exp(-dt / this.t1);
    const e2 = Math.exp(-dt / this.t2);
    for (let i = 0; i < M.length; i++) {
      const m = rotateAboutZ(M[i], theta);
      M[i] = [m[0] * e2, m[1] * e2, this.m0 + (m[2] - this.m0) * e1];
    }
  }
}
