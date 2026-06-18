import type { Vec3 } from './types';
import { rotateAboutX, rotateAboutZ } from './physics';

/**
 * Evolves magnetization over time — the engine (pure, headless-testable).
 * M1: RF tip (about x) + free precession (about z). T1/T2 relaxation comes later.
 * `larmorHz` is a *visual* precession rate, not a real Larmor frequency.
 */
export class Simulator {
  constructor(public larmorHz = 0.4) {}

  /** Apply an RF pulse: tip every spin by `flipDeg` about the x-axis. */
  rfTip(M: Vec3[], flipDeg: number): void {
    const a = (flipDeg * Math.PI) / 180;
    for (let i = 0; i < M.length; i++) M[i] = rotateAboutX(M[i], a);
  }

  /** Advance time by `dt` seconds: precess transverse components about z. */
  step(M: Vec3[], dt: number): void {
    const theta = 2 * Math.PI * this.larmorHz * dt;
    for (let i = 0; i < M.length; i++) M[i] = rotateAboutZ(M[i], theta);
  }
}
