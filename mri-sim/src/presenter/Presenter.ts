import { SpinSystem } from '../model/SpinSystem';
import { Simulator } from '../model/Simulator';
import type { Vec3 } from '../model/types';
import type { SpinView } from '../view/SpinView';

/**
 * Wires model → view. Depends on the SpinView interface (real vtk scene in the app,
 * a fake in tests). `tick(dt)` advances one step (no RAF) so it's unit/integration
 * testable; `run()` drives it with requestAnimationFrame.
 *
 * M2: a 3D grid of spins; a slice-SELECTIVE 90° pulse tips only the central z-slab,
 * which then precesses + relaxes (3D spiral). Spins outside the slab stay at +z.
 * Re-pulsed every `period` seconds so it loops.
 */
export class Presenter {
  private readonly spins: SpinSystem;
  private readonly sim: Simulator;
  private readonly positions: Vec3[];
  private readonly M: Vec3[];
  private readonly period = 4; // seconds between RF pulses
  private readonly sliceZ = 0; // selected slab center
  private readonly sliceHalf = 0.6; // half-thickness (picks the z≈0 layer)
  private elapsed = 0;
  private last = 0;

  constructor(private readonly view: SpinView) {
    this.spins = new SpinSystem(6, 6, 1.2, 7, 1.0); // 6×6 in-plane × 7 z-layers
    this.sim = new Simulator(1.0, 2.0, 0.6); // larmorHz, T1, T2 (visual values)
    this.positions = this.spins.positions;
    this.M = this.spins.magnetization.map((m) => [...m] as Vec3); // own mutable copy
  }

  /** Initial render + first slice-selective 90° pulse. */
  start(): void {
    this.view.renderSpins(this.positions, this.M);
    this.pulse();
    this.elapsed = 0;
    this.view.updateSpins(this.M);
  }

  /** Advance dt seconds: re-pulse on schedule, evolve, refresh the view. */
  tick(dt: number): void {
    this.elapsed += dt;
    if (this.elapsed >= this.period) {
      this.pulse();
      this.elapsed = 0;
    }
    this.sim.step(this.M, dt);
    this.view.updateSpins(this.M);
  }

  private pulse(): void {
    this.sim.sliceSelectiveTip(this.M, this.positions, 90, this.sliceZ, this.sliceHalf);
  }

  /** Start the animation loop (browser only). */
  run(): void {
    const loop = (t: number): void => {
      const dt = this.last ? (t - this.last) / 1000 : 0;
      this.last = t;
      if (dt > 0) this.tick(dt);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }
}
