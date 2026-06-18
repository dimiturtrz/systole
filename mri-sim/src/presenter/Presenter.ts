import { SpinSystem } from '../model/SpinSystem';
import { Simulator } from '../model/Simulator';
import type { Vec3 } from '../model/types';
import type { SpinView } from '../view/SpinView';

/**
 * Wires model → view. Depends on the SpinView interface (real vtk scene in the app,
 * a fake in tests). `tick(dt)` advances one step (no RAF) so it's unit/integration
 * testable; `run()` drives it with requestAnimationFrame.
 *
 * Cycle: a 90° RF pulse, then free precession + T1/T2 relaxation (the 3D spiral back
 * to equilibrium); re-pulse every `period` seconds so it loops.
 */
export class Presenter {
  private readonly spins: SpinSystem;
  private readonly sim: Simulator;
  private readonly M: Vec3[];
  private readonly period = 4; // seconds between RF pulses
  private elapsed = 0;
  private last = 0;

  constructor(private readonly view: SpinView) {
    this.spins = new SpinSystem(8, 8, 1.2);
    this.sim = new Simulator(1.0, 2.0, 0.6); // larmorHz, T1, T2 (visual values)
    this.M = this.spins.magnetization.map((m) => [...m] as Vec3); // own mutable copy
  }

  /** Initial render + first 90° RF pulse. */
  start(): void {
    this.view.renderSpins(this.spins.positions, this.M);
    this.sim.rfTip(this.M, 90);
    this.elapsed = 0;
    this.view.updateSpins(this.M);
  }

  /** Advance dt seconds: re-pulse on schedule, evolve, refresh the view. */
  tick(dt: number): void {
    this.elapsed += dt;
    if (this.elapsed >= this.period) {
      this.sim.rfTip(this.M, 90);
      this.elapsed = 0;
    }
    this.sim.step(this.M, dt);
    this.view.updateSpins(this.M);
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
