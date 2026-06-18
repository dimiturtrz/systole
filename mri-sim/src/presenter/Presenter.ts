import { SpinSystem } from '../model/SpinSystem';
import { Simulator } from '../model/Simulator';
import type { Vec3 } from '../model/types';
import type { SpinView } from '../view/SpinView';

/**
 * Wires model → view. Depends on the SpinView interface (real vtk scene in the app,
 * a fake in tests). `tick(dt)` advances one step and is pure-ish (no RAF) so the loop
 * can be unit/integration tested; `run()` drives it with requestAnimationFrame.
 */
export class Presenter {
  private readonly spins: SpinSystem;
  private readonly sim: Simulator;
  private readonly M: Vec3[];
  private last = 0;

  constructor(private readonly view: SpinView) {
    this.spins = new SpinSystem(8, 8, 1.2);
    this.sim = new Simulator(0.4);
    this.M = this.spins.magnetization.map((m) => [...m] as Vec3); // own mutable copy
  }

  /** Initial render + a 90° RF pulse (spins → transverse plane). */
  start(): void {
    this.view.renderSpins(this.spins.positions, this.M);
    this.sim.rfTip(this.M, 90);
    this.view.updateSpins(this.M);
  }

  /** Advance the simulation by dt seconds and refresh the view. */
  tick(dt: number): void {
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
