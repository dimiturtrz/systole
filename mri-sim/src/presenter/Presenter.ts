import { SpinSystem } from '../model/SpinSystem';
import { Simulator } from '../model/Simulator';
import { directionFromAngles } from '../model/physics';
import type { Vec3 } from '../model/types';
import type { SpinView } from '../view/SpinView';

/**
 * Wires model → view. Depends on the SpinView interface (real vtk scene in the app,
 * a fake in tests). `tick(dt)` advances one step (no RAF) so it's unit/integration
 * testable; `run()` drives it with requestAnimationFrame.
 *
 * Individual-proton view: a 3D grid of protons all precessing on cones; a
 * slice-selective pulse tips the central z-slab toward transverse. Re-pulsed each period.
 */
export class Presenter {
  private readonly spins: SpinSystem;
  private readonly sim: Simulator;
  private readonly positions: Vec3[];
  private readonly theta: number[];
  private readonly phase: number[];
  private readonly period = 2;
  private readonly sliceZ = 0;
  private readonly sliceHalf = 0.6;
  private speed = 1;
  private elapsed = 0;
  private last = 0;

  constructor(private readonly view: SpinView) {
    this.spins = new SpinSystem(6, 6, 1.2, 7, 1.0, 0.12); // 6×6 × 7 z-layers; tiny rest tilt → nearly static
    this.sim = new Simulator(0.5, 0.12, 2.5);
    this.positions = this.spins.positions;
    this.theta = [...this.spins.theta];
    this.phase = [...this.spins.phase];
  }

  private directions(): Vec3[] {
    return this.positions.map((_, i) => directionFromAngles(this.theta[i], this.phase[i]));
  }

  start(): void {
    this.view.renderSpins(this.positions, this.directions());
    this.pulse();
    this.elapsed = 0;
    this.view.updateSpins(this.directions());
  }

  setSpeed(s: number): void {
    this.speed = s;
  }

  tick(dt: number): void {
    const d = dt * this.speed;
    this.elapsed += d;
    if (this.elapsed >= this.period) {
      this.pulse();
      this.elapsed = 0;
    }
    this.sim.step(this.theta, this.phase, d);
    this.view.updateSpins(this.directions());
  }

  private pulse(): void {
    this.sim.exciteSlab(this.theta, this.phase, this.positions, this.sliceZ, this.sliceHalf);
  }

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
