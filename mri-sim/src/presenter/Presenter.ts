import { SpinSystem } from '../model/SpinSystem';
import { Simulator } from '../model/Simulator';
import { Acquisition } from '../model/Acquisition';
import { directionFromAngles } from '../model/physics';
import { magnitudeGrid } from '../model/fft';
import type { Vec3 } from '../model/types';
import type { SpinView } from '../view/SpinView';
import type { Panels } from '../view/Panels';
import type { SequenceView } from '../view/SequenceDiagram';

/**
 * Wires model → views on ONE speed-scaled clock. Each TR: an RF pulse at the cycle
 * start (tips the slab), then a readout at TE (acquires one k-space line). The speed
 * slider scales the clock so precession, the TR/TE cycle, and k-space fill all sync.
 * `tick(dt)` is RAF-free → unit/integration testable; `run()` drives it.
 */
export class Presenter {
  private readonly spins: SpinSystem;
  private readonly sim: Simulator;
  private readonly positions: Vec3[];
  private readonly theta: number[];
  private readonly phase: number[];
  private readonly acq?: Acquisition;
  private readonly sliceZ = 0;
  private readonly sliceHalf = 0.6;

  private tr = 2.0; // repetition time (s)
  private te = 0.5; // echo/readout time after the pulse (s)
  private speed = 1;
  private cycleTime = 0;
  private readThisCycle = false;
  private last = 0;

  constructor(
    private readonly view: SpinView,
    private readonly panels?: Panels,
    phantom?: number[][],
    private readonly seq?: SequenceView,
  ) {
    this.spins = new SpinSystem(6, 6, 1.2, 7, 1.0, 0.12);
    this.sim = new Simulator(0.5, 0.12, 2.5);
    this.positions = this.spins.positions;
    this.theta = [...this.spins.theta];
    this.phase = [...this.spins.phase];
    if (panels && phantom) this.acq = new Acquisition(phantom);
  }

  private directions(): Vec3[] {
    return this.positions.map((_, i) => directionFromAngles(this.theta[i], this.phase[i]));
  }

  start(): void {
    this.view.renderSpins(this.positions, this.directions());
    this.pulse();
    this.cycleTime = 0;
    this.readThisCycle = false;
    this.view.updateSpins(this.directions());
    this.drawPanels();
    this.drawSeq();
  }

  setSpeed(s: number): void {
    this.speed = s;
  }

  setTR(v: number): void {
    this.tr = v;
    if (this.te > v * 0.9) this.te = v * 0.9;
  }

  setTE(v: number): void {
    this.te = Math.min(v, this.tr * 0.9);
  }

  tick(dt: number): void {
    const d = dt * this.speed;
    this.cycleTime += d;
    if (this.cycleTime >= this.tr) {
      this.cycleTime -= this.tr;
      this.pulse(); // RF at the start of each TR
      this.readThisCycle = false;
    }
    if (!this.readThisCycle && this.cycleTime >= this.te) {
      this.readout(); // acquire one k-space line at TE
      this.readThisCycle = true;
    }
    this.sim.step(this.theta, this.phase, d);
    this.view.updateSpins(this.directions());
    this.drawSeq();
  }

  private pulse(): void {
    this.sim.exciteSlab(this.theta, this.phase, this.positions, this.sliceZ, this.sliceHalf);
  }

  private readout(): void {
    if (!this.acq) return;
    if (this.acq.done) this.acq.reset();
    else this.acq.acquireNext();
    this.drawPanels();
  }

  private drawPanels(): void {
    if (!this.acq || !this.panels) return;
    this.panels.drawKspace(magnitudeGrid(this.acq.maskedKspace()));
    this.panels.drawImage(this.acq.reconstruct());
  }

  private drawSeq(): void {
    this.seq?.draw({ tr: this.tr, te: this.te, cycleTime: this.cycleTime });
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
