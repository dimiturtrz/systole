import { SpinSystem } from '../model/SpinSystem';
import { Simulator } from '../model/Simulator';
import { Acquisition } from '../model/Acquisition';
import { directionFromAngles } from '../model/physics';
import { magnitudeGrid } from '../model/fft';
import type { Vec3 } from '../model/types';
import type { SpinView } from '../view/SpinView';
import type { Panels } from '../view/Panels';
import type { SequenceView } from '../view/SequenceDiagram';

const REST_TILT = 0.12; // matches SpinSystem/Simulator rest tilt
const TIP_DUR = 0.15; // sim-seconds to ramp the RF tip (avoids a teleport snap)

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
  private readonly slab: boolean[];
  private readonly halfX: number;
  private readonly halfY: number;
  private tipLeft = 0; // remaining time in the current RF-tip ramp
  private peIndex = 0; // phase-encode step counter (gradient changes each TR)
  private peStep = 0; // current phase-encode value, −1…1

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
    this.slab = this.positions.map((p) => Math.abs(p[2] - this.sliceZ) <= this.sliceHalf);
    this.halfX = Math.max(...this.positions.map((p) => Math.abs(p[0]))) + 0.8;
    this.halfY = Math.max(...this.positions.map((p) => Math.abs(p[1]))) + 0.8;
  }

  private directions(): Vec3[] {
    return this.positions.map((_, i) => directionFromAngles(this.theta[i], this.phase[i]));
  }

  start(): void {
    this.view.renderSpins(this.positions, this.directions());
    this.view.setSlice(this.sliceZ, this.halfX, this.halfY);
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
    this.sim.step(this.theta, this.phase, d); // precess + relax theta toward rest
    if (this.tipLeft > 0) this.applyTip(d); // smooth RF tip overrides slab theta during the ramp
    const flash = this.tipLeft > 0 ? 0.35 * (Math.max(0, this.tipLeft) / TIP_DUR) : 0;
    this.view.flashSlice(flash); // RF pulse flashes the slice plane
    this.view.updateSpins(this.directions());
    this.drawSeq();
  }

  /** Ramp the slab's tilt rest→90° over TIP_DUR (eased) so the pulse flips smoothly. */
  private applyTip(d: number): void {
    this.tipLeft -= d;
    const prog = Math.min(1, 1 - Math.max(0, this.tipLeft) / TIP_DUR);
    const e = 1 - (1 - prog) * (1 - prog); // ease-out
    const tilt = REST_TILT + (Math.PI / 2 - REST_TILT) * e;
    for (let i = 0; i < this.theta.length; i++) if (this.slab[i]) this.theta[i] = tilt;
  }

  private pulse(): void {
    this.tipLeft = TIP_DUR; // start a smooth RF tip ramp (no instant teleport, no phase reset)
    this.peIndex = (this.peIndex + 1) % 16; // phase-encode steps each TR (RF stays the same)
    this.peStep = (this.peIndex / 15) * 2 - 1;
  }

  private readout(): void {
    if (!this.acq || this.acq.done) return; // fill once, then hold (no jarring blur-reset)
    this.acq.acquireNext();
    this.drawPanels();
  }

  private drawPanels(): void {
    if (!this.acq || !this.panels) return;
    this.panels.drawKspace(magnitudeGrid(this.acq.maskedKspace()));
    this.panels.drawImage(this.acq.reconstruct());
  }

  private drawSeq(): void {
    this.seq?.draw({ tr: this.tr, te: this.te, cycleTime: this.cycleTime, peStep: this.peStep });
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
