import { SpinSystem } from '../model/SpinSystem';
import { Simulator } from '../model/Simulator';
import { Acquisition } from '../model/Acquisition';
import { directionFromAngles } from '../model/physics';
import { stageAt } from '../model/sequence';
import { magnitudeGrid } from '../model/fft';
import type { Vec3 } from '../model/types';
import type { SpinView } from '../view/SpinView';
import type { Panels } from '../view/Panels';
import type { SequenceView } from '../view/SequenceDiagram';

const REST_TILT = 0.12; // matches SpinSystem/Simulator rest tilt
const TIP_DUR = 0.15; // sim-seconds to ramp the RF tip (avoids a teleport snap)
const LARMOR_MIN = 63.8; // MHz — slice-select frequency band (≈1.5 T: γ·B0 ≈ 63.87 MHz)
const LARMOR_MAX = 63.95;

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
  private readonly sliceHalf = 0.6;
  private sliceCenter = 0; // offset of the selected slab along sliceDir (set by Larmor)
  private slab: boolean[] = [];
  private readonly halfX: number;
  private readonly halfY: number;
  private readonly halfZ: number;
  private sliceDir: Vec3 = [0, 0, 1]; // slice-select gradient direction (tiltable)
  private freqDir: Vec3 = [-1, 0, 0]; // in-plane, perpendicular to sliceDir
  private phaseDir: Vec3 = [0, 1, 0]; // in-plane, perpendicular to both
  private tipLeft = 0; // remaining time in the current RF-tip ramp
  private peIndex = 0; // phase-encode step counter (gradient changes each TR)
  private peStep = 0; // current phase-encode value, −1…1

  private tr = 0.5; // repetition time (s) — T1w spin-echo-ish (real MRI: ms–seconds)
  private te = 0.015; // echo time (s) ≈ 15 ms (real TE is much shorter than TR)
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
    this.spins = new SpinSystem(8, 8, 1.0, 8, 0.9, 0.12); // 512 spins — batched renderer has the headroom
    this.sim = new Simulator(0.5, 0.12, 2.5);
    this.positions = this.spins.positions;
    this.theta = [...this.spins.theta];
    this.phase = [...this.spins.phase];
    if (panels && phantom) this.acq = new Acquisition(phantom);
    this.halfX = Math.max(...this.positions.map((p) => Math.abs(p[0]))) + 0.8;
    this.halfY = Math.max(...this.positions.map((p) => Math.abs(p[1]))) + 0.8;
    this.halfZ = Math.max(...this.positions.map((p) => Math.abs(p[2]))) + 0.5;
    this.recomputeSlab();
  }

  private directions(): Vec3[] {
    return this.positions.map((_, i) => directionFromAngles(this.theta[i], this.phase[i]));
  }

  start(): void {
    this.view.renderSpins(this.positions, this.directions());
    this.updateSlicePlane();
    this.pulse();
    this.cycleTime = 0;
    this.readThisCycle = false;
    this.view.updateSpins(this.directions(), this.gradientColors());
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

  /** RF/Larmor frequency (MHz) selects the slice height along the gradient. */
  setLarmor(mhz: number): void {
    const t = ((mhz - LARMOR_MIN) / (LARMOR_MAX - LARMOR_MIN)) * 2 - 1; // → −1…1
    this.sliceCenter = t * (this.halfZ - 0.5);
    this.recomputeSlab();
    this.updateSlicePlane();
  }

  /** Tilt the slice-select gradient in the x–z plane by `deg` → oblique slices. */
  setSliceAngle(deg: number): void {
    const r = (deg * Math.PI) / 180;
    this.sliceDir = [Math.sin(r), 0, Math.cos(r)];
    this.freqDir = [-Math.cos(r), 0, Math.sin(r)]; // ⟂ sliceDir, in x–z
    this.phaseDir = [0, 1, 0]; // ⟂ both
    this.recomputeSlab();
    this.updateSlicePlane();
  }

  private recomputeSlab(): void {
    this.slab = this.positions.map(
      (p) => Math.abs(dot(p, this.sliceDir) - this.sliceCenter) <= this.sliceHalf,
    );
  }

  private updateSlicePlane(): void {
    const S = Math.max(this.halfX, this.halfY);
    const u: Vec3 = [this.freqDir[0] * S, this.freqDir[1] * S, this.freqDir[2] * S];
    const v: Vec3 = [this.phaseDir[0] * S, this.phaseDir[1] * S, this.phaseDir[2] * S];
    const c: Vec3 = [
      this.sliceDir[0] * this.sliceCenter,
      this.sliceDir[1] * this.sliceCenter,
      this.sliceDir[2] * this.sliceCenter,
    ];
    this.view.setSlice(c, u, v);
  }

  /** Active gradient direction now (slice/phase/freq), or null when idle. */
  private gradientDirNow(): Vec3 | null {
    switch (stageAt(this.tr, this.te, this.cycleTime)) {
      case 'slice': return this.sliceDir;
      case 'phase': return this.phaseDir;
      case 'freq': return this.freqDir;
      default: return null;
    }
  }

  /** When a gradient is on, color spins by local Larmor (position along its direction). */
  private gradientColors(): Vec3[] | undefined {
    const dir = this.gradientDirNow();
    if (!dir) return undefined; // no gradient → view colors by transverse mag
    const half = Math.max(...this.positions.map((p) => Math.abs(dot(p, dir)))) || 1;
    return this.positions.map((p) => freqColor(dot(p, dir) / half));
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
    this.view.updateSpins(this.directions(), this.gradientColors());
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
    if (!this.acq) return;
    if (this.acq.done) this.acq.reset(); // loop: one k-space line per TR, rebuild when full
    else this.acq.acquireNext();
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

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/** Diverging colormap for Larmor frequency: low (blue) ↔ high (red). */
function freqColor(t: number): Vec3 {
  const u = Math.max(-1, Math.min(1, t));
  return [0.5 + 0.5 * u, 0.35, 0.5 - 0.5 * u];
}
