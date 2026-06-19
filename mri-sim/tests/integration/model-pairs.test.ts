import { describe, it, expect } from 'vitest';
import { SpinSystem } from '../../src/model/SpinSystem';
import { Simulator } from '../../src/model/Simulator';
import { seqWindows, stageAt, phaseEncodeOffset } from '../../src/model/sequence';

// Integration: the model pieces working on each other's outputs (not each in isolation).
const mag = (v: number[]) => Math.hypot(v[0], v[1], v[2]);

describe('SpinSystem -> Simulator (the system arrays evolve under the simulator)', () => {
  it('precess advances every phase by the Larmor amount; directions stay unit', () => {
    const s = new SpinSystem(4, 4, 1, 1, 1, 0.3);
    const sim = new Simulator(0.5, 0.3, 2.5);
    const before = [...s.phase];
    sim.precess(s.phase, 0.1);
    const dphi = 2 * Math.PI * 0.5 * 0.1;
    for (let i = 0; i < s.count; i++) expect(s.phase[i]).toBeCloseTo(before[i] + dphi, 6);
    for (const d of SpinSystem.directions(s.theta, s.phase)) expect(mag(d)).toBeCloseTo(1, 6);
  });

  it('relax pulls a tipped spin toward rest tilt; resting spins stay put', () => {
    const s = new SpinSystem(2, 2, 1, 1, 1, 0.3);
    const sim = new Simulator(0.5, 0.3, 2.5);
    s.theta[0] = Math.PI / 2; // tipped by an RF pulse
    sim.relax(s.theta, 0.5);
    expect(s.theta[0]).toBeLessThan(Math.PI / 2); // decayed
    expect(s.theta[0]).toBeGreaterThan(0.3); // not past rest
    expect(s.theta[1]).toBeCloseTo(0.3, 6); // an at-rest spin is unchanged
  });

  it('step == precess then relax composed (same arrays)', () => {
    const s = new SpinSystem(2, 2, 1, 1, 1, 0.3);
    const a = new Simulator(0.5, 0.3, 2.5);
    const b = new Simulator(0.5, 0.3, 2.5);
    s.theta[0] = 1.0;
    const th = [...s.theta];
    const ph = [...s.phase];
    a.step(s.theta, s.phase, 0.2);
    b.precess(ph, 0.2);
    b.relax(th, 0.2);
    expect(s.phase).toEqual(ph);
    expect(s.theta).toEqual(th);
  });
});

describe('sequence -> stage selection (windows drive the timeline)', () => {
  const tr = 0.5, te = 0.015;

  it('windows are ordered: slice end < phase < readout', () => {
    const w = seqWindows(tr, te);
    expect(w.sliceEnd).toBeLessThan(w.peStart);
    expect(w.peStart).toBeLessThan(w.peEnd);
    expect(w.peEnd).toBeLessThanOrEqual(w.roStart);
  });

  it('stageAt walks slice -> phase -> freq -> idle across the TR', () => {
    const w = seqWindows(tr, te);
    expect(stageAt(tr, te, w.sliceEnd / 2)).toBe('slice');
    expect(stageAt(tr, te, (w.peStart + w.peEnd) / 2)).toBe('phase');
    expect(stageAt(tr, te, (w.roStart + w.roEnd) / 2)).toBe('freq');
    expect(stageAt(tr, te, tr * 0.9)).toBe('idle'); // long after readout
  });
});

describe('sequence -> phase-encode -> azimuth offset', () => {
  it('winds proportional to gradient & position, sign flips, bounded by ramp progress', () => {
    expect(phaseEncodeOffset(1, 1, 1)).toBeCloseTo(2 * Math.PI, 6); // full gradient, edge spin, full ramp
    expect(phaseEncodeOffset(-1, 1, 1)).toBeCloseTo(-2 * Math.PI, 6); // reversed gradient -> opposite wind
    expect(phaseEncodeOffset(1, 0, 1)).toBe(0); // centre spin -> no wind
    expect(phaseEncodeOffset(1, 1, 0)).toBe(0); // ramp hasn't started
  });
});
