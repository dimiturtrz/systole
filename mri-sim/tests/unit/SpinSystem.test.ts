import { describe, it, expect } from 'vitest';
import { SpinSystem } from '../../src/model/SpinSystem';

// UNIT: pure model (proton grid), no vtk/WebGL.

describe('SpinSystem (proton grid)', () => {
  it('creates nx*ny*nz protons', () => {
    const s = new SpinSystem(6, 6, 1.2, 7, 1.0);
    expect(s.count).toBe(6 * 6 * 7);
    expect(s.theta.length).toBe(s.count);
    expect(s.phase.length).toBe(s.count);
  });

  it('all protons start at the rest tilt', () => {
    const s = new SpinSystem(4, 4, 1, 1, 1, 0.3);
    for (const t of s.theta) expect(t).toBeCloseTo(0.3, 12);
  });

  it('phases are spread, not all identical', () => {
    const s = new SpinSystem(4, 4, 1);
    expect(new Set(s.phase.map((p) => p.toFixed(3))).size).toBeGreaterThan(1);
  });

  it('directions are unit vectors with cosθ as the z-component', () => {
    const s = new SpinSystem(3, 3, 1, 1, 1, 0.4);
    for (const d of SpinSystem.directions(s.theta, s.phase)) {
      expect(Math.hypot(d[0], d[1], d[2])).toBeCloseTo(1, 12);
      expect(d[2]).toBeCloseTo(Math.cos(0.4), 12);
    }
  });

  it('builds a 3D grid centered on z with nz layers', () => {
    const s = new SpinSystem(2, 2, 1, 3, 1);
    expect(s.count).toBe(12);
    const zs = [...new Set(s.positions.map((p) => p[2]))].sort((a, b) => a - b);
    expect(zs).toEqual([-1, 0, 1]);
  });

  it('places protons in the z=0 plane when nz=1', () => {
    const s = new SpinSystem(3, 3, 2);
    for (const p of s.positions) expect(p[2]).toBe(0);
  });
});
