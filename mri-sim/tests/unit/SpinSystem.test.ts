import { describe, it, expect } from 'vitest';
import { SpinSystem } from '../../src/model/SpinSystem';

// UNIT: pure model, no vtk/WebGL. Fast, deterministic. Physics correctness lives here.

describe('SpinSystem', () => {
  it('creates nx*ny spins', () => {
    const s = new SpinSystem(8, 8, 1.2);
    expect(s.count).toBe(64);
    expect(s.positions.length).toBe(64);
    expect(s.magnetization.length).toBe(64);
  });

  it('initializes every spin at equilibrium along +z (B0)', () => {
    const s = new SpinSystem(4, 4, 1);
    for (const m of s.magnetization) expect(m).toEqual([0, 0, 1]);
  });

  it('centers the grid on the origin', () => {
    const s = new SpinSystem(4, 4, 1);
    const sum: [number, number, number] = [0, 0, 0];
    for (const p of s.positions) {
      sum[0] += p[0];
      sum[1] += p[1];
      sum[2] += p[2];
    }
    expect(Math.abs(sum[0])).toBeLessThan(1e-9);
    expect(Math.abs(sum[1])).toBeLessThan(1e-9);
    expect(sum[2]).toBe(0);
  });

  it('places all spins in the z=0 plane', () => {
    const s = new SpinSystem(3, 3, 2);
    for (const p of s.positions) expect(p[2]).toBe(0);
  });
});
