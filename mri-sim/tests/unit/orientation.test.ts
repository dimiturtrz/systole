import { describe, it, expect } from 'vitest';
import { directionToAxisAngleDeg } from '../../src/model/physics';
import type { Vec3 } from '../../src/model/types';

// Rodrigues rotation of v about UNIT axis k by angle (rad).
function rot(v: Vec3, k: Vec3, ang: number): Vec3 {
  const c = Math.cos(ang), s = Math.sin(ang);
  const dot = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];
  const cr: Vec3 = [k[1] * v[2] - k[2] * v[1], k[2] * v[0] - k[0] * v[2], k[0] * v[1] - k[1] * v[0]];
  return [
    v[0] * c + cr[0] * s + k[0] * dot * (1 - c),
    v[1] * c + cr[1] * s + k[1] * dot * (1 - c),
    v[2] * c + cr[2] * s + k[2] * dot * (1 - c),
  ];
}

const X: Vec3 = [1, 0, 0];

describe('directionToAxisAngleDeg', () => {
  it('rotates +X onto the target direction (transverse + degenerate cases)', () => {
    // includes ±X (degenerate axis) and near-±X transverse — the case that made arrows vanish
    const dirs: Vec3[] = [
      [0, 0, 1], [0, -1, 0], [0, 1, 0], [1, 0, 0], [-1, 0, 0],
      [0.3, -0.6, 0.74], [0.999, 0.001, 0], [0.0001, 1, 0],
    ];
    for (const d of dirs) {
      const { axis, angleDeg } = directionToAxisAngleDeg(d);
      const r = rot(X, axis, (angleDeg * Math.PI) / 180);
      const len = Math.hypot(d[0], d[1], d[2]);
      expect(Math.abs(r[0] - d[0] / len)).toBeLessThan(1e-6);
      expect(Math.abs(r[1] - d[1] / len)).toBeLessThan(1e-6);
      expect(Math.abs(r[2] - d[2] / len)).toBeLessThan(1e-6);
    }
  });

  it('always returns a UNIT axis (non-unit axis = the disappearing-arrows bug)', () => {
    const dirs: Vec3[] = [[0, 0, 1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0.2, 0.2, 0.95], [0.5, -0.5, 0]];
    for (const d of dirs) {
      const { axis } = directionToAxisAngleDeg(d);
      expect(Math.abs(Math.hypot(axis[0], axis[1], axis[2]) - 1)).toBeLessThan(1e-9);
    }
  });
});
