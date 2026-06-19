import { describe, it, expect } from 'vitest';
import { dot, cross, norm, lerp } from '../../src/model/vec3';
import type { Vec3 } from '../../src/model/types';

// vec3 was extracted from per-file copies; only exercised indirectly elsewhere. Test it directly.

describe('vec3', () => {
  it('dot: orthogonal=0, parallel=product, sign with angle', () => {
    expect(dot([1, 0, 0], [0, 1, 0])).toBe(0); // orthogonal
    expect(dot([2, 0, 0], [3, 0, 0])).toBe(6); // parallel
    expect(dot([1, 0, 0], [-1, 0, 0])).toBe(-1); // opposite -> negative
  });

  it('cross: right-handed, parallel->0, anticommutative, ⟂ to both inputs', () => {
    expect(cross([1, 0, 0], [0, 1, 0])).toEqual([0, 0, 1]); // x × y = z
    expect(cross([1, 0, 0], [2, 0, 0])).toEqual([0, 0, 0]); // parallel -> zero
    const a: Vec3 = [1, 2, 3], b: Vec3 = [4, 5, 6];
    const ab = cross(a, b);
    expect(cross(b, a)).toEqual(ab.map((x) => -x)); // anticommutative
    expect(dot(ab, a)).toBeCloseTo(0); // perpendicular to a
    expect(dot(ab, b)).toBeCloseTo(0); // perpendicular to b
  });

  it('norm: unit length; zero vector is guarded (not NaN)', () => {
    expect(norm([3, 0, 4])).toEqual([0.6, 0, 0.8]); // |.|=5
    expect(norm([0, 0, 0])).toEqual([0, 0, 0]); // || 1 guard, not [NaN,NaN,NaN]
  });

  it('lerp: endpoints and midpoint', () => {
    expect(lerp([0, 0, 0], [10, 20, 30], 0)).toEqual([0, 0, 0]); // t=0 -> a
    expect(lerp([0, 0, 0], [10, 20, 30], 1)).toEqual([10, 20, 30]); // t=1 -> b
    expect(lerp([0, 0, 0], [10, 20, 30], 0.5)).toEqual([5, 10, 15]); // midpoint
  });
});
