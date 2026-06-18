import { describe, it, expect } from 'vitest';
import { rotateAboutX, rotateAboutZ, magnitude } from '../../src/model/physics';
import type { Vec3 } from '../../src/model/types';

const close = (a: number, b: number, eps = 1e-9) => expect(Math.abs(a - b)).toBeLessThan(eps);

describe('physics rotations', () => {
  it('90° about x tips +z into the transverse plane', () => {
    const m = rotateAboutX([0, 0, 1], Math.PI / 2);
    close(m[2], 0); // no longer along z
    close(Math.hypot(m[0], m[1]), 1); // fully transverse
  });

  it('rotation about z preserves z and magnitude, advances phase', () => {
    const m: Vec3 = [1, 0, 0];
    const r = rotateAboutZ(m, Math.PI / 2);
    close(r[0], 0);
    close(r[1], 1);
    close(r[2], 0);
    close(magnitude(r), 1);
  });

  it('both rotations preserve magnitude', () => {
    const m: Vec3 = [0.3, -0.6, 0.74];
    close(magnitude(rotateAboutX(m, 1.1)), magnitude(m));
    close(magnitude(rotateAboutZ(m, -2.3)), magnitude(m));
  });
});
