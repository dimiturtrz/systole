import { describe, it, expect } from 'vitest';
import { Simulator } from '../../src/model/Simulator';
import { magnitude } from '../../src/model/physics';
import type { Vec3 } from '../../src/model/types';

const close = (a: number, b: number, eps = 1e-6) => expect(Math.abs(a - b)).toBeLessThan(eps);

describe('Simulator', () => {
  it('90° RF pulse tips equilibrium spins into the transverse plane', () => {
    const M: Vec3[] = [[0, 0, 1], [0, 0, 1]];
    new Simulator().rfTip(M, 90);
    for (const m of M) {
      close(m[2], 0); // no longer along z
      close(Math.hypot(m[0], m[1]), 1); // fully transverse
    }
  });

  it('step precesses transverse magnetization a quarter turn at 0.25 Hz over 1 s', () => {
    const sim = new Simulator(0.25);
    const M: Vec3[] = [[1, 0, 0]];
    sim.step(M, 1); // 0.25 Hz × 1 s = quarter cycle = 90°
    close(M[0][0], 0);
    close(M[0][1], 1);
    close(M[0][2], 0);
    close(magnitude(M[0]), 1, 1e-9);
  });

  it('rfTip then step keeps spins transverse (Mz≈0) and |M|≈1', () => {
    const sim = new Simulator(0.4);
    const M: Vec3[] = [[0, 0, 1]];
    sim.rfTip(M, 90);
    sim.step(M, 0.37);
    close(M[0][2], 0);
    close(magnitude(M[0]), 1, 1e-9);
  });

  it('relaxation: T2 decays transverse, T1 recovers longitudinal (larmor 0 to isolate)', () => {
    const sim = new Simulator(0, 2, 1); // no precession, T1=2, T2=1
    const M: Vec3[] = [[0, 1, 0]]; // pure transverse
    sim.step(M, 1);
    close(M[0][0], 0);
    close(M[0][1], Math.exp(-1)); // |Mxy| = exp(-dt/T2)
    close(M[0][2], 1 - Math.exp(-0.5)); // Mz = m0·(1 - exp(-dt/T1))
  });

  it('equilibrium spin (0,0,1) is a fixed point of step', () => {
    const sim = new Simulator(1, 2, 0.6);
    const M: Vec3[] = [[0, 0, 1]];
    sim.step(M, 0.5);
    close(M[0][0], 0);
    close(M[0][1], 0);
    close(M[0][2], 1);
  });

  it('sliceSelectiveTip tips only spins inside the z-slab', () => {
    const M: Vec3[] = [[0, 0, 1], [0, 0, 1], [0, 0, 1]];
    const pos: Vec3[] = [[0, 0, -1], [0, 0, 0], [0, 0, 1]];
    new Simulator().sliceSelectiveTip(M, pos, 90, 0, 0.6);
    close(Math.abs(M[1][2]), 0); // z=0 spin tipped → transverse
    expect(M[0]).toEqual([0, 0, 1]); // z=-1 untouched
    expect(M[2]).toEqual([0, 0, 1]); // z=+1 untouched
  });
});
