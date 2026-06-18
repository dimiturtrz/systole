import { describe, it, expect } from 'vitest';
import { diskPhantom } from '../../src/model/phantom';
import { fft2d, ifft2d, zerosLike } from '../../src/model/fft';

describe('phantom', () => {
  it('is N×N with a body, a bright feature, and empty background', () => {
    const g = diskPhantom(24);
    expect(g.length).toBe(24);
    expect(g[0].length).toBe(24);
    expect(g[12][12]).toBeGreaterThan(0); // center is inside the body
    expect(Math.max(...g.flat())).toBeGreaterThan(1); // a bright feature (>1)
    expect(Math.min(...g.flat())).toBe(0); // background is 0
  });

  it('survives the k-space roundtrip (fft → ifft recovers it)', () => {
    const g = diskPhantom(16);
    const k = fft2d(g, zerosLike(g));
    const back = ifft2d(k.re, k.im);
    for (let y = 0; y < 16; y++) {
      for (let x = 0; x < 16; x++) {
        expect(Math.abs(back.re[y][x] - g[y][x])).toBeLessThan(1e-6);
      }
    }
  });
});
