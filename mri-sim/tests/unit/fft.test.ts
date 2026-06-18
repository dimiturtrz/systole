import { describe, it, expect } from 'vitest';
import { fft2d, ifft2d, magnitudeGrid, zerosLike } from '../../src/model/fft';

const close = (a: number, b: number, eps = 1e-9) => expect(Math.abs(a - b)).toBeLessThan(eps);

describe('2D FFT', () => {
  it('inverse undoes forward (roundtrip recovers the image)', () => {
    const re = [
      [1, 2, 3, 4],
      [0, -1, 2, 1],
      [3, 3, 0, -2],
      [1, 0, 1, 0],
    ];
    const im = zerosLike(re);
    const k = fft2d(re, im);
    const back = ifft2d(k.re, k.im);
    for (let y = 0; y < 4; y++) {
      for (let x = 0; x < 4; x++) {
        close(back.re[y][x], re[y][x]);
        close(back.im[y][x], 0);
      }
    }
  });

  it('a single bright pixel (delta) → flat k-space magnitude', () => {
    const N = 4;
    const re = Array.from({ length: N }, () => Array.from({ length: N }, () => 0));
    re[0][0] = 1;
    const mag = magnitudeGrid(fft2d(re, zerosLike(re)));
    for (const row of mag) for (const v of row) close(v, 1);
  });

  it('a constant image → k-space energy only at the origin (low frequency)', () => {
    const N = 4;
    const re = Array.from({ length: N }, () => Array.from({ length: N }, () => 1));
    const mag = magnitudeGrid(fft2d(re, zerosLike(re)));
    close(mag[0][0], N * N); // DC term = sum
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        if (x === 0 && y === 0) continue;
        close(mag[y][x], 0);
      }
    }
  });
});
