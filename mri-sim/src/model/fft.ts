// Small, clear complex DFT (O(N²) per axis — fine for the sim's tiny grids).
// k-space ←→ image is a 2D Fourier transform; this is that math, pure + testable.

export interface Grid2D {
  re: number[][];
  im: number[][];
}

function dft1d(re: number[], im: number[], inverse: boolean): [number[], number[]] {
  const N = re.length;
  const oRe = new Array<number>(N).fill(0);
  const oIm = new Array<number>(N).fill(0);
  const s = inverse ? 1 : -1;
  for (let k = 0; k < N; k++) {
    let sr = 0;
    let si = 0;
    for (let n = 0; n < N; n++) {
      const a = (s * 2 * Math.PI * k * n) / N;
      const c = Math.cos(a);
      const sn = Math.sin(a);
      sr += re[n] * c - im[n] * sn;
      si += re[n] * sn + im[n] * c;
    }
    if (inverse) {
      sr /= N;
      si /= N;
    }
    oRe[k] = sr;
    oIm[k] = si;
  }
  return [oRe, oIm];
}

function transform2d(re: number[][], im: number[][], inverse: boolean): Grid2D {
  const H = re.length;
  const W = re[0].length;
  const R = re.map((row) => [...row]);
  const I = im.map((row) => [...row]);
  // rows
  for (let y = 0; y < H; y++) {
    const [rr, ii] = dft1d(R[y], I[y], inverse);
    R[y] = rr;
    I[y] = ii;
  }
  // columns
  for (let x = 0; x < W; x++) {
    const cr = R.map((row) => row[x]);
    const ci = I.map((row) => row[x]);
    const [rr, ii] = dft1d(cr, ci, inverse);
    for (let y = 0; y < H; y++) {
      R[y][x] = rr[y];
      I[y][x] = ii[y];
    }
  }
  return { re: R, im: I };
}

/** Forward 2D DFT: image → k-space. */
export function fft2d(re: number[][], im: number[][]): Grid2D {
  return transform2d(re, im, false);
}

/** Inverse 2D DFT: k-space → image. */
export function ifft2d(re: number[][], im: number[][]): Grid2D {
  return transform2d(re, im, true);
}

/** Per-cell magnitude sqrt(re²+im²) — for displaying k-space or an image. */
export function magnitudeGrid(g: Grid2D): number[][] {
  return g.re.map((row, y) => row.map((v, x) => Math.hypot(v, g.im[y][x])));
}

/** A zero-filled imaginary plane matching a real image, for convenience. */
export function zerosLike(re: number[][]): number[][] {
  return re.map((row) => row.map(() => 0));
}
