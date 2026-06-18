// A simple 2D proton-density phantom — the "object" being imaged. Its 2D Fourier
// transform is its k-space; inverse-FFT rebuilds it (see fft.ts).

/** N×N density map: a disk body with a dark and a bright feature (recognizable shape). */
export function diskPhantom(N: number): number[][] {
  const g: number[][] = Array.from({ length: N }, () => new Array<number>(N).fill(0));
  const cx = (N - 1) / 2;
  const cy = (N - 1) / 2;
  const R = N * 0.38;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const r = Math.hypot(x - cx, y - cy);
      if (r < R) g[y][x] = 1; // body
      if (Math.hypot(x - cx * 0.65, y - cy * 0.85) < N * 0.1) g[y][x] = 0.25; // dark feature
      if (Math.hypot(x - cx * 1.35, y - cy * 1.1) < N * 0.08) g[y][x] = 1.7; // bright feature
    }
  }
  return g;
}
