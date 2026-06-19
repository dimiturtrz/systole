import { describe, it, expect } from 'vitest';
import { largestCcPerClass } from '../../src/postprocess';

// Build a d×(w·h) mask stack from a fill callback (z,y,x)->label.
function stack(d: number, w: number, h: number, fill: (z: number, y: number, x: number) => number): Uint8Array[] {
  const out = Array.from({ length: d }, () => new Uint8Array(w * h));
  for (let z = 0; z < d; z++) for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[z][y * w + x] = fill(z, y, x);
  return out;
}
const count = (m: Uint8Array[], lab: number) => m.reduce((s, sl) => s + sl.reduce((a, v) => a + (v === lab ? 1 : 0), 0), 0);

describe('largestCcPerClass', () => {
  const W = 16, H = 16, D = 4;
  const inBlob = (z: number, y: number, x: number, r: number) =>
    z === 2 && Math.abs(y - 8) < r && Math.abs(x - 8) < r;

  it('single component per class is unchanged (identity)', () => {
    const m = stack(D, W, H, (z, y, x) => (inBlob(z, y, x, 3) ? 3 : 0));
    const out = largestCcPerClass(m, W, H);
    expect(count(out, 3)).toBe(count(m, 3));
  });

  it('drops a disconnected island, keeps the largest', () => {
    const m = stack(D, W, H, (z, y, x) => (inBlob(z, y, x, 4) ? 3 : 0));
    m[0][0] = 3; // stray speck, disconnected
    const out = largestCcPerClass(m, W, H);
    expect(out[0][0]).toBe(0);            // island gone
    expect(count(out, 3)).toBe(count(m, 3) - 1); // everything else kept
  });

  it('cleans each class independently; absent class stays absent', () => {
    const m = stack(D, W, H, (z, y, x) => {
      if (z === 1 && Math.abs(y - 4) < 3 && Math.abs(x - 4) < 3) return 1; // RV blob
      if (z === 1 && Math.abs(y - 12) < 3 && Math.abs(x - 12) < 3) return 2; // myo blob
      return 0;
    });
    m[3][0] = 1; // RV island
    const out = largestCcPerClass(m, W, H);
    expect(count(out, 2)).toBe(count(m, 2)); // myo untouched
    expect(out[3][0]).toBe(0);               // RV island dropped
    expect(count(out, 3)).toBe(0);           // absent class stays absent
  });

  it('empty mask stays empty', () => {
    const m = stack(D, W, H, () => 0);
    const out = largestCcPerClass(m, W, H);
    expect(count(out, 1) + count(out, 2) + count(out, 3)).toBe(0);
  });

  it('keeps a component spanning multiple slices (3D connectivity)', () => {
    // a 2×2 column through all D slices at (8,8) — one 3D component
    const m = stack(D, W, H, (_z, y, x) => (y >= 8 && y < 10 && x >= 8 && x < 10 ? 3 : 0));
    const out = largestCcPerClass(m, W, H);
    expect(count(out, 3)).toBe(D * 4); // all slices' voxels survive (connected in z)
  });
});
