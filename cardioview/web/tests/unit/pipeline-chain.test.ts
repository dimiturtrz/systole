import { describe, it, expect } from 'vitest';
import { resizeBilinear, zscore, fitSquare, resampledSize, argmaxChannels, countLabel, volumeMl } from '../../src/preprocess';
import { largestCcPerClass } from '../../src/postprocess';
import { efFrom } from '../../src/metrics';

// Module-pair / pipeline-chain integration for the in-browser import path. Unit tests check
// each helper alone; these check the chains the helpers actually form (the ONNX step omitted —
// we feed synthetic logits in its place), so an output of A is a valid input to B end to end.

describe('preprocess chain: resample -> zscore -> fit_square', () => {
  // resampledSize maps spacing to the common 1.5mm grid; the chain must land on [size,size].
  for (const [spacing, note] of [[3.0, 'downsample'], [0.75, 'upsample'], [1.5, 'identity']] as [number, string][]) {
    it(`lands on the model grid (${note})`, () => {
      const h = 10, w = 12;
      const raw = Float32Array.from({ length: h * w }, (_, i) => 50 + (i % 7) * 10); // uncalibrated
      const nh = resampledSize(h, spacing), nw = resampledSize(w, spacing);
      const res = resizeBilinear(raw, h, w, nh, nw);
      expect(res.length).toBe(nh * nw);              // resample output is a valid zscore input
      const z = zscore(res);
      expect(Math.abs(z.reduce((a, v) => a + v, 0) / z.length)).toBeLessThan(1e-4); // zero-mean
      const sq = fitSquare(z, nh, nw, 32, 0);        // 32 > resampled dims on these inputs -> pad path
      expect(sq.length).toBe(32 * 32);               // chain reaches the square grid
    });
  }

  it('fit_square pads with the z-score mean (no intensity bias at the border)', () => {
    const raw = Float32Array.from({ length: 6 * 6 }, (_, i) => 100 + i);
    const z = zscore(raw);
    const sq = fitSquare(z, 6, 6, 16, 0);            // 16 > 6 -> a pure-pad border
    expect(sq[0]).toBe(0);                            // corner is padding == the z-score mean (0)
  });
});

describe('inference -> EF chain: argmax -> largest-CC -> count -> volume -> EF', () => {
  const W = 8, H = 8, HW = W * H, D = 4, CLASSES = 4, VOX = 1.5 * 1.5 * 8; // mm^3

  // one slice of logits with class 3 (LV-cav) raised on a centred (2r)^2 block
  const lvSlice = (r: number): Float32Array => {
    const L = new Float32Array(CLASSES * HW); // all 0 -> argmax -> bg, except the block
    for (let y = 4 - r; y < 4 + r; y++) for (let x = 4 - r; x < 4 + r; x++) L[3 * HW + y * W + x] = 10;
    return L;
  };
  const stack = (r: number) => Array.from({ length: D }, () => argmaxChannels(lvSlice(r), CLASSES, HW));

  it('a synthetic ED/ES segmentation flows through to a physiological EF', () => {
    const ed = stack(2);                              // 4x4 LV block per slice
    const es = stack(1);                              // 2x2 LV block per slice (contracted)
    ed[0][0] = 3;                                     // stray island in ED

    const edC = largestCcPerClass(ed, W, H);
    const esC = largestCcPerClass(es, W, H);
    expect(edC[0][0]).toBe(0);                        // island dropped by postproc
    expect(countLabel(edC, 3)).toBe(16 * D);         // 4x4 block kept on every slice
    expect(countLabel(esC, 3)).toBe(4 * D);

    const edv = volumeMl(countLabel(edC, 3), VOX);
    const esv = volumeMl(countLabel(esC, 3), VOX);
    expect(edv).toBeGreaterThan(esv);                // diastole larger
    const ef = efFrom(edv, esv);
    expect(ef).toBeGreaterThan(0);
    expect(ef).toBeLessThan(100);
    expect(ef).toBeCloseTo(75, 0);                   // (64-16)/64 = 75%, spacing cancels
  });

  it('empty ED (no LV) -> EF is NaN, not a crash', () => {
    const empty = Array.from({ length: D }, () => new Uint8Array(HW));
    const edv = volumeMl(countLabel(largestCcPerClass(empty, W, H), 3), VOX);
    expect(Number.isNaN(efFrom(edv, 1))).toBe(true);
  });
});
