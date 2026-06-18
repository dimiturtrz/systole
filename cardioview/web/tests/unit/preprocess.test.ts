import { describe, it, expect } from 'vitest';
import { zscore, fitSquare, resizeBilinear, argmaxChannels, resampledSize } from '../../src/preprocess';

describe('zscore', () => {
  it('produces zero mean and ~unit std', () => {
    const out = zscore(new Float32Array([1, 2, 3, 4, 5]));
    const mean = out.reduce((a, b) => a + b, 0) / out.length;
    expect(mean).toBeCloseTo(0, 5);
    const std = Math.sqrt(out.reduce((a, b) => a + b * b, 0) / out.length);
    expect(std).toBeCloseTo(1, 3); // eps makes it just under 1
  });
});

describe('fitSquare', () => {
  it('center-pads a smaller slice', () => {
    const out = fitSquare(new Float32Array([1, 1, 1, 1]), 2, 2, 4); // 2x2 -> 4x4, centered
    // padded into rows 1-2, cols 1-2
    expect(out[1 * 4 + 1]).toBe(1);
    expect(out[2 * 4 + 2]).toBe(1);
    expect(out[0]).toBe(0); // corner padding
    expect(out.reduce((a, b) => a + b, 0)).toBe(4);
  });

  it('center-crops a larger slice', () => {
    // 4x4 ramp -> crop center 2x2
    const big = new Float32Array(16).map((_, i) => i);
    const out = fitSquare(big, 4, 4, 2);
    expect(Array.from(out)).toEqual([5, 6, 9, 10]); // center 2x2 of the 4x4 grid
  });
});

describe('resizeBilinear', () => {
  it('is identity when size unchanged', () => {
    const a = new Float32Array([0, 1, 2, 3]);
    expect(Array.from(resizeBilinear(a, 2, 2, 2, 2))).toEqual([0, 1, 2, 3]);
  });

  it('keeps values within input range when upsampling', () => {
    const a = new Float32Array([0, 10, 20, 30]);
    const up = resizeBilinear(a, 2, 2, 4, 4);
    expect(up.length).toBe(16);
    expect(Math.min(...up)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...up)).toBeLessThanOrEqual(30);
  });
});

describe('argmaxChannels', () => {
  it('picks the max-logit class per pixel', () => {
    // 4 classes, 2 pixels. pixel0 -> class2, pixel1 -> class0.
    const logits = new Float32Array([
      0.1, 0.9, // c0
      0.2, 0.1, // c1
      0.8, 0.3, // c2
      0.0, 0.0, // c3
    ]);
    expect(Array.from(argmaxChannels(logits, 4, 2))).toEqual([2, 0]);
  });
});

describe('resampledSize', () => {
  it('rounds n*spacing/1.5', () => {
    expect(resampledSize(216, 1.5625)).toBe(225); // 216*1.5625/1.5 = 225
    expect(resampledSize(100, 1.5)).toBe(100); // already at target
  });
});
