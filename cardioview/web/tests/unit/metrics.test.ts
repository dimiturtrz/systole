import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import { efFrom, efError, efCategory, fmtMl, fmtPct } from '../../src/metrics';

describe('metrics', () => {
  it('efFrom is the clinical identity (EDV−ESV)/EDV', () => {
    expect(efFrom(120, 60)).toBeCloseTo(50);
    expect(efFrom(100, 45)).toBeCloseTo(55);
    expect(efFrom(0, 0)).toBeNaN(); // guard divide-by-zero
    expect(efFrom(100, 0)).toBeCloseTo(100); // empty systole -> full ejection
    expect(efFrom(80, 80)).toBeCloseTo(0); // no contraction -> EF 0
  });

  it('efError is symmetric absolute difference', () => {
    expect(efError(55, 48)).toBeCloseTo(7);
    expect(efError(48, 55)).toBeCloseTo(7);
  });

  it('efCategory bands + exact boundary edges (>= is inclusive)', () => {
    expect(efCategory(60)).toBe('normal');
    expect(efCategory(50)).toBe('normal'); // 50 is the inclusive edge
    expect(efCategory(49.9)).toBe('mildly reduced');
    expect(efCategory(40)).toBe('mildly reduced');
    expect(efCategory(39.9)).toBe('moderately reduced');
    expect(efCategory(30)).toBe('moderately reduced');
    expect(efCategory(29.9)).toBe('severely reduced');
  });

  it('formatters render clinical units', () => {
    expect(fmtMl(123.4)).toBe('123 mL');
    expect(fmtPct(54.6)).toBe('55%');
  });
});

// Integration: the exported manifest's EF must equal EF derived from its own EDV/ESV —
// catches drift between what's computed and what's shown.
describe('manifest self-consistency', () => {
  const path = 'public/data/manifest.json';
  it.skipIf(!existsSync(path))('pred.ef ≈ efFrom(pred.edv, pred.esv) for every heart', () => {
    const data = JSON.parse(readFileSync(path, 'utf8'));
    const entries = Array.isArray(data) ? data : data.hearts; // {model, hearts} or legacy array
    expect(entries.length).toBeGreaterThan(0);
    for (const e of entries) {
      expect(efFrom(e.pred.edv, e.pred.esv)).toBeCloseTo(e.pred.ef, 0); // within rounding
      expect(e.glb.ED ?? e.glb.ES).toBeTruthy(); // has at least one mesh
    }
  });
});
