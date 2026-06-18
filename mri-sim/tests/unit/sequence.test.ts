import { describe, it, expect } from 'vitest';
import { seqWindows, stageAt } from '../../src/model/sequence';

describe('pulse-sequence timing', () => {
  const tr = 0.5;
  const te = 0.015;

  it('windows are ordered slice → phase → readout and all finish around TE', () => {
    const w = seqWindows(tr, te);
    expect(w.sliceEnd).toBeLessThan(w.peStart); // no overlap
    expect(w.peEnd).toBeLessThanOrEqual(w.roStart);
    expect(w.roStart).toBeLessThan(te);
    expect(w.roEnd).toBeGreaterThan(te); // readout window straddles the echo
    expect(w.roEnd).toBeLessThan(tr); // everything done well before the next TR
  });

  it('stageAt walks the encodings in order, then idles until TR', () => {
    const w = seqWindows(tr, te);
    expect(stageAt(tr, te, 0)).toBe('slice');
    expect(stageAt(tr, te, (w.peStart + w.peEnd) / 2)).toBe('phase');
    expect(stageAt(tr, te, te)).toBe('freq'); // readout centered on TE
    expect(stageAt(tr, te, w.roEnd + (tr - w.roEnd) / 2)).toBe('idle');
    expect(stageAt(tr, te, tr * 0.99)).toBe('idle'); // long relaxation/wait
  });

  it('ordering holds for a realistic TE ≪ TR (the regime that broke the old TR-keyed windows)', () => {
    const seen = new Set<string>();
    const w = seqWindows(2.0, 0.02);
    for (let ct = 0; ct < 0.05; ct += 0.0005) seen.add(stageAt(2.0, 0.02, ct));
    expect(seen.has('slice')).toBe(true);
    expect(seen.has('phase')).toBe(true);
    expect(seen.has('freq')).toBe(true);
    expect(w.roEnd).toBeLessThan(0.05); // encodes confined to the early TR, not the whole 2 s
  });
});
