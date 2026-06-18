import { describe, it, expect } from 'vitest';
import { Acquisition } from '../../src/model/Acquisition';
import { diskPhantom } from '../../src/model/phantom';

describe('Acquisition', () => {
  it('starts partial (DC line only), not done', () => {
    const a = new Acquisition(diskPhantom(8));
    expect(a.acquiredLines).toBe(1);
    expect(a.done).toBe(false);
  });

  it('acquiring all lines reconstructs the phantom', () => {
    const ph = diskPhantom(8);
    const a = new Acquisition(ph);
    while (!a.done) a.acquireNext();
    expect(a.done).toBe(true);
    const img = a.reconstruct();
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
        expect(Math.abs(img[y][x] - ph[y][x])).toBeLessThan(1e-6);
      }
    }
  });

  it('reset returns to the starting (DC-only) state', () => {
    const a = new Acquisition(diskPhantom(8));
    while (!a.done) a.acquireNext();
    a.reset();
    expect(a.acquiredLines).toBe(1);
    expect(a.done).toBe(false);
  });
});
