import { describe, it, expect } from 'vitest';
import { wrapIndex } from '../../src/util';

describe('wrapIndex', () => {
  it('passes through in-range indices', () => {
    expect(wrapIndex(0, 5)).toBe(0);
    expect(wrapIndex(4, 5)).toBe(4);
  });

  it('wraps past the end and before the start', () => {
    expect(wrapIndex(5, 5)).toBe(0);
    expect(wrapIndex(7, 5)).toBe(2);
    expect(wrapIndex(-1, 5)).toBe(4);
    expect(wrapIndex(-6, 5)).toBe(4);
  });

  it('is safe for an empty cycle', () => {
    expect(wrapIndex(3, 0)).toBe(0);
  });
});
