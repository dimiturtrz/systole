import { describe, it, expect } from 'vitest';
import { CHAMBERS, chamberSlot } from '../../src/chambers';

describe('chamberSlot', () => {
  it('maps each chamber color to its own slot', () => {
    CHAMBERS.forEach((c, i) => {
      expect(chamberSlot(...c.color)).toBe(i);
    });
  });

  it('tolerates the small color shift from a glTF roundtrip', () => {
    expect(chamberSlot(0.9, 0.3, 0.28)).toBe(0); // LV cavity red-ish
    expect(chamberSlot(0.98, 0.78, 0.4)).toBe(1); // myocardium gold-ish
    expect(chamberSlot(0.4, 0.55, 0.9)).toBe(2); // RV blue-ish
  });

  it('returns -1 for an unrecognized color', () => {
    expect(chamberSlot(0.5, 0.5, 0.5)).toBe(-1); // grey
    expect(chamberSlot(0, 0, 0)).toBe(-1);
  });

  it('the three chamber colors are mutually distinct slots', () => {
    const slots = CHAMBERS.map((c) => chamberSlot(...c.color));
    expect(new Set(slots).size).toBe(3);
  });
});
