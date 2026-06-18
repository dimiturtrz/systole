// Pure measurement helpers — no DOM, no vtk. Unit-tested.

export interface Vols {
  ef: number; // ejection fraction, %
  edv: number; // end-diastolic volume (ml, full)
  esv: number; // end-systolic volume (ml, empty)
}

/** EF from volumes: (EDV − ESV) / EDV, in percent. The clinical identity. */
export function efFrom(edv: number, esv: number): number {
  return edv > 0 ? ((edv - esv) / edv) * 100 : NaN;
}

/** Absolute EF error (model vs ground truth), percentage points. */
export function efError(a: number, b: number): number {
  return Math.abs(a - b);
}

/** Rough clinical EF bands (LV). For a descriptive label only — not a diagnosis. */
export function efCategory(ef: number): string {
  if (ef >= 50) return 'normal';
  if (ef >= 40) return 'mildly reduced';
  if (ef >= 30) return 'moderately reduced';
  return 'severely reduced';
}

export const fmtMl = (v: number): string => `${v.toFixed(0)} mL`;
export const fmtPct = (v: number): string => `${v.toFixed(0)}%`;
