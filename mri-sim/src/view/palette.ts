// Single source of truth for scene colors. SpinScene (spin tilt), Presenter (gradient
// Larmor), and Legend all read from here so the legend can never drift from what's drawn.
import type { Vec3 } from '../model/types';

export const REST: Vec3 = [0.30, 0.45, 0.85]; // longitudinal / at rest (blue)
export const TIPPED: Vec3 = [1.0, 0.65, 0.15]; // fully transverse (orange)
export const SLICE_PLANE: Vec3 = [0.4, 0.9, 1.0]; // RF transmit slab (cyan)
export const B0_AXIS: Vec3 = [0.5, 0.5, 0.55]; // main-field axis (gray)

/** Spin color by transverse magnitude: rest (blue) → tipped (orange). */
export function transverseColor(d: Vec3): Vec3 {
  const t = Math.min(1, Math.hypot(d[0], d[1]));
  return lerp(REST, TIPPED, t);
}

/** Diverging colormap for Larmor frequency along a gradient: low (blue) ↔ high (red). */
export function freqColor(t: number): Vec3 {
  const u = Math.max(-1, Math.min(1, t));
  return [0.5 + 0.5 * u, 0.35, 0.5 - 0.5 * u];
}

/** Vec3 (0–1 floats) → CSS `rgb(...)` for DOM swatches. */
export function toCss(c: Vec3): string {
  const b = (x: number): number => Math.round(x * 255);
  return `rgb(${b(c[0])},${b(c[1])},${b(c[2])})`;
}

function lerp(a: Vec3, b: Vec3, t: number): Vec3 {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}
