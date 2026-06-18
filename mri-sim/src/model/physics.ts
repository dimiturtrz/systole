import type { Vec3 } from './types';

/** Rotate a magnetization vector about the z-axis by theta radians (precession). */
export function rotateAboutZ(m: Vec3, theta: number): Vec3 {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [m[0] * c - m[1] * s, m[0] * s + m[1] * c, m[2]];
}

/** Rotate about the x-axis by alpha radians (an RF pulse applied along x). */
export function rotateAboutX(m: Vec3, alpha: number): Vec3 {
  const c = Math.cos(alpha);
  const s = Math.sin(alpha);
  return [m[0], m[1] * c - m[2] * s, m[1] * s + m[2] * c];
}

export function magnitude(m: Vec3): number {
  return Math.hypot(m[0], m[1], m[2]);
}

/** Unit direction of a proton at tilt `theta` (from +z) and azimuth `phase`. */
export function directionFromAngles(theta: number, phase: number): Vec3 {
  const st = Math.sin(theta);
  return [st * Math.cos(phase), st * Math.sin(phase), Math.cos(theta)];
}

const DEG = 180 / Math.PI;

/**
 * Axis+angle (degrees) that rotates the +X axis onto direction `d`.
 * Returns a UNIT axis (renderers need a normalized axis — a non-unit axis makes the
 * rotation degenerate, which previously made arrows vanish near the ±X direction).
 */
export function directionToAxisAngleDeg(d: Vec3): { axis: Vec3; angleDeg: number } {
  const len = Math.hypot(d[0], d[1], d[2]) || 1;
  const x = d[0] / len, y = d[1] / len, z = d[2] / len;
  const angleDeg = Math.acos(Math.max(-1, Math.min(1, x))) * DEG;
  let ax = 0, ay = -z, az = y; // cross(+X, d)
  const an = Math.hypot(ax, ay, az);
  if (an < 1e-8) {
    return { axis: [0, 1, 0], angleDeg }; // d parallel ±X → any perpendicular axis
  }
  return { axis: [ax / an, ay / an, az / an], angleDeg };
}
