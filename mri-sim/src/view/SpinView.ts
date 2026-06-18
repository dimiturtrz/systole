import type { Vec3 } from '../model/types';

/**
 * View interface the presenter depends on (dependency inversion).
 * Real impl = vtk.js SpinScene; tests inject a fake → presenter testable headless.
 */
export interface SpinView {
  /** Build the spins once (positions fixed thereafter). Optional per-spin RGB. */
  renderSpins(positions: Vec3[], magnetization: Vec3[], colors?: Vec3[]): void;
  /** Re-orient existing spins; optional per-spin RGB (else colored by transverse mag). */
  updateSpins(magnetization: Vec3[], colors?: Vec3[]): void;
  /** Configure the RF slice plane (at z=`z`, spanning ±halfX × ±halfY). */
  setSlice(z: number, halfX: number, halfY: number): void;
  /** RF-pulse flash on the slice plane: opacity 0 (hidden) … ~0.35 (bright). */
  flashSlice(opacity: number): void;
}
