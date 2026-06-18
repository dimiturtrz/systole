import type { Vec3 } from '../model/types';

/**
 * View interface the presenter depends on (dependency inversion).
 * Real impl = vtk.js SpinScene; tests inject a fake → presenter testable headless.
 */
export interface SpinView {
  /** Build the spins once (positions fixed thereafter). */
  renderSpins(positions: Vec3[], magnetization: Vec3[]): void;
  /** Re-orient existing spins to new magnetization (per animation frame). */
  updateSpins(magnetization: Vec3[]): void;
}
