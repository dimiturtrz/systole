import { SpinSystem } from '../model/SpinSystem';
import { SpinScene } from '../view/SpinScene';

/**
 * The PRESENTER: wires model → view. (Later: UI controls, animation loop, sequence.)
 * M0: build a grid of equilibrium spins and render them once.
 */
export class Presenter {
  start(): void {
    const spins = new SpinSystem(8, 8, 1.2);
    const scene = new SpinScene();
    scene.renderSpins(spins.positions, spins.magnetization);
  }
}
