import '@kitware/vtk.js/Rendering/Profiles/Geometry'; // registers the WebGL backend
import vtkFullScreenRenderWindow from '@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkArrowSource from '@kitware/vtk.js/Filters/Sources/ArrowSource';
import type { Vec3 } from '../model/types';
import { directionToAxisAngleDeg } from '../model/physics';
import type { SpinView } from './SpinView';

/**
 * The VIEW: draws spins as arrow glyphs via vtk.js. No physics here.
 * M0/M1: one arrow actor per spin (simple + reliable; revisit instancing for large
 * grids). Renderer is swappable behind the SpinView interface.
 */
export class SpinScene implements SpinView {
  private readonly renderer: any;
  private readonly renderWindow: any;
  private readonly arrowSource: any;
  private actors: any[] = [];

  constructor() {
    const fs = vtkFullScreenRenderWindow.newInstance({ background: [0.06, 0.07, 0.09] });
    this.renderer = fs.getRenderer();
    this.renderWindow = fs.getRenderWindow();
    // vtkArrowSource points along +X by default; we rotate each actor to its spin direction.
    this.arrowSource = vtkArrowSource.newInstance({ tipResolution: 16, shaftResolution: 16 });
  }

  /** Build one arrow actor per spin, oriented along its magnetization vector. */
  renderSpins(positions: Vec3[], magnetization: Vec3[]): void {
    const scale = 0.9;
    for (let k = 0; k < positions.length; k++) {
      const mapper = vtkMapper.newInstance();
      mapper.setInputConnection(this.arrowSource.getOutputPort());
      const actor = vtkActor.newInstance();
      actor.setMapper(mapper);
      actor.getProperty().setColor(0.36, 0.82, 0.77);
      actor.setScale(scale, scale, scale);
      actor.setPosition(...positions[k]);
      this.orientAlong(actor, magnetization[k]);
      this.renderer.addActor(actor);
      this.actors.push(actor);
    }
    this.renderer.resetCamera();
    const cam = this.renderer.getActiveCamera();
    cam.azimuth(35);
    cam.elevation(-25);
    this.renderer.resetCameraClippingRange();
    this.renderWindow.render();
  }

  /** Re-orient existing arrows to new magnetization (per frame). */
  updateSpins(magnetization: Vec3[]): void {
    const n = Math.min(this.actors.length, magnetization.length);
    for (let k = 0; k < n; k++) {
      const actor = this.actors[k];
      actor.setOrientation(0, 0, 0); // reset before applying absolute rotation
      this.orientAlong(actor, magnetization[k]);
    }
    this.renderWindow.render();
  }

  /** Rotate an actor so the arrow's +X axis aligns with direction d. */
  private orientAlong(actor: any, d: Vec3): void {
    const { axis, angleDeg } = directionToAxisAngleDeg(d);
    actor.rotateWXYZ(angleDeg, axis[0], axis[1], axis[2]);
  }
}
