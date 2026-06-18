import '@kitware/vtk.js/Rendering/Profiles/Geometry'; // registers the WebGL backend
import vtkFullScreenRenderWindow from '@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkArrowSource from '@kitware/vtk.js/Filters/Sources/ArrowSource';
import type { Vec3 } from '../model/types';

const DEG = 180 / Math.PI;

/**
 * The VIEW: draws spins as arrow glyphs via vtk.js. No physics here.
 * M0: one arrow actor per spin (simple + reliable; revisit Glyph3DMapper /
 * instancing for large grids). Renderer is swappable behind this class.
 */
export class SpinScene {
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
    try {
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
      // eslint-disable-next-line no-console
      console.log(`[mri-sim] rendered ${this.actors.length} spin arrows`);
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error('[mri-sim] renderSpins failed:', e);
    }
  }

  /** Rotate an actor so the arrow's +X axis aligns with unit-ish direction d. */
  private orientAlong(actor: any, d: Vec3): void {
    const len = Math.hypot(d[0], d[1], d[2]) || 1;
    const x = d[0] / len, y = d[1] / len, z = d[2] / len;
    const dot = Math.max(-1, Math.min(1, x)); // dot with +X axis
    const angle = Math.acos(dot) * DEG;
    // rotation axis = cross(+X, d) = (0, -z, y)
    let ax = 0, ay = -z, az = y;
    if (Math.hypot(ax, ay, az) < 1e-6) {
      // d parallel to ±X: pick any perpendicular axis
      ay = 1;
    }
    actor.rotateWXYZ(angle, ax, ay, az);
  }
}
