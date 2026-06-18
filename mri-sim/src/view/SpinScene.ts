import '@kitware/vtk.js/Rendering/Profiles/Geometry'; // registers the WebGL backend
import vtkFullScreenRenderWindow from '@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkArrowSource from '@kitware/vtk.js/Filters/Sources/ArrowSource';
import type { Vec3 } from '../model/types';
import { directionToAxisAngleDeg, magnitude } from '../model/physics';
import type { SpinView } from './SpinView';

const REST: Vec3 = [0.30, 0.45, 0.85]; // blue  — longitudinal (resting, |Mxy|≈0)
const HOT: Vec3 = [1.0, 0.65, 0.15]; // orange — fully transverse (excited, |Mxy|≈1)

/**
 * The VIEW: spins as arrow glyphs via vtk.js. No physics here.
 * One arrow actor per spin; colored by transverse magnitude so excited spins glow,
 * scaled by |M| (shows decay/recovery), oriented along M. Renderer swappable behind
 * the SpinView interface.
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
    this.arrowSource = vtkArrowSource.newInstance({ tipResolution: 16, shaftResolution: 16 });
  }

  renderSpins(positions: Vec3[], magnetization: Vec3[]): void {
    const base = 0.85;
    for (let k = 0; k < positions.length; k++) {
      const mapper = vtkMapper.newInstance();
      mapper.setInputConnection(this.arrowSource.getOutputPort());
      const actor = vtkActor.newInstance();
      actor.setMapper(mapper);
      actor.setScale(base, base, base);
      actor.setPosition(...positions[k]);
      this.applyColor(actor, magnetization[k]);
      this.orientAlong(actor, magnetization[k]);
      this.renderer.addActor(actor);
      this.actors.push(actor);
    }
    this.addB0Axis(positions);

    this.renderer.resetCamera();
    const cam = this.renderer.getActiveCamera();
    cam.azimuth(35);
    cam.elevation(-25);
    this.renderer.resetCameraClippingRange();
    this.renderWindow.render();
  }

  /** Re-orient + re-scale + re-color existing arrows to new magnetization (per frame). */
  updateSpins(magnetization: Vec3[]): void {
    const base = 0.85;
    const n = Math.min(this.actors.length, magnetization.length);
    for (let k = 0; k < n; k++) {
      const actor = this.actors[k];
      const s = base * magnitude(magnetization[k]); // length ∝ |M|
      actor.setScale(s, s, s);
      actor.setOrientation(0, 0, 0);
      this.orientAlong(actor, magnetization[k]);
      this.applyColor(actor, magnetization[k]);
    }
    this.renderWindow.render();
  }

  /** Color by transverse magnitude: resting (blue) → excited (orange). */
  private applyColor(actor: any, m: Vec3): void {
    const t = Math.min(1, Math.hypot(m[0], m[1])); // |Mxy|
    actor.getProperty().setColor(
      REST[0] + (HOT[0] - REST[0]) * t,
      REST[1] + (HOT[1] - REST[1]) * t,
      REST[2] + (HOT[2] - REST[2]) * t,
    );
  }

  /** Rotate an actor so the arrow's +X axis aligns with direction d. */
  private orientAlong(actor: any, d: Vec3): void {
    const { axis, angleDeg } = directionToAxisAngleDeg(d);
    actor.rotateWXYZ(angleDeg, axis[0], axis[1], axis[2]);
  }

  /** A dim gray reference arrow for B0 (+z), beside the grid. */
  private addB0Axis(positions: Vec3[]): void {
    let minX = Infinity, maxZ = -Infinity, minZ = Infinity;
    for (const p of positions) {
      if (p[0] < minX) minX = p[0];
      if (p[2] > maxZ) maxZ = p[2];
      if (p[2] < minZ) minZ = p[2];
    }
    const len = (maxZ - minZ) + 2;
    const mapper = vtkMapper.newInstance();
    mapper.setInputConnection(this.arrowSource.getOutputPort());
    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);
    actor.setScale(len, len * 0.25, len * 0.25);
    actor.setPosition(minX - 1.5, 0, minZ - 1);
    actor.getProperty().setColor(0.5, 0.5, 0.55);
    this.orientAlong(actor, [0, 0, 1]); // point along +z (B0)
    this.renderer.addActor(actor);
  }
}
