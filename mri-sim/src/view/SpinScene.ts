import '@kitware/vtk.js/Rendering/Profiles/Geometry'; // registers the WebGL backend
import vtkFullScreenRenderWindow from '@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkLineSource from '@kitware/vtk.js/Filters/Sources/LineSource';
import vtkConeSource from '@kitware/vtk.js/Filters/Sources/ConeSource';
import vtkSphereSource from '@kitware/vtk.js/Filters/Sources/SphereSource';
import type { Vec3 } from '../model/types';
import type { SpinView } from './SpinView';

const REST: Vec3 = [0.30, 0.45, 0.85]; // blue  — resting (|Mxy|≈0)
const HOT: Vec3 = [1.0, 0.65, 0.15]; // orange — excited (|Mxy|≈1)
const SHAFT = 0.7; // shaft length
const HEAD = 0.35; // arrowhead length
const HEAD_C = SHAFT + HEAD / 2; // arrowhead center along the direction

/**
 * The VIEW: each spin = a PROTON sphere at its base position + a line shaft + a cone
 * arrowhead. Direction is applied via the cone/line *geometry* (cone has a `direction`
 * property; line by endpoints) — no per-frame actor rotation, so orientations are exact
 * and never accumulate. Colored by transverse magnitude so the excited slab glows.
 */
export class SpinScene implements SpinView {
  private readonly renderer: any;
  private readonly renderWindow: any;
  private readonly protonSphere: any;
  private positions: Vec3[] = [];
  private lines: any[] = [];
  private cones: any[] = [];
  private protons: any[] = []; // base sphere actors (recolor)
  private shafts: any[] = []; // shaft actors (recolor)
  private heads: any[] = []; // head actors (recolor)

  constructor() {
    const fs = vtkFullScreenRenderWindow.newInstance({ background: [0.06, 0.07, 0.09] });
    this.renderer = fs.getRenderer();
    this.renderWindow = fs.getRenderWindow();
    this.protonSphere = vtkSphereSource.newInstance({ radius: 0.13, thetaResolution: 10, phiResolution: 10 });
  }

  renderSpins(positions: Vec3[], directions: Vec3[]): void {
    this.positions = positions;
    for (let k = 0; k < positions.length; k++) {
      const p = positions[k];
      const d = directions[k];

      // proton at the base
      const pm = vtkMapper.newInstance();
      pm.setInputConnection(this.protonSphere.getOutputPort());
      const pa = vtkActor.newInstance();
      pa.setMapper(pm);
      pa.setPosition(...p);
      this.renderer.addActor(pa);

      // shaft (line)
      const line = vtkLineSource.newInstance({ point1: [...p], point2: this.along(p, d, SHAFT) });
      const lm = vtkMapper.newInstance();
      lm.setInputConnection(line.getOutputPort());
      const la = vtkActor.newInstance();
      la.setMapper(lm);
      la.getProperty().setLineWidth(3);
      this.renderer.addActor(la);

      // arrowhead (cone) — oriented by the source's `direction`, not actor rotation
      const cone = vtkConeSource.newInstance({
        height: HEAD, radius: 0.12, resolution: 14,
        center: this.along(p, d, HEAD_C), direction: [...d],
      });
      const cm = vtkMapper.newInstance();
      cm.setInputConnection(cone.getOutputPort());
      const ca = vtkActor.newInstance();
      ca.setMapper(cm);
      this.renderer.addActor(ca);

      this.protons.push(pa);
      this.lines.push(line);
      this.shafts.push(la);
      this.cones.push(cone);
      this.heads.push(ca);
      this.color(d, pa, la, ca);
    }
    this.addB0Axis(positions);

    this.renderer.resetCamera();
    const cam = this.renderer.getActiveCamera();
    cam.azimuth(35);
    cam.elevation(-25);
    this.renderer.resetCameraClippingRange();
    this.renderWindow.render();
  }

  updateSpins(directions: Vec3[]): void {
    const n = Math.min(this.lines.length, directions.length);
    for (let k = 0; k < n; k++) {
      const p = this.positions[k];
      const d = directions[k];
      this.lines[k].setPoint2(this.along(p, d, SHAFT));
      this.cones[k].setCenter(...this.along(p, d, HEAD_C));
      this.cones[k].setDirection(...d);
      this.color(d, this.protons[k], this.shafts[k], this.heads[k]);
    }
    this.renderWindow.render();
  }

  private along(p: Vec3, d: Vec3, t: number): Vec3 {
    return [p[0] + d[0] * t, p[1] + d[1] * t, p[2] + d[2] * t];
  }

  /** Color by transverse magnitude: resting (blue) → excited (orange). */
  private color(d: Vec3, ...actors: any[]): void {
    const t = Math.min(1, Math.hypot(d[0], d[1]));
    const c: Vec3 = [
      REST[0] + (HOT[0] - REST[0]) * t,
      REST[1] + (HOT[1] - REST[1]) * t,
      REST[2] + (HOT[2] - REST[2]) * t,
    ];
    for (const a of actors) a.getProperty().setColor(c[0], c[1], c[2]);
  }

  /** A dim gray reference arrow for B0 (+z), beside the grid. */
  private addB0Axis(positions: Vec3[]): void {
    let minX = Infinity, minZ = Infinity, maxZ = -Infinity;
    for (const p of positions) {
      if (p[0] < minX) minX = p[0];
      if (p[2] < minZ) minZ = p[2];
      if (p[2] > maxZ) maxZ = p[2];
    }
    const x = minX - 1.5;
    const line = vtkLineSource.newInstance({ point1: [x, 0, minZ - 1], point2: [x, 0, maxZ + 0.6] });
    const lm = vtkMapper.newInstance();
    lm.setInputConnection(line.getOutputPort());
    const la = vtkActor.newInstance();
    la.setMapper(lm);
    la.getProperty().setColor(0.5, 0.5, 0.55);
    la.getProperty().setLineWidth(2);
    this.renderer.addActor(la);

    const cone = vtkConeSource.newInstance({ height: 0.6, radius: 0.2, resolution: 16, center: [x, 0, maxZ + 0.9], direction: [0, 0, 1] });
    const cm = vtkMapper.newInstance();
    cm.setInputConnection(cone.getOutputPort());
    const ca = vtkActor.newInstance();
    ca.setMapper(cm);
    ca.getProperty().setColor(0.5, 0.5, 0.55);
    this.renderer.addActor(ca);
  }
}
