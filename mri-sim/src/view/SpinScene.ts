import '@kitware/vtk.js/Rendering/Profiles/Geometry'; // registers the WebGL backend
import vtkFullScreenRenderWindow from '@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkLineSource from '@kitware/vtk.js/Filters/Sources/LineSource';
import vtkConeSource from '@kitware/vtk.js/Filters/Sources/ConeSource';
import vtkSphereSource from '@kitware/vtk.js/Filters/Sources/SphereSource';
import vtkPlaneSource from '@kitware/vtk.js/Filters/Sources/PlaneSource';
import type { Vec3 } from '../model/types';
import type { SpinView } from './SpinView';

const REST: Vec3 = [0.30, 0.45, 0.85];
const HOT: Vec3 = [1.0, 0.65, 0.15];
const SHAFT = 0.7;
const HEAD = 0.35;
const HEAD_C = SHAFT + HEAD / 2;

/**
 * The VIEW: per spin = proton sphere at base + line shaft + cone arrowhead.
 * Shaft uses line endpoints; the arrowhead is ONE shared cone mesh re-placed each frame
 * via a fresh actor matrix (cheap — no geometry rebuild, no rotation accumulation).
 */
export class SpinScene implements SpinView {
  private readonly renderer: any;
  private readonly renderWindow: any;
  private readonly protonSphere: any;
  private readonly coneSource: any; // shared head geometry (+X), height HEAD
  private positions: Vec3[] = [];
  private lines: any[] = [];
  private protons: any[] = [];
  private shafts: any[] = [];
  private heads: any[] = [];
  private slicePlane: any; // RF slice plane (flashes on pulse)

  constructor() {
    const fs = vtkFullScreenRenderWindow.newInstance({ background: [0.06, 0.07, 0.09] });
    this.renderer = fs.getRenderer();
    this.renderWindow = fs.getRenderWindow();
    this.protonSphere = vtkSphereSource.newInstance({ radius: 0.13, thetaResolution: 8, phiResolution: 8 });
    this.coneSource = vtkConeSource.newInstance({ height: HEAD, radius: 0.12, resolution: 12, direction: [1, 0, 0], center: [0, 0, 0] });
  }

  renderSpins(positions: Vec3[], directions: Vec3[], colors?: Vec3[]): void {
    this.positions = positions;
    for (let k = 0; k < positions.length; k++) {
      const p = positions[k];
      const d = directions[k];

      const pa = vtkActor.newInstance();
      pa.setMapper(this.mapperFor(this.protonSphere));
      pa.setPosition(...p);
      this.renderer.addActor(pa);

      const line = vtkLineSource.newInstance({ point1: [...p], point2: this.along(p, d, SHAFT) });
      const la = vtkActor.newInstance();
      la.setMapper(this.mapperFor(line));
      la.getProperty().setLineWidth(3);
      this.renderer.addActor(la);

      const ca = vtkActor.newInstance();
      ca.setMapper(this.mapperFor(this.coneSource));
      ca.setUserMatrix(arrowMatrix(this.along(p, d, HEAD_C), d));
      this.renderer.addActor(ca);

      this.protons.push(pa);
      this.lines.push(line);
      this.shafts.push(la);
      this.heads.push(ca);
      this.applyColor([pa, la, ca], colors?.[k] ?? this.transverseColor(d));
    }
    this.addB0Axis(positions);

    this.renderer.resetCamera();
    const cam = this.renderer.getActiveCamera();
    cam.azimuth(35);
    cam.elevation(-25);
    this.renderer.resetCameraClippingRange();
    this.renderWindow.render();
  }

  updateSpins(directions: Vec3[], colors?: Vec3[]): void {
    const n = Math.min(this.lines.length, directions.length);
    for (let k = 0; k < n; k++) {
      const p = this.positions[k];
      const d = directions[k];
      this.lines[k].setPoint2(this.along(p, d, SHAFT));
      this.heads[k].setUserMatrix(arrowMatrix(this.along(p, d, HEAD_C), d));
      this.applyColor([this.protons[k], this.shafts[k], this.heads[k]], colors?.[k] ?? this.transverseColor(d));
    }
    this.renderWindow.render();
  }

  setSlice(z: number, halfX: number, halfY: number): void {
    const plane = vtkPlaneSource.newInstance({
      origin: [-halfX, -halfY, z],
      point1: [halfX, -halfY, z],
      point2: [-halfX, halfY, z],
    });
    const actor = vtkActor.newInstance();
    actor.setMapper(this.mapperFor(plane));
    const prop = actor.getProperty();
    prop.setColor(0.4, 0.9, 1.0); // RF cyan
    prop.setOpacity(0);
    prop.setLighting(false);
    this.renderer.addActor(actor);
    this.slicePlane = actor;
  }

  flashSlice(opacity: number): void {
    if (this.slicePlane) this.slicePlane.getProperty().setOpacity(opacity);
  }

  private mapperFor(source: any): any {
    const m = vtkMapper.newInstance();
    m.setInputConnection(source.getOutputPort());
    return m;
  }

  private along(p: Vec3, d: Vec3, t: number): Vec3 {
    return [p[0] + d[0] * t, p[1] + d[1] * t, p[2] + d[2] * t];
  }

  private transverseColor(d: Vec3): Vec3 {
    const t = Math.min(1, Math.hypot(d[0], d[1]));
    return [
      REST[0] + (HOT[0] - REST[0]) * t,
      REST[1] + (HOT[1] - REST[1]) * t,
      REST[2] + (HOT[2] - REST[2]) * t,
    ];
  }

  private applyColor(actors: any[], c: Vec3): void {
    for (const a of actors) a.getProperty().setColor(c[0], c[1], c[2]);
  }

  private addB0Axis(positions: Vec3[]): void {
    let minX = Infinity, minZ = Infinity, maxZ = -Infinity;
    for (const p of positions) {
      if (p[0] < minX) minX = p[0];
      if (p[2] < minZ) minZ = p[2];
      if (p[2] > maxZ) maxZ = p[2];
    }
    const x = minX - 1.5;
    const la = vtkActor.newInstance();
    la.setMapper(this.mapperFor(vtkLineSource.newInstance({ point1: [x, 0, minZ - 1], point2: [x, 0, maxZ + 0.6] })));
    la.getProperty().setColor(0.5, 0.5, 0.55);
    la.getProperty().setLineWidth(2);
    this.renderer.addActor(la);
    const ca = vtkActor.newInstance();
    ca.setMapper(this.mapperFor(this.coneSource));
    ca.setUserMatrix(arrowMatrix([x, 0, maxZ + 0.9], [0, 0, 1]));
    ca.getProperty().setColor(0.5, 0.5, 0.55);
    this.renderer.addActor(ca);
  }
}

/** Column-major mat4 placing the shared +X cone at `tip`, rotated so +X → unit(d). */
function arrowMatrix(tip: Vec3, d: Vec3): Float64Array {
  const len = Math.hypot(d[0], d[1], d[2]) || 1;
  const b0 = d[0] / len, b1 = d[1] / len, b2 = d[2] / len;
  let r: number[][];
  if (1 + b0 < 1e-8) {
    r = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]; // d ≈ -X
  } else {
    const f = 1 / (1 + b0);
    r = [
      [b0, -b1, -b2],
      [b1, 1 - b1 * b1 * f, -b1 * b2 * f],
      [b2, -b1 * b2 * f, 1 - b2 * b2 * f],
    ];
  }
  return new Float64Array([
    r[0][0], r[1][0], r[2][0], 0,
    r[0][1], r[1][1], r[2][1], 0,
    r[0][2], r[1][2], r[2][2], 0,
    tip[0], tip[1], tip[2], 1,
  ]);
}
