import '@kitware/vtk.js/Rendering/Profiles/Geometry'; // registers the WebGL backend
import vtkFullScreenRenderWindow from '@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';
import vtkLineSource from '@kitware/vtk.js/Filters/Sources/LineSource';
import vtkConeSource from '@kitware/vtk.js/Filters/Sources/ConeSource';
import vtkPlaneSource from '@kitware/vtk.js/Filters/Sources/PlaneSource';
import type { Vec3 } from '../model/types';
import type { SpinView } from './SpinView';
import { transverseColor, SLICE_PLANE, B0_AXIS } from './palette';

const SHAFT = 0.7;
const HEAD = 0.35;
const HEAD_R = 0.13;
const K = 6; // arrowhead ring segments
const PPS = 3 + K; // points per spin: base, shaftEnd, apex, K ring

/**
 * The VIEW — BATCHED: every spin (proton dot + line shaft + cone head) is merged into
 * ONE vtkPolyData with per-point RGBA scalars. One actor, one buffer upload per frame,
 * GPU coloring — scales to thousands of spins. (Was ~3 actors/spin.)
 */
export class SpinScene implements SpinView {
  private readonly renderer: any;
  private readonly renderWindow: any;
  private readonly poly: any;
  private readonly scalars: any;
  private positions: Vec3[] = [];
  private pts!: Float32Array;
  private cols!: Uint8Array;
  private slicePlane: any;

  constructor() {
    const fs = vtkFullScreenRenderWindow.newInstance({ background: [0.06, 0.07, 0.09] });
    this.renderer = fs.getRenderer();
    this.renderWindow = fs.getRenderWindow();
    this.poly = vtkPolyData.newInstance();
    this.scalars = vtkDataArray.newInstance({ name: 'colors', numberOfComponents: 4, values: new Uint8Array(0) });
  }

  renderSpins(positions: Vec3[], directions: Vec3[], colors?: Vec3[]): void {
    this.positions = positions;
    const N = positions.length;
    this.pts = new Float32Array(N * PPS * 3);
    this.cols = new Uint8Array(N * PPS * 4);

    // static topology (indices fixed; only point coords + colors change per frame)
    const verts = new Uint32Array(N * 2); // proton dot at base
    const lines = new Uint32Array(N * 3); // shaft
    const polys = new Uint32Array(N * K * 4); // head triangles
    for (let s = 0; s < N; s++) {
      const o = s * PPS;
      verts[s * 2] = 1;
      verts[s * 2 + 1] = o; // base
      lines[s * 3] = 2;
      lines[s * 3 + 1] = o; // base
      lines[s * 3 + 2] = o + 1; // shaftEnd
      for (let i = 0; i < K; i++) {
        const t = s * K * 4 + i * 4;
        polys[t] = 3;
        polys[t + 1] = o + 2; // apex
        polys[t + 2] = o + 3 + i; // ring i
        polys[t + 3] = o + 3 + ((i + 1) % K); // ring i+1
      }
    }
    this.poly.getPoints().setData(this.pts, 3);
    this.poly.getVerts().setData(verts);
    this.poly.getLines().setData(lines);
    this.poly.getPolys().setData(polys);
    this.poly.getPointData().setScalars(this.scalars);

    const mapper = vtkMapper.newInstance();
    mapper.setInputData(this.poly);
    mapper.setScalarVisibility(true);
    if (mapper.setColorModeToDirectScalars) mapper.setColorModeToDirectScalars();
    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);
    actor.getProperty().setLineWidth(2);
    actor.getProperty().setPointSize(7);
    this.renderer.addActor(actor);

    this.writeFrame(directions, colors);
    this.addB0Axis(positions);

    this.renderer.resetCamera();
    const cam = this.renderer.getActiveCamera();
    cam.azimuth(35);
    cam.elevation(-25);
    this.renderer.resetCameraClippingRange();
    this.renderWindow.render();
  }

  updateSpins(directions: Vec3[], colors?: Vec3[]): void {
    this.writeFrame(directions, colors);
    this.poly.getPoints().setData(this.pts, 3);
    this.scalars.setData(this.cols, 4);
    this.poly.modified();
    this.renderWindow.render();
  }

  /** Fill the merged point + color buffers for the current directions/colors. */
  private writeFrame(directions: Vec3[], colors?: Vec3[]): void {
    const N = Math.min(this.positions.length, directions.length);
    for (let s = 0; s < N; s++) {
      const p = this.positions[s];
      const d = directions[s];
      const [u, v] = perpBasis(d);
      const base = p;
      const shaftEnd: Vec3 = [p[0] + d[0] * SHAFT, p[1] + d[1] * SHAFT, p[2] + d[2] * SHAFT];
      const apex: Vec3 = [p[0] + d[0] * (SHAFT + HEAD), p[1] + d[1] * (SHAFT + HEAD), p[2] + d[2] * (SHAFT + HEAD)];
      const o = s * PPS;
      this.setPt(o, base);
      this.setPt(o + 1, shaftEnd);
      this.setPt(o + 2, apex);
      for (let i = 0; i < K; i++) {
        const a = (i / K) * Math.PI * 2;
        const c = Math.cos(a) * HEAD_R;
        const sn = Math.sin(a) * HEAD_R;
        this.setPt(o + 3 + i, [
          shaftEnd[0] + c * u[0] + sn * v[0],
          shaftEnd[1] + c * u[1] + sn * v[1],
          shaftEnd[2] + c * u[2] + sn * v[2],
        ]);
      }
      const col = colors?.[s] ?? transverseColor(d);
      const r = (col[0] * 255) | 0, g = (col[1] * 255) | 0, b = (col[2] * 255) | 0;
      for (let i = 0; i < PPS; i++) {
        const ci = (o + i) * 4;
        this.cols[ci] = r;
        this.cols[ci + 1] = g;
        this.cols[ci + 2] = b;
        this.cols[ci + 3] = 255;
      }
    }
  }

  private setPt(i: number, p: Vec3): void {
    this.pts[i * 3] = p[0];
    this.pts[i * 3 + 1] = p[1];
    this.pts[i * 3 + 2] = p[2];
  }

  setSlice(center: Vec3, uHalf: Vec3, vHalf: Vec3): void {
    if (this.slicePlane) this.renderer.removeActor(this.slicePlane);
    const corner = (su: number, sv: number): Vec3 => [
      center[0] + su * uHalf[0] + sv * vHalf[0],
      center[1] + su * uHalf[1] + sv * vHalf[1],
      center[2] + su * uHalf[2] + sv * vHalf[2],
    ];
    const plane = vtkPlaneSource.newInstance({ origin: corner(-1, -1), point1: corner(1, -1), point2: corner(-1, 1) });
    const m = vtkMapper.newInstance();
    m.setInputConnection(plane.getOutputPort());
    const actor = vtkActor.newInstance();
    actor.setMapper(m);
    const prop = actor.getProperty();
    prop.setColor(...SLICE_PLANE);
    prop.setOpacity(0);
    prop.setLighting(false);
    this.renderer.addActor(actor);
    this.slicePlane = actor;
  }

  flashSlice(opacity: number): void {
    if (this.slicePlane) this.slicePlane.getProperty().setOpacity(opacity);
  }

  private addB0Axis(positions: Vec3[]): void {
    let minX = Infinity, minZ = Infinity, maxZ = -Infinity;
    for (const p of positions) {
      if (p[0] < minX) minX = p[0];
      if (p[2] < minZ) minZ = p[2];
      if (p[2] > maxZ) maxZ = p[2];
    }
    const x = minX - 1.5;
    const lm = vtkMapper.newInstance();
    lm.setInputConnection(vtkLineSource.newInstance({ point1: [x, 0, minZ - 1], point2: [x, 0, maxZ + 0.6] }).getOutputPort());
    const la = vtkActor.newInstance();
    la.setMapper(lm);
    la.getProperty().setColor(...B0_AXIS);
    la.getProperty().setLineWidth(2);
    this.renderer.addActor(la);
    const cm = vtkMapper.newInstance();
    cm.setInputConnection(vtkConeSource.newInstance({ height: 0.6, radius: 0.2, resolution: 16, center: [x, 0, maxZ + 0.9], direction: [0, 0, 1] }).getOutputPort());
    const ca = vtkActor.newInstance();
    ca.setMapper(cm);
    ca.getProperty().setColor(...B0_AXIS);
    this.renderer.addActor(ca);
  }
}

/** Two unit vectors perpendicular to unit `d` (and each other). */
function perpBasis(d: Vec3): [Vec3, Vec3] {
  const ax: Vec3 = Math.abs(d[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  const u = norm([ax[1] * d[2] - ax[2] * d[1], ax[2] * d[0] - ax[0] * d[2], ax[0] * d[1] - ax[1] * d[0]]);
  const v: Vec3 = [d[1] * u[2] - d[2] * u[1], d[2] * u[0] - d[0] * u[2], d[0] * u[1] - d[1] * u[0]];
  return [u, v];
}

function norm(a: Vec3): Vec3 {
  const m = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / m, a[1] / m, a[2] / m];
}
