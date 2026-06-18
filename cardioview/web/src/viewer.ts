import '@kitware/vtk.js/Rendering/Profiles/Geometry'; // WebGL backend
import vtkFullScreenRenderWindow from '@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow';
import vtkGLTFImporter from '@kitware/vtk.js/IO/Geometry/GLTFImporter';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkCubeAxesActor from '@kitware/vtk.js/Rendering/Core/CubeAxesActor';

// Three chamber slots, identified by the glb's baked color. glTF drops opacity → re-set it.
const CHAMBERS = [
  { name: 'LV cavity', color: [0.94, 0.33, 0.31], opacity: 1.0, test: (r: number, g: number, b: number) => r > 0.6 && g < 0.5 && b < 0.5 },
  { name: 'myocardium', color: [1.0, 0.79, 0.36], opacity: 0.32, test: (r: number, g: number, b: number) => r > 0.6 && g > 0.55 && b < 0.55 },
  { name: 'RV cavity', color: [0.36, 0.56, 0.93], opacity: 1.0, test: (r: number, _g: number, b: number) => b > 0.55 && b > r },
];

/**
 * Renders chamber meshes from precomputed glb. The beating cycle keeps only THREE actors
 * (one per chamber) and swaps their polydata per frame — not one actor per frame, which
 * exhausts WebGL. Frame swaps are an input + render, so playback is smooth.
 */
export class HeartViewer {
  private readonly renderer: any;
  private readonly renderWindow: any;
  private actors: any[] = []; // 3 persistent chamber actors
  private framePolys: (any | null)[][] = []; // [frame][slot] polydata
  private timer: number | null = null;
  private token = 0;
  private cameraSet = false;

  constructor() {
    const fs = vtkFullScreenRenderWindow.newInstance({ background: [1, 1, 1] });
    this.renderer = fs.getRenderer();
    this.renderWindow = fs.getRenderWindow();
    this.renderer.setUseDepthPeeling?.(true);
    this.renderer.setOcclusionRatio?.(0.1);
    this.renderer.setMaximumNumberOfPeels?.(4);
    // Coordinate box with mm tick labels (meshes are in mm) — read scale / cm off it.
    this.cubeAxes = vtkCubeAxesActor.newInstance();
    this.cubeAxes.setCamera(this.renderer.getActiveCamera());
    this.cubeAxes.getProperty().setColor(0.1, 0.12, 0.16); // dark on white
    this.cubeAxes.setGridLines(false);
    this.cubeAxes.setAxisLabels?.(['X (mm)', 'Y (mm)', 'Z (mm)']); // ticks are in mm
    this.cubeAxes.setVisibility(false); // until bounds are known
    this.renderer.addActor(this.cubeAxes);
  }

  private cubeAxes: any;

  private updateAxes(): void {
    const b = [Infinity, -Infinity, Infinity, -Infinity, Infinity, -Infinity];
    for (const a of this.actors) {
      if (!a.getVisibility()) continue;
      const ab = a.getBounds();
      for (let i = 0; i < 6; i += 2) {
        b[i] = Math.min(b[i], ab[i]);
        b[i + 1] = Math.max(b[i + 1], ab[i + 1]);
      }
    }
    if (b[0] < b[1]) {
      this.cubeAxes.setDataBounds(b);
      this.cubeAxes.setVisibility(true);
    }
  }

  /** Single static scene. */
  async load(url: string): Promise<void> {
    await this.loadSequence([url]);
  }

  /** A cycle of glb frames; extract per-chamber geometry, show frame 0. */
  async loadSequence(urls: string[]): Promise<number> {
    const my = ++this.token;
    // Sequential: each import briefly adds its actors (to realize geometry), we copy the
    // polydata out, then remove them — so only ~3 transient actors exist at a time (no
    // WebGL exhaustion). Parallel would stack all frames' actors at once and lose context.
    const collected: (any | null)[][] = [];
    for (const url of urls) {
      const polys = await this.extractChambers(url);
      if (my !== this.token) return 0;
      collected.push(polys);
    }
    this.pause();
    this.ensureActors();
    this.framePolys = collected;
    this.showFrame(0);
    if (!this.cameraSet) {
      this.renderer.resetCamera();
      this.cameraSet = true;
    }
    this.updateAxes();
    this.renderWindow.render();
    return collected.length;
  }

  showFrame(i: number): void {
    if (this.framePolys.length === 0) return;
    const n = this.framePolys.length;
    const frame = this.framePolys[((i % n) + n) % n];
    for (let s = 0; s < this.actors.length; s++) {
      const pd = frame[s];
      if (pd) {
        this.actors[s].getMapper().setInputData(pd);
        this.actors[s].setVisibility(true);
      } else {
        this.actors[s].setVisibility(false);
      }
    }
    this.renderWindow.render();
  }

  play(fps: number, onFrame: (i: number) => void): void {
    this.pause();
    const n = this.framePolys.length;
    if (n <= 1) return;
    let i = 0;
    this.timer = window.setInterval(() => {
      i = (i + 1) % n;
      this.showFrame(i);
      onFrame(i);
    }, 1000 / fps);
  }

  pause(): void {
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  get isPlaying(): boolean {
    return this.timer !== null;
  }

  /** Import a glb, pull each chamber's polydata by color, drop the imported actors. */
  private async extractChambers(url: string): Promise<(any | null)[]> {
    const importer = vtkGLTFImporter.newInstance();
    importer.setRenderer(this.renderer);
    try {
      // loadData resolves on FETCH, but parse (which builds actors) runs after, in an
      // un-awaited .then — so wait for onReady, else importActors races ahead of parse.
      // loadData resolves on FETCH; parse (builds actors) runs after in an un-awaited .then —
      // wait for onReady. importActors() realizes the mapper geometry (getInputData is null
      // until then); we copy the polydata out and remove the actor (keep only our 3).
      await new Promise<void>((resolve, reject) => {
        importer.onReady(resolve);
        importer.setUrl(url, { binary: true }).catch(reject);
      });
      importer.importActors();
    } catch (e: any) {
      console.error('[viewer] import failed', url, e?.message, e?.stack);
      return [null, null, null];
    }
    const polys: (any | null)[] = [null, null, null];
    for (const a of importer.getActors().values()) {
      const [r, g, b] = a.getProperty().getColor();
      const slot = CHAMBERS.findIndex((c) => c.test(r, g, b));
      const mapper = a.getMapper();
      if (slot >= 0 && mapper) polys[slot] = mapper.getInputData();
      this.renderer.removeActor(a);
    }
    return polys;
  }

  private ensureActors(): void {
    if (this.actors.length) return;
    for (const c of CHAMBERS) {
      const actor = vtkActor.newInstance();
      actor.setMapper(vtkMapper.newInstance());
      actor.getProperty().setColor(c.color[0], c.color[1], c.color[2]);
      actor.getProperty().setOpacity(c.opacity);
      actor.getProperty().setInterpolationToPhong();
      this.renderer.addActor(actor);
      this.actors.push(actor);
    }
  }
}
