import '@kitware/vtk.js/Rendering/Profiles/Geometry'; // WebGL backend
import vtkFullScreenRenderWindow from '@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow';
import vtkGLTFImporter from '@kitware/vtk.js/IO/Geometry/GLTFImporter';

/**
 * Renders precomputed chamber-mesh glb scenes (one per patient/phase) with a trackball
 * camera. Pure view — geometry + colors come baked into the glb from the Python export.
 */
export class HeartViewer {
  private readonly renderer: any;
  private readonly renderWindow: any;
  private actors: any[] = [];
  private token = 0; // guards against out-of-order async loads (fast phase toggling)

  constructor() {
    const fs = vtkFullScreenRenderWindow.newInstance({ background: [0.055, 0.066, 0.086] });
    this.renderer = fs.getRenderer();
    this.renderWindow = fs.getRenderWindow();
    // Correct blending for the translucent myocardium over the chambers inside it.
    this.renderer.setUseDepthPeeling?.(true);
    this.renderer.setOcclusionRatio?.(0.1);
    this.renderer.setMaximumNumberOfPeels?.(4);
  }

  /** Load a glb, replacing whatever is shown. Keeps the camera once one is set. */
  async load(url: string, keepCamera = false): Promise<void> {
    const my = ++this.token;
    const importer = vtkGLTFImporter.newInstance();
    importer.setRenderer(this.renderer);
    await importer.setUrl(url, { binary: true });
    await importer.loadData();
    if (my !== this.token) return; // a newer load superseded this one
    this.clear();
    importer.importActors();
    this.actors = Array.from(importer.getActors().values());
    // glTF drops per-mesh opacity → the gold myocardium imports opaque and hides the
    // cavities inside. Re-apply translucency to it (detected by its gold color).
    for (const a of this.actors) {
      const [r, g, b] = a.getProperty().getColor();
      if (r > 0.6 && g > 0.55 && b < 0.55) a.getProperty().setOpacity(0.32); // myocardium
    }
    if (!keepCamera) this.renderer.resetCamera();
    this.renderWindow.render();
  }

  private clear(): void {
    for (const a of this.actors) this.renderer.removeActor(a);
    this.actors = [];
  }
}
