import '@kitware/vtk.js/Rendering/Profiles/Geometry'; // registers the WebGL backend
import vtkFullScreenRenderWindow from '@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkArrowSource from '@kitware/vtk.js/Filters/Sources/ArrowSource';
import vtkGlyph3DMapper from '@kitware/vtk.js/Rendering/Core/Glyph3DMapper';
import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';
import type { Vec3 } from '../model/types';

/**
 * The VIEW: draws spins as arrow glyphs via vtk.js. No physics here.
 * Renderer is swappable behind this class (could become Three.js later).
 */
export class SpinScene {
  private readonly renderer: any;
  private readonly renderWindow: any;

  constructor() {
    const fs = vtkFullScreenRenderWindow.newInstance({
      background: [0.06, 0.07, 0.09],
    });
    this.renderer = fs.getRenderer();
    this.renderWindow = fs.getRenderWindow();
  }

  /** Draw one arrow per spin, oriented by its magnetization vector. */
  renderSpins(positions: Vec3[], magnetization: Vec3[]): void {
    const n = positions.length;
    const pts = new Float32Array(n * 3);
    const dirs = new Float32Array(n * 3);
    for (let k = 0; k < n; k++) {
      pts[k * 3 + 0] = positions[k][0];
      pts[k * 3 + 1] = positions[k][1];
      pts[k * 3 + 2] = positions[k][2];
      dirs[k * 3 + 0] = magnetization[k][0];
      dirs[k * 3 + 1] = magnetization[k][1];
      dirs[k * 3 + 2] = magnetization[k][2];
    }

    const pd = vtkPolyData.newInstance();
    pd.getPoints().setData(pts, 3);
    pd.getPointData().addArray(
      vtkDataArray.newInstance({ name: 'direction', numberOfComponents: 3, values: dirs }),
    );

    const arrow = vtkArrowSource.newInstance({ tipResolution: 12, shaftResolution: 12 });
    const mapper = vtkGlyph3DMapper.newInstance();
    mapper.setInputData(pd, 0);
    mapper.setSourceConnection(arrow.getOutputPort());
    mapper.setOrientationArray('direction');
    mapper.setOrientationModeToDirection();
    mapper.setScaleFactor(0.8);

    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);
    actor.getProperty().setColor(0.36, 0.82, 0.77);

    this.renderer.addActor(actor);
    this.renderer.resetCamera();
    this.renderWindow.render();
  }
}
