import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';
import vtkImageMarchingCubes from '@kitware/vtk.js/Filters/General/ImageMarchingCubes';
import { SIZE, TARGET_MM } from './preprocess';

// Chamber label per viewer slot (matches chambers.ts: 0=LV cav, 1=myo, 2=RV).
const SLOT_LABELS = [3, 2, 1];

/** Marching-cubes chamber surfaces from a stack of [SIZE,SIZE] label masks (in-browser meshing). */
export function chamberPolys(masks: Uint8Array[], zSpacing: number): (any | null)[] {
  const d = masks.length;
  const out: (any | null)[] = [null, null, null];
  for (let slot = 0; slot < SLOT_LABELS.length; slot++) {
    const label = SLOT_LABELS[slot];
    const bin = new Uint8Array(d * SIZE * SIZE);
    let any = false;
    for (let z = 0; z < d; z++) {
      const m = masks[z];
      for (let i = 0; i < m.length; i++) {
        if (m[i] === label) {
          bin[z * SIZE * SIZE + i] = 1;
          any = true;
        }
      }
    }
    if (!any) continue;
    const img: any = vtkImageData.newInstance();
    img.setDimensions(SIZE, SIZE, d);
    img.setSpacing([TARGET_MM, TARGET_MM, zSpacing]); // in-plane resampled to 1.5 mm; z as acquired
    img.getPointData().setScalars(vtkDataArray.newInstance({ numberOfComponents: 1, values: bin }));
    const mc = vtkImageMarchingCubes.newInstance({ contourValue: 0.5, computeNormals: true });
    mc.setInputData(img);
    const poly = mc.getOutputData();
    out[slot] = poly.getNumberOfPoints() > 0 ? poly : null;
  }
  return out;
}
