import * as nifti from 'nifti-reader-js';

export interface Volume {
  data: Float32Array; // [d,h,w] row-major (z,y,x) — same layout NIfTI stores (x fastest)
  d: number;
  h: number;
  w: number;
  spacingYX: [number, number]; // (y, x) mm
  zSpacing: number; // mm
}

/** Read a (possibly gzipped) .nii / .nii.gz File into a [z,y,x] float volume + spacing. */
export async function readNifti(file: File): Promise<Volume> {
  let buf: ArrayBuffer = await file.arrayBuffer();
  if (nifti.isCompressed(buf)) buf = nifti.decompress(buf) as ArrayBuffer;
  if (!nifti.isNIFTI(buf)) throw new Error('not a NIfTI file');
  const hdr = nifti.readHeader(buf);
  const raw = nifti.readImage(hdr, buf);
  const [, nx, ny, nz] = hdr.dims; // dims[0]=ndim
  const [, sx, sy, sz] = hdr.pixDims;
  return {
    data: toFloat32(raw, hdr.datatypeCode),
    d: nz,
    h: ny,
    w: nx,
    spacingYX: [sy, sx],
    zSpacing: sz,
  };
}

function toFloat32(buf: ArrayBuffer, code: number): Float32Array {
  // NIfTI datatype codes -> typed view, then widen to Float32.
  const view =
    code === nifti.NIFTI1.TYPE_UINT8 ? new Uint8Array(buf) :
    code === nifti.NIFTI1.TYPE_INT16 ? new Int16Array(buf) :
    code === nifti.NIFTI1.TYPE_UINT16 ? new Uint16Array(buf) :
    code === nifti.NIFTI1.TYPE_INT32 ? new Int32Array(buf) :
    code === nifti.NIFTI1.TYPE_FLOAT32 ? new Float32Array(buf) :
    code === nifti.NIFTI1.TYPE_FLOAT64 ? new Float64Array(buf) :
    new Int16Array(buf); // ACDC is typically int16
  return view instanceof Float32Array ? view : Float32Array.from(view);
}
