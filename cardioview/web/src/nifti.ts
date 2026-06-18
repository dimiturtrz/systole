import * as nifti from 'nifti-reader-js';

export interface Volume {
  data: Float32Array; // [t,d,h,w] row-major (x fastest, t slowest) — t=1 for a 3D scan
  d: number;
  h: number;
  w: number;
  t: number; // cine frames (1 if 3D)
  spacingYX: [number, number]; // (y, x) mm
  zSpacing: number; // mm
}

/** Read a (possibly gzipped) .nii / .nii.gz File into a [t,z,y,x] float volume + spacing. */
export async function readNifti(file: File): Promise<Volume> {
  let buf: ArrayBuffer = await file.arrayBuffer();
  if (nifti.isCompressed(buf)) buf = nifti.decompress(buf) as ArrayBuffer;
  if (!nifti.isNIFTI(buf)) throw new Error('not a NIfTI file');
  const hdr = nifti.readHeader(buf);
  const raw = nifti.readImage(hdr, buf);
  const [ndim, nx, ny, nz, nt] = hdr.dims; // dims[0]=ndim
  const [, sx, sy, sz] = hdr.pixDims;
  return {
    data: toFloat32(raw, hdr.datatypeCode),
    d: nz,
    h: ny,
    w: nx,
    t: ndim >= 4 ? nt : 1,
    spacingYX: [sy, sx],
    zSpacing: sz,
  };
}

/** One [z,y,x] frame of a cine (i in [0,t)). */
export function frame(v: Volume, i: number): Float32Array {
  const n = v.d * v.h * v.w;
  return v.data.subarray(i * n, (i + 1) * n) as Float32Array;
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
