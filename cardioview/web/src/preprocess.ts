// In-browser mirror of the cardioseg preprocessing, so a model run in the browser matches
// the Python pipeline: resample in-plane to 1.5 mm, z-score the whole volume, center
// pad/crop each slice to 256. Pure (no DOM/onnx) — unit-tested.

export const SIZE = 256;
export const TARGET_MM = 1.5;

/** Per-volume z-score (population std + eps) — matches cardioseg.preprocessing.zscore. */
export function zscore(vol: Float32Array): Float32Array {
  let mean = 0;
  for (let i = 0; i < vol.length; i++) mean += vol[i];
  mean /= vol.length;
  let varsum = 0;
  for (let i = 0; i < vol.length; i++) varsum += (vol[i] - mean) ** 2;
  const std = Math.sqrt(varsum / vol.length) + 1e-6;
  const out = new Float32Array(vol.length);
  for (let i = 0; i < vol.length; i++) out[i] = (vol[i] - mean) / std;
  return out;
}

/** Center pad/crop a [h,w] slice to size×size — matches cardioseg.training.dataset.fit_square. */
export function fitSquare(slice: Float32Array, h: number, w: number, size = SIZE, pad = 0): Float32Array {
  const out = new Float32Array(size * size).fill(pad);
  const sh = Math.max(0, (h - size) >> 1);
  const sw = Math.max(0, (w - size) >> 1);
  const ch = Math.min(h, size);
  const cw = Math.min(w, size);
  const dh = (size - ch) >> 1;
  const dw = (size - cw) >> 1;
  for (let r = 0; r < ch; r++)
    for (let c = 0; c < cw; c++) out[(dh + r) * size + dw + c] = slice[(sh + r) * w + sw + c];
  return out;
}

/** Bilinear resize a [h,w] slice to [nh,nw] (resample in-plane to the target spacing). */
export function resizeBilinear(slice: Float32Array, h: number, w: number, nh: number, nw: number): Float32Array {
  const out = new Float32Array(nh * nw);
  const ry = h / nh;
  const rx = w / nw;
  for (let i = 0; i < nh; i++) {
    const fy = Math.min(h - 1, Math.max(0, (i + 0.5) * ry - 0.5));
    const y0 = Math.floor(fy);
    const y1 = Math.min(h - 1, y0 + 1);
    const wy = fy - y0;
    for (let j = 0; j < nw; j++) {
      const fx = Math.min(w - 1, Math.max(0, (j + 0.5) * rx - 0.5));
      const x0 = Math.floor(fx);
      const x1 = Math.min(w - 1, x0 + 1);
      const wx = fx - x0;
      const a = slice[y0 * w + x0], b = slice[y0 * w + x1];
      const c = slice[y1 * w + x0], d = slice[y1 * w + x1];
      out[i * nw + j] = a * (1 - wy) * (1 - wx) + b * (1 - wy) * wx + c * wy * (1 - wx) + d * wy * wx;
    }
  }
  return out;
}

/** Per-pixel argmax over the first `classes` channels of a [classes,hw] logits buffer. */
export function argmaxChannels(logits: Float32Array, classes: number, hw: number): Uint8Array {
  const out = new Uint8Array(hw);
  for (let p = 0; p < hw; p++) {
    let best = 0;
    let bestVal = logits[p];
    for (let c = 1; c < classes; c++) {
      const v = logits[c * hw + p];
      if (v > bestVal) {
        bestVal = v;
        best = c;
      }
    }
    out[p] = best;
  }
  return out;
}

/** Target in-plane size after resampling spacing (mm) to TARGET_MM. */
export function resampledSize(n: number, spacingMm: number): number {
  return Math.round((n * spacingMm) / TARGET_MM);
}
