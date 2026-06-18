import * as ort from 'onnxruntime-web';
import { SIZE, zscore, fitSquare, resizeBilinear, argmaxChannels, resampledSize } from './preprocess';

const CLASSES = 4; // bg, RV, LV-myo, LV-cav

// Serve the wasm runtime from a CDN (avoids bundling the .wasm files through vite).
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/';

/** Runs the exported U-Net in the browser to segment an uploaded volume, matching the pipeline. */
export class Segmenter {
  private session: ort.InferenceSession | null = null;

  async load(url: string): Promise<void> {
    this.session = await ort.InferenceSession.create(url, { executionProviders: ['wasm'] });
  }

  /**
   * Segment a [d,h,w] volume (flat, row-major) given in-plane spacing (y,x mm).
   * Pipeline order: resample in-plane → z-score whole volume → fit_square per slice → run.
   * Returns one [SIZE,SIZE] label mask per slice.
   */
  async segmentVolume(vol: Float32Array, d: number, h: number, w: number, spacingYX: [number, number]): Promise<Uint8Array[]> {
    if (!this.session) throw new Error('model not loaded');
    const nh = resampledSize(h, spacingYX[0]);
    const nw = resampledSize(w, spacingYX[1]);
    const resampled = new Float32Array(d * nh * nw);
    for (let s = 0; s < d; s++) {
      const slice = vol.subarray(s * h * w, (s + 1) * h * w);
      resampled.set(resizeBilinear(slice as Float32Array, h, w, nh, nw), s * nh * nw);
    }
    const z = zscore(resampled);
    const masks: Uint8Array[] = [];
    for (let s = 0; s < d; s++) {
      const sq = fitSquare(z.subarray(s * nh * nw, (s + 1) * nh * nw) as Float32Array, nh, nw);
      masks.push(await this.runSlice(sq));
    }
    return masks;
  }

  private async runSlice(input: Float32Array): Promise<Uint8Array> {
    const t = new ort.Tensor('float32', input, [1, 1, SIZE, SIZE]);
    const out = await this.session!.run({ input: t });
    const logits = out.logits.data as Float32Array; // [1, CLASSES, SIZE, SIZE]
    return argmaxChannels(logits, CLASSES, SIZE * SIZE);
  }
}
