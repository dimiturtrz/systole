import { wrapIndex } from './util';

/** One cine frame's slices: per-slice grayscale + predicted label mask, each w*h (heart-cropped). */
export interface SliceFrame {
  gray: Float32Array[];
  mask: Uint8Array[];
  w: number;
  h: number;
}

const COLORS: Record<number, [number, number, number]> = {
  1: [91, 141, 239], // RV
  2: [255, 202, 91], // LV-myo
  3: [239, 83, 80], // LV-cav
};
const ALPHA = 0.5;
const DRAG_PX = 14; // vertical pixels per slice step when dragging

/**
 * Single short-axis slice (grayscale MRI + colored prediction overlay), full-screen. Drag up/down
 * scrubs through the slice stack (the "rotation changes the slice" interaction); the heart's own
 * play / ED / ES controls drive the cine in time. Pure canvas2d.
 */
export class SliceView {
  readonly root: HTMLDivElement;
  private readonly canvas: HTMLCanvasElement;
  private readonly off: HTMLCanvasElement;
  private readonly label: HTMLDivElement;
  private readonly msg: HTMLDivElement;

  private frames: SliceFrame[] = [];
  private t = 0;
  private z = 0;
  private timer: number | null = null;

  constructor() {
    this.root = el('div',
      'position:fixed;inset:0;z-index:5;background:#0e1116;display:none;cursor:ns-resize;' +
      'align-items:center;justify-content:center;flex-direction:column;gap:10px;') as HTMLDivElement;
    this.canvas = document.createElement('canvas');
    this.canvas.style.cssText = 'background:#000;border-radius:6px;max-width:96vw;max-height:88vh;';
    this.off = document.createElement('canvas');
    this.label = el('div', 'color:#8aa0b6;font:12px system-ui;user-select:none;') as HTMLDivElement;
    this.msg = el('div', 'color:#8aa0b6;font:14px system-ui;text-align:center;max-width:60vw;display:none;') as HTMLDivElement;
    this.root.append(this.canvas, this.label, this.msg);
    document.body.appendChild(this.root);
    this.installDrag();
  }

  get hasData(): boolean {
    return this.frames.length > 0;
  }
  get isPlaying(): boolean {
    return this.timer !== null;
  }

  /** Load a cine of slices (one SliceFrame per cine frame); opens at startFrame, mid-slice. */
  setSequence(frames: SliceFrame[], startFrame = 0): void {
    this.pause();
    this.msg.style.display = 'none';
    this.canvas.style.display = '';
    this.label.style.display = '';
    this.frames = frames;
    this.t = startFrame;
    this.z = this.midSlice(frames[startFrame] ?? frames[0]);
    this.render();
  }

  show(): void {
    this.root.style.display = 'flex';
    this.render();
  }

  hide(): void {
    this.pause();
    this.root.style.display = 'none';
  }

  setMessage(text: string): void {
    this.pause();
    this.frames = [];
    this.canvas.style.display = 'none';
    this.label.style.display = 'none';
    this.msg.textContent = text;
    this.msg.style.display = 'block';
  }

  /** Cine frame i (time) — driven by the shared play/scrub controls. Keeps the current slice depth. */
  showFrame(i: number): void {
    if (!this.frames.length) return;
    this.t = wrapIndex(i, this.frames.length);
    this.render();
  }

  play(fps: number, onFrame: (i: number) => void): void {
    this.pause();
    if (this.frames.length <= 1) return;
    this.timer = window.setInterval(() => {
      this.t = (this.t + 1) % this.frames.length;
      this.render();
      onFrame(this.t);
    }, 1000 / fps);
  }

  pause(): void {
    if (this.timer !== null) { clearInterval(this.timer); this.timer = null; }
  }

  // ---- drag-to-scrub-depth ----
  private installDrag(): void {
    let startY = 0, startZ = 0, dragging = false;
    this.root.addEventListener('pointerdown', (e) => {
      if (!this.frames.length) return;
      dragging = true; startY = e.clientY; startZ = this.z;
      this.root.setPointerCapture(e.pointerId);
    });
    this.root.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      const d = this.frames[this.t]?.mask.length ?? 1;
      const z = startZ + Math.round((startY - e.clientY) / DRAG_PX); // drag up -> deeper
      this.z = Math.max(0, Math.min(d - 1, z));
      this.render();
    });
    const end = () => { dragging = false; };
    this.root.addEventListener('pointerup', end);
    this.root.addEventListener('pointercancel', end);
  }

  private midSlice(f?: SliceFrame): number {
    if (!f) return 0;
    let best = 0, bestN = -1;
    for (let s = 0; s < f.mask.length; s++) {
      let n = 0;
      for (let i = 0; i < f.mask[s].length; i++) if (f.mask[s][i]) n++;
      if (n > bestN) { bestN = n; best = s; }
    }
    return best;
  }

  private render(): void {
    const f = this.frames[this.t];
    if (!f) return;
    const { w, h } = f;
    const z = Math.min(this.z, f.mask.length - 1);
    const gray = f.gray[z];
    const mask = f.mask[z];
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < gray.length; i++) { if (gray[i] < mn) mn = gray[i]; if (gray[i] > mx) mx = gray[i]; }
    const rng = mx - mn || 1;

    this.off.width = w;
    this.off.height = h;
    const octx = this.off.getContext('2d')!;
    const img = octx.createImageData(w, h);
    const out = img.data;
    for (let i = 0; i < w * h; i++) {
      const g = Math.round(((gray[i] - mn) / rng) * 255);
      let r = g, gg = g, b = g;
      const c = COLORS[mask[i]];
      if (c) {
        r = Math.round(g * (1 - ALPHA) + c[0] * ALPHA);
        gg = Math.round(g * (1 - ALPHA) + c[1] * ALPHA);
        b = Math.round(g * (1 - ALPHA) + c[2] * ALPHA);
      }
      const j = i * 4;
      out[j] = r; out[j + 1] = gg; out[j + 2] = b; out[j + 3] = 255;
    }
    octx.putImageData(img, 0, 0);

    // upscale to fill the screen, keeping aspect (bilinear for a clean look; data unchanged)
    const box = Math.min(window.innerWidth * 0.96, window.innerHeight * 0.88);
    const scale = box / Math.max(w, h);
    this.canvas.width = Math.round(w * scale);
    this.canvas.height = Math.round(h * scale);
    const ctx = this.canvas.getContext('2d')!;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(this.off, 0, 0, this.canvas.width, this.canvas.height);

    const tag = this.frames.length > 1 ? `frame ${this.t + 1}/${this.frames.length}  ·  ` : '';
    this.label.textContent = `${tag}slice ${z + 1}/${f.mask.length}  —  drag to scrub depth`;
  }
}

/** Decode a baked slice strip (PNG: R=grayscale, G=label, D heart-cropped slices stacked) -> SliceFrame.
 *  Strip width = W, height = D*H; `d` slices given by the manifest (sliceD). */
export async function decodeSliceStrip(url: string, d: number): Promise<SliceFrame> {
  const img = await new Promise<HTMLImageElement>((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error(`load ${url}`));
    im.src = url;
  });
  const w = img.width;
  const h = Math.max(1, Math.round(img.height / d));
  const c = document.createElement('canvas');
  c.width = w;
  c.height = img.height;
  const ctx = c.getContext('2d')!;
  ctx.drawImage(img, 0, 0);
  const px = ctx.getImageData(0, 0, w, img.height).data;
  const gray: Float32Array[] = [];
  const mask: Uint8Array[] = [];
  const n = w * h;
  for (let z = 0; z < d; z++) {
    const g = new Float32Array(n);
    const m = new Uint8Array(n);
    const base = z * n * 4;
    for (let i = 0; i < n; i++) {
      g[i] = px[base + i * 4] / 255;
      m[i] = px[base + i * 4 + 1];
    }
    gray.push(g);
    mask.push(m);
  }
  return { gray, mask, w, h };
}

function el(tag: string, css: string): HTMLElement {
  const e = document.createElement(tag);
  e.style.cssText = css;
  return e;
}
