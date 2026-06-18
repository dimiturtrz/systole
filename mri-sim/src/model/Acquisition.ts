import { fft2d, ifft2d, type Grid2D } from './fft';

/**
 * Progressive k-space acquisition (pure, testable). The object's full k-space is its 2D
 * Fourier transform; each "readout" reveals one k_y line (the signal the excited slab
 * would produce under that phase-encode). Lines are acquired low→high frequency, so the
 * reconstructed image goes blurry→sharp. When full it can reset and loop.
 */
export class Acquisition {
  readonly N: number;
  private readonly full: Grid2D;
  private readonly order: number[]; // row indices, low frequency first
  private count: number; // how many lines acquired

  constructor(phantom: number[][]) {
    this.N = phantom.length;
    this.full = fft2d(phantom, phantom.map((r) => r.map(() => 0)));
    const fd = (y: number): number => Math.min(y, this.N - y); // distance from DC (wraps)
    this.order = [...Array(this.N).keys()].sort((a, b) => fd(a) - fd(b));
    this.count = 1; // start with the DC line
  }

  get done(): boolean {
    return this.count >= this.N;
  }

  get acquiredLines(): number {
    return this.count;
  }

  acquireNext(): void {
    if (this.count < this.N) this.count++;
  }

  reset(): void {
    this.count = 1;
  }

  /** k-space with only the acquired rows present (others zeroed). */
  maskedKspace(): Grid2D {
    const acq = new Set(this.order.slice(0, this.count));
    return {
      re: this.full.re.map((row, y) => (acq.has(y) ? [...row] : row.map(() => 0))),
      im: this.full.im.map((row, y) => (acq.has(y) ? [...row] : row.map(() => 0))),
    };
  }

  /** Reconstructed (real) image from the currently-acquired k-space. */
  reconstruct(): number[][] {
    const k = this.maskedKspace();
    return ifft2d(k.re, k.im).re;
  }
}
