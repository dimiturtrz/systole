// Largest-connected-component clean-up for predicted masks — pure, unit-tested.
// Mirrors cardioseg/evaluation/postprocess.largest_cc_per_class so the in-browser import
// path matches the pipeline: a 2D-slice model leaves stray false-positive islands that
// barely move Dice but inflate volumes (and EF is a ratio of volumes).

/**
 * Keep only the largest 3D connected component (6-connectivity) of each class label.
 * `masks` is one [w*h] label slice per z; returns a cleaned copy, same shape.
 */
export function largestCcPerClass(
  masks: Uint8Array[],
  w: number,
  h: number,
  labels: number[] = [1, 2, 3],
): Uint8Array[] {
  const d = masks.length;
  const plane = w * h;
  const out: Uint8Array[] = Array.from({ length: d }, () => new Uint8Array(plane));
  const seen = new Uint8Array(d * plane);

  for (const lab of labels) {
    seen.fill(0);
    let best: number[] = [];
    for (let z = 0; z < d; z++) {
      for (let p = 0; p < plane; p++) {
        if (masks[z][p] !== lab || seen[z * plane + p]) continue;
        // flood the component this voxel belongs to (iterative, 6-connected)
        const comp: number[] = [];
        const stack = [z * plane + p];
        seen[z * plane + p] = 1;
        while (stack.length) {
          const cur = stack.pop()!;
          comp.push(cur);
          const cz = (cur / plane) | 0;
          const rem = cur - cz * plane;
          const cy = (rem / w) | 0;
          const cx = rem - cy * w;
          const visit = (nz: number, ny: number, nx: number) => {
            if (nz < 0 || nz >= d || ny < 0 || ny >= h || nx < 0 || nx >= w) return;
            const ni = nz * plane + ny * w + nx;
            if (seen[ni] || masks[nz][ny * w + nx] !== lab) return;
            seen[ni] = 1;
            stack.push(ni);
          };
          visit(cz - 1, cy, cx);
          visit(cz + 1, cy, cx);
          visit(cz, cy - 1, cx);
          visit(cz, cy + 1, cx);
          visit(cz, cy, cx - 1);
          visit(cz, cy, cx + 1);
        }
        if (comp.length > best.length) best = comp;
      }
    }
    for (const idx of best) out[(idx / plane) | 0][idx % plane] = lab;
  }
  return out;
}
