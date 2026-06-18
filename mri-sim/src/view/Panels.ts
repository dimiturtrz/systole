// 2D canvas panels: k-space (magnitude, log-scaled, DC-centered) + reconstructed image.
// Pure DOM/canvas view — no physics.

function fftshift(g: number[][]): number[][] {
  const H = g.length;
  const W = g[0].length;
  const out = g.map((r) => [...r]);
  const hy = Math.floor(H / 2);
  const hx = Math.floor(W / 2);
  const s: number[][] = Array.from({ length: H }, () => new Array<number>(W).fill(0));
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      s[(y + hy) % H][(x + hx) % W] = out[y][x];
    }
  }
  return s;
}

function drawGrid(canvas: HTMLCanvasElement, grid: number[][], logScale: boolean): void {
  const H = grid.length;
  const W = grid[0].length;
  const scale = canvas.width / W;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  let max = 1e-9;
  const vals = grid.map((row) => row.map((v) => (logScale ? Math.log(1 + Math.abs(v)) : Math.abs(v))));
  for (const row of vals) for (const v of row) if (v > max) max = v;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const g = Math.round((vals[y][x] / max) * 255);
      ctx.fillStyle = `rgb(${g},${g},${g})`;
      ctx.fillRect(x * scale, y * scale, scale, scale);
    }
  }
}

export interface Panels {
  drawKspace(mag: number[][]): void;
  drawImage(grid: number[][]): void;
}

export function mountPanels(N: number, px = 5): Panels {
  const wrap = document.createElement('div');
  wrap.style.cssText =
    'position:fixed;left:12px;bottom:12px;z-index:10;display:flex;gap:12px;' +
    'background:rgba(20,24,30,.82);padding:10px;border-radius:8px;border:1px solid #2a323d;' +
    'font:11px system-ui,sans-serif;color:#9aa7b8;';

  const make = (title: string): HTMLCanvasElement => {
    const col = document.createElement('div');
    const label = document.createElement('div');
    label.textContent = title;
    label.style.marginBottom = '4px';
    const c = document.createElement('canvas');
    c.width = N * px;
    c.height = N * px;
    c.style.cssText = 'background:#000;border:1px solid #2a323d;display:block;image-rendering:pixelated;';
    col.appendChild(label);
    col.appendChild(c);
    wrap.appendChild(col);
    return c;
  };

  const kCanvas = make('k-space');
  const imgCanvas = make('reconstructed image');
  document.body.appendChild(wrap);

  return {
    drawKspace: (mag) => drawGrid(kCanvas, fftshift(mag), true),
    drawImage: (grid) => drawGrid(imgCanvas, grid, false),
  };
}
