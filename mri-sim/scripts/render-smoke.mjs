// VISUAL SMOKE (e2e tier): load the running dev server headless, screenshot, and
// ASSERT the spins are actually visible (count teal pixels). Fails (exit 1) if the
// scene is blank or the arrows don't render — the class of bug unit tests can't catch.
//
// Requires `npm run dev` running. Usage: npm run smoke
import puppeteer from 'puppeteer';
import { PNG } from 'pngjs';

const URL = process.env.URL || 'http://localhost:5173/';
const OUT = process.env.OUT || 'debug-shot.png';
const MIN_TEAL = Number(process.env.MIN_TEAL || 300); // spins must paint at least this many px

const browser = await puppeteer.launch({
  headless: 'shell',
  args: [
    '--use-gl=angle',
    '--use-angle=swiftshader',
    '--enable-unsafe-swiftshader',
    '--ignore-gpu-blocklist',
    '--enable-webgl',
    '--no-sandbox',
  ],
});
const page = await browser.newPage();
await page.setViewport({ width: 1000, height: 800 });
const logs = [];
page.on('console', (m) => logs.push(`${m.type()} ${m.text()}`));
page.on('pageerror', (e) => logs.push(`pageerror ${e.message}`));

await page.goto(URL, { waitUntil: 'networkidle0', timeout: 30000 });
await new Promise((r) => setTimeout(r, 1500));
const buf = await page.screenshot({ path: OUT });
await browser.close();

// classify pixels
const png = PNG.sync.read(buf);
let teal = 0;
let nonBackground = 0;
for (let i = 0; i < png.data.length; i += 4) {
  const r = png.data[i], g = png.data[i + 1], b = png.data[i + 2];
  if (r > 35 || g > 40 || b > 45) nonBackground++; // brighter than the ~[15,18,23] bg
  // tealish: green & blue clearly above red (the spin color), reasonably bright
  if (g > 70 && b > 70 && g > r + 10 && b > r + 10) teal++;
}

console.log(`[smoke] non-background px = ${nonBackground}, teal(spin) px = ${teal} (min ${MIN_TEAL})`);
if (logs.some((l) => l.startsWith('pageerror') || l.includes('no webgl'))) {
  console.error('[smoke] page errors:\n' + logs.join('\n'));
}

if (teal < MIN_TEAL) {
  console.error(`[smoke] FAIL: spins not visible (teal px ${teal} < ${MIN_TEAL}). Scene blank or arrows not rendering.`);
  process.exit(1);
}
console.log('[smoke] PASS: spins are visible.');
