// Visual smoke: load the app headless, assert the colored chamber meshes actually paint.
// Fails (exit 1) if the canvas is blank — catches a broken glb load / importer regression.
import puppeteer from 'puppeteer';
import { PNG } from 'pngjs';

const URL = process.env.URL || 'http://localhost:5173/';
const OUT = process.env.OUT || 'debug-shot.png';
const MIN_CHAMBER = Number(process.env.MIN_CHAMBER || 500);
const WAIT = Number(process.env.WAIT || 4000);

const browser = await puppeteer.launch({
  headless: 'shell',
  args: ['--enable-unsafe-swiftshader', '--use-gl=swiftshader', '--no-sandbox'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1000, height: 800 });
page.on('console', (m) => console.log('[page]', m.text()));
await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 30000 });
await new Promise((r) => setTimeout(r, WAIT));

const buf = await page.screenshot({ type: 'png' });
await browser.close();
const png = PNG.sync.read(Buffer.from(buf));

let chamber = 0;
for (let i = 0; i < png.data.length; i += 4) {
  const r = png.data[i], g = png.data[i + 1], b = png.data[i + 2];
  // chamber colors are saturated reds / blues / golds, clearly off the dark bg
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  if (max > 80 && max - min > 40) chamber++;
}
const { writeFileSync } = await import('node:fs');
writeFileSync(OUT, PNG.sync.write(png));
console.log(`[smoke] chamber-colored px = ${chamber} (min ${MIN_CHAMBER}) -> ${OUT}`);
if (chamber < MIN_CHAMBER) {
  console.error('[smoke] FAIL: chambers not visible (blank scene / glb load broken).');
  process.exit(1);
}
console.log('[smoke] PASS: chambers visible.');
