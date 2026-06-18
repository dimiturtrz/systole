// End-to-end smoke for the IMPORT path: upload a real .nii.gz, run in-browser ONNX
// segmentation, assert the chambers render. Needs a local scan (NOT committed):
//   SCAN=/d/data/raw/mri/acdc/training/patient073/patient073_frame01.nii.gz npm run smoke:import
import puppeteer from 'puppeteer';
import { PNG } from 'pngjs';
import { existsSync } from 'node:fs';

const URL = process.env.URL || 'http://localhost:5173/';
const SCAN = process.env.SCAN;
const OUT = process.env.OUT || 'debug-import.png';
const WAIT = Number(process.env.WAIT || 40000);

if (!SCAN || !existsSync(SCAN)) {
  console.error(`[import-smoke] set SCAN to a local .nii.gz (got: ${SCAN || 'unset'})`);
  process.exit(2);
}

const browser = await puppeteer.launch({
  headless: 'shell',
  args: ['--enable-unsafe-swiftshader', '--use-gl=swiftshader', '--no-sandbox'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1000, height: 800 });
page.on('console', (m) => console.log('[page]', m.text()));
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 30000 });
await new Promise((r) => setTimeout(r, 2000));

const input = await page.$('input[accept*="nii"]');
if (!input) {
  console.error('[import-smoke] scan file input not found');
  await browser.close();
  process.exit(1);
}
await input.uploadFile(SCAN);
await new Promise((r) => setTimeout(r, WAIT)); // model download + wasm inference + mesh + render

const buf = await page.screenshot({ type: 'png' });
const readout = await page.$eval('#cv-readout', (e) => e.textContent?.replace(/\s+/g, ' ').trim()).catch(() => '');
await browser.close();
console.log(`[import-smoke] readout="${readout}"`);

const png = PNG.sync.read(Buffer.from(buf));
let chamber = 0;
for (let i = 0; i < png.data.length; i += 4) {
  const r = png.data[i], g = png.data[i + 1], b = png.data[i + 2];
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  if (max > 80 && max - min > 40) chamber++;
}
const { writeFileSync } = await import('node:fs');
writeFileSync(OUT, PNG.sync.write(png));
console.log(`[import-smoke] chamber px = ${chamber} -> ${OUT}`);
if (chamber < 500) {
  console.error('[import-smoke] FAIL: imported scan did not segment/render.');
  process.exit(1);
}
console.log('[import-smoke] PASS: imported scan segmented + rendered.');
