// Headless-browser probe: load the dev server, capture console + a screenshot.
// Lets the dev verify the WebGL render without a human in the loop.
import puppeteer from 'puppeteer';

const URL = process.env.URL || 'http://localhost:5173/';
const OUT = process.env.OUT || 'debug-shot.png';

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
page.on('console', (m) => console.log('PAGE:', m.type(), m.text()));
page.on('pageerror', (e) => console.log('PAGEERROR:', e.message));

await page.goto(URL, { waitUntil: 'networkidle0', timeout: 30000 });
await new Promise((r) => setTimeout(r, 2000));
await page.screenshot({ path: OUT });
await browser.close();
console.log('SHOT_SAVED', OUT);
