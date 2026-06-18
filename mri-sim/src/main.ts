import { Presenter } from './presenter/Presenter';
import { SpinScene } from './view/SpinScene';
import { mountSpeedSlider } from './view/controls';
import { diskPhantom } from './model/phantom';
import { fft2d, ifft2d, magnitudeGrid, zerosLike } from './model/fft';
import { mountPanels } from './view/Panels';

// --- spins (3D precession + slice select) ---
const presenter = new Presenter(new SpinScene());
presenter.start();

const DEFAULT_SPEED = 0.3; // slow-mo default
presenter.setSpeed(DEFAULT_SPEED);
mountSpeedSlider(DEFAULT_SPEED, (s) => presenter.setSpeed(s));

presenter.run();

// --- k-space → image panels: fill k-space line-by-line (low→high freq), image sharpens ---
const N = 24;
const phantom = diskPhantom(N);
const full = fft2d(phantom, zerosLike(phantom)); // the complete k-space of the object
const panels = mountPanels(N);

// acquire rows by frequency distance from DC (wraps): low frequencies first → blurry→sharp
const freqDist = (y: number): number => Math.min(y, N - y);
const rowOrder = [...Array(N).keys()].sort((a, b) => freqDist(a) - freqDist(b));

let step = 0;
let frame = 0;
let hold = 0;

function renderFill(): void {
  const acquired = new Set(rowOrder.slice(0, step + 1));
  const kr = full.re.map((row, y) => (acquired.has(y) ? row : row.map(() => 0)));
  const ki = full.im.map((row, y) => (acquired.has(y) ? row : row.map(() => 0)));
  panels.drawKspace(magnitudeGrid({ re: kr, im: ki }));
  panels.drawImage(ifft2d(kr, ki).re);
}

renderFill();
function fillLoop(): void {
  frame++;
  if (step < N - 1) {
    if (frame % 5 === 0) {
      step++;
      renderFill();
    }
  } else if (++hold > 60) {
    step = 0;
    hold = 0;
    renderFill();
  }
  requestAnimationFrame(fillLoop);
}
requestAnimationFrame(fillLoop);
