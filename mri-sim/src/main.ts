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

// --- k-space → image panels (the object's Fourier transform, reconstructed) ---
const N = 24;
const phantom = diskPhantom(N);
const k = fft2d(phantom, zerosLike(phantom));
const image = ifft2d(k.re, k.im); // inverse FFT rebuilds the phantom
const panels = mountPanels(N);
panels.drawKspace(magnitudeGrid(k));
panels.drawImage(image.re);
