import { Presenter } from './presenter/Presenter';
import { SpinScene } from './view/SpinScene';
import { mountSpeedSlider } from './view/controls';
import { diskPhantom } from './model/phantom';
import { mountPanels } from './view/Panels';

const N = 20;
const phantom = diskPhantom(N);
const panels = mountPanels(N);

// Presenter drives the spins AND the k-space acquisition off one speed-scaled clock,
// so the slider controls both the precession and how fast k-space fills.
const presenter = new Presenter(new SpinScene(), panels, phantom);
presenter.start();

const DEFAULT_SPEED = 0.3;
presenter.setSpeed(DEFAULT_SPEED);
mountSpeedSlider(DEFAULT_SPEED, (s) => presenter.setSpeed(s));

presenter.run();
