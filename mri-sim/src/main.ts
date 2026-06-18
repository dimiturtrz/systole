import { Presenter } from './presenter/Presenter';
import { SpinScene } from './view/SpinScene';
import { mountSpeedSlider } from './view/controls';

const presenter = new Presenter(new SpinScene());
presenter.start();

const DEFAULT_SPEED = 0.35; // slow-mo default (it's quite fast at 1×)
presenter.setSpeed(DEFAULT_SPEED);
mountSpeedSlider(DEFAULT_SPEED, (s) => presenter.setSpeed(s));

presenter.run();
