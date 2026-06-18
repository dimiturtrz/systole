import { Presenter } from './presenter/Presenter';
import { SpinScene } from './view/SpinScene';
import { mountControls } from './view/controls';
import { diskPhantom } from './model/phantom';
import { mountPanels } from './view/Panels';
import { mountSequenceDiagram } from './view/SequenceDiagram';

const N = 20;
const phantom = diskPhantom(N);
const panels = mountPanels(N);
const seq = mountSequenceDiagram();

// One presenter drives spins + k-space acquisition + sequence diagram on a shared clock.
const presenter = new Presenter(new SpinScene(), panels, phantom, seq);
presenter.start();

const DEFAULT_SPEED = 0.3;
presenter.setSpeed(DEFAULT_SPEED);
presenter.setTR(2.0);
presenter.setTE(0.5);

mountControls([
  { label: 'Speed', min: 0.05, max: 2, step: 0.05, value: DEFAULT_SPEED, fmt: (v) => `${v.toFixed(2)}×`, onChange: (v) => presenter.setSpeed(v) },
  { label: 'Larmor', min: 0.1, max: 2, step: 0.05, value: 0.5, fmt: (v) => `${v.toFixed(2)} Hz`, onChange: (v) => presenter.setLarmor(v) },
  { label: 'TR (s)', min: 0.5, max: 5, step: 0.1, value: 2.0, fmt: (v) => v.toFixed(1), onChange: (v) => presenter.setTR(v) },
  { label: 'TE (s)', min: 0.1, max: 4, step: 0.05, value: 0.5, fmt: (v) => v.toFixed(2), onChange: (v) => presenter.setTE(v) },
]);

presenter.run();
