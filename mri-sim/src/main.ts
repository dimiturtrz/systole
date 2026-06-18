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

// Real MRI timing (T1w spin-echo-ish): TR ≈ 500 ms, TE ≈ 15 ms. Speed 1× = real time;
// the sequence runs far too fast to watch at 1×, so default to slow-motion.
const DEFAULT_SPEED = 0.01;
const ms = (v: number): string => `${(v * 1000).toFixed(v < 0.1 ? 1 : 0)} ms`;
presenter.setSpeed(DEFAULT_SPEED);
presenter.setTR(0.5);
presenter.setTE(0.015);
presenter.setLarmor(63.87); // ≈1.5 T centre frequency → middle slice

mountControls([
  { label: 'Speed', min: 1e-4, max: 1, step: 0.001, value: DEFAULT_SPEED, log: true, fmt: (v) => (v < 0.01 ? `${v.toExponential(1)}×` : `${v.toFixed(2)}×`), onChange: (v) => presenter.setSpeed(v) },
  { label: 'Larmor', min: 63.8, max: 63.95, step: 0.005, value: 63.87, fmt: (v) => `${v.toFixed(3)} MHz`, onChange: (v) => presenter.setLarmor(v) },
  { label: 'TR', min: 0.05, max: 3, step: 0.01, value: 0.5, fmt: ms, onChange: (v) => presenter.setTR(v) },
  { label: 'TE', min: 0.003, max: 0.15, step: 0.001, value: 0.015, fmt: ms, onChange: (v) => presenter.setTE(v) },
  { label: 'Slice angle', min: 0, max: 70, step: 1, value: 0, fmt: (v) => `${v.toFixed(0)}°`, onChange: (v) => presenter.setSliceAngle(v) },
]);

presenter.run();
