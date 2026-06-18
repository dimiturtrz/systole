import { Presenter } from './presenter/Presenter';
import { SpinScene } from './view/SpinScene';

const presenter = new Presenter(new SpinScene());
presenter.start();
presenter.run();
