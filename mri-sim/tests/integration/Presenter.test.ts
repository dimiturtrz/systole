import { describe, it, expect } from 'vitest';
import { Presenter } from '../../src/presenter/Presenter';
import type { SpinView } from '../../src/view/SpinView';
import type { Vec3 } from '../../src/model/types';

// INTEGRATION: presenter + simulator + a fake view (no vtk/WebGL).
class FakeView implements SpinView {
  renderCalls = 0;
  updates: Vec3[][] = [];
  renderSpins(_positions: Vec3[], _magnetization: Vec3[]): void {
    this.renderCalls++;
  }
  updateSpins(magnetization: Vec3[]): void {
    this.updates.push(magnetization.map((v) => [...v] as Vec3));
  }
  last(): Vec3[] {
    return this.updates[this.updates.length - 1];
  }
}

describe('Presenter (presenter + simulator + mock view)', () => {
  it('start renders once and tips every spin into the transverse plane', () => {
    const view = new FakeView();
    new Presenter(view).start();
    expect(view.renderCalls).toBe(1);
    for (const m of view.last()) expect(Math.abs(m[2])).toBeLessThan(1e-6);
  });

  it('tick precesses AND relaxes (phase moves, transverse decays, longitudinal recovers)', () => {
    const view = new FakeView();
    const p = new Presenter(view);
    p.start();
    const before = view.last()[0]; // just-tipped: transverse, Mz≈0
    p.tick(0.5);
    const after = view.last()[0];
    const moved = Math.hypot(after[0] - before[0], after[1] - before[1]);
    expect(moved).toBeGreaterThan(1e-3); // precessed
    expect(Math.hypot(after[0], after[1])).toBeLessThan(Math.hypot(before[0], before[1])); // T2 decay
    expect(after[2]).toBeGreaterThan(before[2]); // T1 recovery toward +z
  });
});
