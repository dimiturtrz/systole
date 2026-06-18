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
  it('start renders once and tips ONLY the selected slab (others stay at +z)', () => {
    const view = new FakeView();
    new Presenter(view).start();
    expect(view.renderCalls).toBe(1);
    const M = view.last();
    const tipped = M.filter((m) => Math.abs(m[2]) < 1e-6);
    const resting = M.filter((m) => Math.abs(m[2] - 1) < 1e-6 && Math.hypot(m[0], m[1]) < 1e-6);
    expect(tipped.length).toBeGreaterThan(0); // slab excited
    expect(resting.length).toBeGreaterThan(0); // others untouched
    expect(tipped.length).toBeLessThan(M.length); // not everyone
  });

  it('tick precesses AND relaxes a tipped spin (phase moves, transverse decays, longitudinal recovers)', () => {
    const view = new FakeView();
    const p = new Presenter(view);
    p.start();
    const M0 = view.last();
    const idx = M0.findIndex((m) => Math.abs(m[2]) < 1e-6); // a tipped (slab) spin
    expect(idx).toBeGreaterThanOrEqual(0);
    const before = M0[idx];
    p.tick(0.5);
    const after = view.last()[idx];
    expect(Math.hypot(after[0] - before[0], after[1] - before[1])).toBeGreaterThan(1e-3); // precessed
    expect(Math.hypot(after[0], after[1])).toBeLessThan(Math.hypot(before[0], before[1])); // T2 decay
    expect(after[2]).toBeGreaterThan(before[2]); // T1 recovery
  });

  it('setSpeed scales evolution: 0 freezes, higher moves more', () => {
    const tippedIdx = (v: FakeView) => v.last().findIndex((m) => Math.abs(m[2]) < 1e-6);

    const v0 = new FakeView();
    const p0 = new Presenter(v0);
    p0.start();
    p0.setSpeed(0);
    const i = tippedIdx(v0);
    const b0 = v0.last()[i];
    p0.tick(0.3);
    const a0 = v0.last()[i];
    expect(Math.hypot(a0[0] - b0[0], a0[1] - b0[1])).toBeLessThan(1e-9); // frozen

    const v2 = new FakeView();
    const p2 = new Presenter(v2);
    p2.start();
    p2.setSpeed(2);
    const b2 = v2.last()[i];
    p2.tick(0.3);
    const a2 = v2.last()[i];
    expect(Math.hypot(a2[0] - b2[0], a2[1] - b2[1])).toBeGreaterThan(1e-2); // moved
  });
});
