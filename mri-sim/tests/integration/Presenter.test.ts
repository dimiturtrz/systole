import { describe, it, expect } from 'vitest';
import { Presenter } from '../../src/presenter/Presenter';
import type { SpinView } from '../../src/view/SpinView';
import type { Panels } from '../../src/view/Panels';
import { diskPhantom } from '../../src/model/phantom';
import type { Vec3 } from '../../src/model/types';

const REST_Z = Math.cos(0.12); // Presenter's rest tilt → resting proton z-component

class FakeView implements SpinView {
  renderCalls = 0;
  updates: Vec3[][] = [];
  lastColors?: Vec3[];
  renderSpins(_p: Vec3[], _m: Vec3[]): void {
    this.renderCalls++;
  }
  updateSpins(m: Vec3[], colors?: Vec3[]): void {
    this.updates.push(m.map((v) => [...v] as Vec3));
    this.lastColors = colors;
  }
  setSlice(): void {}
  flashSlice(): void {}
  last(): Vec3[] {
    return this.updates[this.updates.length - 1];
  }
}

describe('Presenter (proton view: presenter + simulator + mock view)', () => {
  it('the RF tip ramp brings ONLY the slab to transverse; rest keeps a tiny tilt', () => {
    const v = new FakeView();
    const p = new Presenter(v);
    p.start();
    expect(v.renderCalls).toBe(1);
    p.tick(0.2); // complete the smooth tip ramp (TIP_DUR = 0.15s)
    const M = v.last();
    const tipped = M.filter((m) => Math.abs(m[2]) < 1e-6); // transverse slab
    const resting = M.filter((m) => Math.abs(m[2] - REST_Z) < 1e-9); // near +z
    expect(tipped.length).toBeGreaterThan(0);
    expect(resting.length).toBeGreaterThan(0);
    expect(tipped.length).toBeLessThan(M.length);
  });

  it('tick precesses every proton, including resting ones (transverse direction rotates)', () => {
    const v = new FakeView();
    const p = new Presenter(v);
    p.start();
    const M0 = v.last();
    const ri = M0.findIndex((m) => Math.abs(m[2] - REST_Z) < 1e-9); // a resting proton
    expect(ri).toBeGreaterThanOrEqual(0);
    const before = M0[ri];
    p.tick(0.5);
    const after = v.last()[ri];
    expect(Math.hypot(after[0] - before[0], after[1] - before[1])).toBeGreaterThan(1e-3);
  });

  it('setSpeed scales evolution: 0 freezes, higher moves more', () => {
    const v0 = new FakeView();
    const p0 = new Presenter(v0);
    p0.start();
    p0.setSpeed(0);
    const i = v0.last().findIndex((m) => Math.abs(m[2] - REST_Z) < 1e-9);
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

  it('colors spins by Larmor while a gradient is on, none when idle', () => {
    const v = new FakeView();
    const p = new Presenter(v);
    p.start(); // cycleTime 0 → slice-select gradient on
    expect(v.lastColors).toBeDefined();
    expect(v.lastColors!.length).toBeGreaterThan(0);
    p.tick(0.36); // ct well past the readout window (te≈15 ms) → relaxation/wait, no gradient
    expect(v.lastColors).toBeUndefined();
  });

  it('setSliceAngle selects a different (oblique) slab', () => {
    const tippedIdx = (angle: number): number[] => {
      const v = new FakeView();
      const p = new Presenter(v);
      p.setSliceAngle(angle);
      p.start();
      p.tick(0.2); // complete the tip ramp
      return v.last().flatMap((m, i) => (Math.abs(m[2]) < 1e-6 ? [i] : []));
    };
    const axial = tippedIdx(0); // slice ⟂ z
    const oblique = tippedIdx(90); // slice ⟂ x → a different set of spins
    expect(axial.length).toBeGreaterThan(0);
    expect(oblique.length).toBeGreaterThan(0);
    expect(oblique).not.toEqual(axial); // different spins selected (set, not just count)
  });

  it('Larmor selects the slice height (different Larmor → different slab)', () => {
    const tippedIdx = (larmor: number): number[] => {
      const v = new FakeView();
      const p = new Presenter(v);
      p.setLarmor(larmor);
      p.start();
      p.tick(0.2);
      return v.last().flatMap((m, i) => (Math.abs(m[2]) < 1e-6 ? [i] : []));
    };
    const low = tippedIdx(63.83); // RF tuned to a low slice (MHz)
    const high = tippedIdx(63.91); // RF tuned to a high slice
    expect(low.length).toBeGreaterThan(0);
    expect(high.length).toBeGreaterThan(0);
    expect(low).not.toEqual(high); // different slab selected
  });
});

class FakePanels implements Panels {
  draws = 0;
  drawKspace(): void {
    this.draws++;
  }
  drawImage(): void {
    this.draws++;
  }
}

describe('Presenter k-space acquisition (synced to speed)', () => {
  it('ticking acquires k-space lines (panels redraw); speed 0 freezes acquisition', () => {
    const panels = new FakePanels();
    const p = new Presenter(new FakeView(), panels, diskPhantom(8));
    p.start();
    const afterStart = panels.draws;

    p.setSpeed(0);
    p.tick(1.0);
    expect(panels.draws).toBe(afterStart); // frozen → no new lines

    p.setSpeed(1);
    p.tick(1.0); // ≥ several LINE_DT of sim time → acquires lines → redraws
    expect(panels.draws).toBeGreaterThan(afterStart);
  });
});
