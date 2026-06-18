import { describe, it, expect } from 'vitest';
import { Simulator } from '../../src/model/Simulator';
import { directionFromAngles } from '../../src/model/physics';

const close = (a: number, b: number, eps = 1e-9) => expect(Math.abs(a - b)).toBeLessThan(eps);

describe('Simulator (proton precession)', () => {
  it('step advances phase by 2π·larmorHz·dt', () => {
    const sim = new Simulator(0.25, 0.12, Infinity); // t1=∞ → no tilt relax
    const theta = [0.12];
    const phase = [0];
    sim.step(theta, phase, 1); // 0.25 Hz × 1 s = quarter turn
    close(phase[0], 2 * Math.PI * 0.25);
    close(theta[0], 0.12); // unchanged (t1=∞)
  });

  it('tilt relaxes toward restTilt (T1)', () => {
    const sim = new Simulator(0, 0.12, 1); // no precession, t1=1
    const theta = [Math.PI / 2];
    const phase = [0];
    sim.step(theta, phase, 1);
    close(theta[0], 0.12 + (Math.PI / 2 - 0.12) * Math.exp(-1));
  });

  it('precession rotates the transverse direction', () => {
    const sim = new Simulator(0.25, 0.12, Infinity);
    const theta = [Math.PI / 2];
    const phase = [0];
    sim.step(theta, phase, 1); // quarter turn
    const d = directionFromAngles(theta[0], phase[0]);
    close(d[0], 0, 1e-6);
    close(d[1], 1, 1e-6);
  });

  it('exciteSlab tips only in-slab protons to transverse + aligns their phase', () => {
    const theta = [0.12, 0.12, 0.12];
    const phase = [1, 2, 3];
    const pos: [number, number, number][] = [[0, 0, -1], [0, 0, 0], [0, 0, 1]];
    new Simulator().exciteSlab(theta, phase, pos, 0, 0.6, Math.PI / 2);
    close(theta[1], Math.PI / 2); // slab tipped
    expect(phase[1]).toBe(0); // coherent
    expect(theta[0]).toBe(0.12); // others untouched
    expect(theta[2]).toBe(0.12);
    expect(phase[0]).toBe(1);
    expect(phase[2]).toBe(3);
  });
});
