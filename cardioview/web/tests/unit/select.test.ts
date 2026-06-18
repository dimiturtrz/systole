import { describe, it, expect } from 'vitest';
import { pickDefault } from '../../src/select';
import type { HeartEntry } from '../../src/manifest';

const heart = (patient: string, group: string, ef: number): HeartEntry => ({
  patient,
  group,
  held_out: true,
  source: 'pred',
  pred: { ef, edv: 100, esv: 100 - ef },
  gt: { ef, edv: 100, esv: 100 - ef },
  glb: { ED: `${patient}_ED.gltf` },
});

describe('pickDefault', () => {
  it('prefers a NOR (normal) heart even if its EF is not the highest', () => {
    const entries = [heart('a', 'DCM', 15), heart('b', 'NOR', 58), heart('c', 'HCM', 62)];
    expect(pickDefault(entries).patient).toBe('b');
  });

  it('falls back to the highest-EF heart when no NOR present', () => {
    const entries = [heart('a', 'DCM', 15), heart('b', 'MINF', 27), heart('c', 'HCM', 62)];
    expect(pickDefault(entries).patient).toBe('c');
  });

  it('handles a single-entry list', () => {
    expect(pickDefault([heart('a', 'DCM', 15)]).patient).toBe('a');
  });
});
