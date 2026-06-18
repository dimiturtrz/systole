// Shared pulse-sequence timing — the SINGLE source of truth for where, within a TR,
// each spatial encoding happens. Both the Presenter (gradient coloring) and the
// SequenceDiagram (drawing) read these so they can never drift apart.
//
// Real MRI: the encodings all live in the early part of the TR, finishing at the echo
// (TE); the rest of the TR is relaxation/wait. So the windows scale with TE, not TR —
// that keeps slice → phase → readout ordered for any realistic TE ≪ TR.

export type Stage = 'slice' | 'phase' | 'freq' | 'idle';

export interface SeqWindows {
  sliceEnd: number; // Gz slice-select (+ RF) ends
  peStart: number; // Gy phase-encode blip
  peEnd: number;
  roStart: number; // Gx readout / ADC window (centered on TE)
  roEnd: number;
}

/** Stage window boundaries (seconds into the TR), as fractions of TE. */
export function seqWindows(_tr: number, te: number): SeqWindows {
  const roHalf = 0.18 * te;
  return {
    sliceEnd: 0.22 * te,
    peStart: 0.30 * te,
    peEnd: 0.58 * te,
    roStart: te - roHalf,
    roEnd: te + roHalf,
  };
}

/** Which encoding is active at `ct` seconds into the TR (idle = relaxation/wait). */
export function stageAt(tr: number, te: number, ct: number): Stage {
  const w = seqWindows(tr, te);
  if (ct < w.sliceEnd) return 'slice';
  if (ct >= w.peStart && ct < w.peEnd) return 'phase';
  if (ct >= w.roStart && ct <= w.roEnd) return 'freq';
  return 'idle';
}
