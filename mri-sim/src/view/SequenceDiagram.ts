// Pulse-sequence diagram (canvas): walks the three spatial encodings each TR —
// 1) slice select (Gz + RF), 2) phase encode (Gy), 3) frequency encode / readout (Gx + ADC).
// The active stage is highlighted as the playhead sweeps. View only.

import { seqWindows, stageAt } from '../model/sequence';

export interface SeqState {
  tr: number;
  te: number;
  cycleTime: number;
  peStep: number; // phase-encode value −1…1 (steps each TR; RF stays the same)
}

export interface SequenceView {
  draw(s: SeqState): void;
}

const ROWS = ['RF', 'Gz slice', 'Gy phase', 'Gx read', 'ADC'];
const CYAN = '93,209,196';
const BLUE = '122,162,255';
const ORANGE = '240,179,91';
const GREEN = '110,231,168';

export function mountSequenceDiagram(): SequenceView {
  const W = 384;
  const H = 196;
  const wrap = document.createElement('div');
  wrap.style.cssText =
    'position:fixed;right:12px;bottom:12px;z-index:10;background:rgba(20,24,30,.85);' +
    'padding:8px 10px;border-radius:8px;border:1px solid #2a323d;font:11px system-ui,sans-serif;color:#9aa7b8;';
  const title = document.createElement('div');
  title.textContent = 'pulse sequence';
  title.style.marginBottom = '4px';
  const canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  wrap.appendChild(title);
  wrap.appendChild(canvas);
  document.body.appendChild(wrap);
  const ctx = canvas.getContext('2d')!;

  const padL = 48;
  const padR = 10;
  const padT = 22; // caption row
  const padB = 16; // TR/TE labels
  const plot = W - padL - padR;
  const rowH = (H - padT - padB) / ROWS.length;

  return {
    draw({ tr, te, cycleTime, peStep }: SeqState): void {
      const w = seqWindows(tr, te);
      // Real TE ≪ TR → zoom the x-axis to the encode window; the long TR wait is captioned.
      const span = Math.min(tr, w.roEnd * 1.3);
      const truncated = tr > span * 1.02;
      const tx = (t: number): number => padL + (Math.max(0, Math.min(t, span)) / span) * plot;
      const yMid = (i: number): number => padT + i * rowH + rowH / 2;
      const amp = rowH * 0.34;

      const ct = cycleTime;
      const stage = stageAt(tr, te, ct);
      const a = (on: boolean): number => (on ? 1 : 0.3);

      ctx.clearRect(0, 0, W, H);
      ctx.font = '10px system-ui';

      // caption
      const waitMs = Math.max(0, (tr - ct) * 1000);
      const caption =
        stage === 'slice' ? '1 · slice select  (Gz + RF)' :
        stage === 'phase' ? '2 · phase encode  (Gy)' :
        stage === 'freq' ? '3 · frequency encode / readout  (Gx + ADC)' :
        `· relaxation / wait ⏩ — next RF in ${waitMs.toFixed(0)} ms`;
      ctx.fillStyle = stage === 'idle' ? '#7b8a9c' : '#e8edf4';
      ctx.fillText(caption, padL, 13);

      // row baselines + labels
      ROWS.forEach((name, i) => {
        ctx.fillStyle = '#7b8a9c';
        ctx.fillText(name, 4, yMid(i) + 3);
        ctx.strokeStyle = '#2a323d';
        ctx.beginPath();
        ctx.moveTo(padL, yMid(i));
        ctx.lineTo(W - padR, yMid(i));
        ctx.stroke();
      });

      const x0 = tx(0);
      const xTE = tx(te);

      // RF spike (row 0)
      ctx.strokeStyle = `rgba(${CYAN},${a(stage === 'slice')})`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x0, yMid(0));
      ctx.lineTo(x0, yMid(0) - amp * 1.6);
      ctx.stroke();

      // Gz slice-select block during RF (row 1)
      ctx.fillStyle = `rgba(${BLUE},${a(stage === 'slice') * 0.7})`;
      ctx.fillRect(x0 - 5, yMid(1) - amp, tx(w.sliceEnd) - x0 + 5, amp);

      // Gy phase-encode blip, height steps with peStep (row 2)
      const peH = amp * 0.9 * peStep;
      ctx.fillStyle = `rgba(${ORANGE},${a(stage === 'phase') * 0.85})`;
      const gx = tx(w.peStart);
      ctx.fillRect(gx, yMid(2) - Math.max(0, peH), tx(w.peEnd) - gx, Math.abs(peH) || 1);

      // Gx readout gradient during ADC (row 3)
      ctx.fillStyle = `rgba(${GREEN},${a(stage === 'freq') * 0.7})`;
      ctx.fillRect(tx(w.roStart), yMid(3) - amp, tx(w.roEnd) - tx(w.roStart), amp);

      // ADC window (row 4)
      ctx.fillStyle = `rgba(${CYAN},${a(stage === 'freq') * 0.4})`;
      ctx.fillRect(tx(w.roStart), yMid(4) - amp, tx(w.roEnd) - tx(w.roStart), amp);
      ctx.strokeStyle = `rgba(${CYAN},${a(stage === 'freq')})`;
      ctx.lineWidth = 1;
      ctx.strokeRect(tx(w.roStart), yMid(4) - amp, tx(w.roEnd) - tx(w.roStart), amp);

      // TE marker
      ctx.strokeStyle = `rgba(${ORANGE},.8)`;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(xTE, padT);
      ctx.lineTo(xTE, H - padB);
      ctx.stroke();
      ctx.setLineDash([]);

      // playhead
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(tx(ct), padT);
      ctx.lineTo(tx(ct), H - padB);
      ctx.stroke();

      // axis-break marker: the plotted span is only the encode window; TR is far later
      if (truncated) {
        ctx.fillStyle = '#7b8a9c';
        ctx.fillText('⋯ wait', W - padR - 38, padT - 9);
      }

      // labels (ms — real MRI units)
      const ms = (s: number): string => (s < 1 ? `${(s * 1000).toFixed(s < 0.1 ? 1 : 0)} ms` : `${s.toFixed(2)} s`);
      ctx.fillStyle = '#9aa7b8';
      ctx.textAlign = 'right';
      ctx.fillText(`TR=${ms(tr)}`, W - padR, H - 3);
      ctx.textAlign = 'left';
      ctx.fillStyle = `rgb(${ORANGE})`;
      ctx.fillText(`TE=${ms(te)}`, padL, H - 3);
    },
  };
}
