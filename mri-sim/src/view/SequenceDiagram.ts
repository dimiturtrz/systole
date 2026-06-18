// Pulse-sequence diagram (canvas): RF, slice-select, phase-encode, readout rows over
// one TR, with TR/TE marked and a playhead at the current cycle time. View only.

export interface SeqState {
  tr: number;
  te: number;
  cycleTime: number;
  peStep: number; // phase-encode value −1…1 (steps each TR; RF stays the same)
}

export interface SequenceView {
  draw(s: SeqState): void;
}

const ROWS = ['RF', 'Gss', 'Gpe', 'ADC'];

export function mountSequenceDiagram(): SequenceView {
  const W = 360;
  const H = 150;
  const wrap = document.createElement('div');
  wrap.style.cssText =
    'position:fixed;right:12px;bottom:12px;z-index:10;background:rgba(20,24,30,.82);' +
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

  const padL = 34;
  const padR = 10;
  const padT = 6;
  const plot = W - padL - padR;
  const rowH = (H - padT - 14) / ROWS.length;
  const tx = (t: number, tr: number): number => padL + (t / tr) * plot;

  return {
    draw({ tr, te, cycleTime, peStep }: SeqState): void {
      ctx.clearRect(0, 0, W, H);
      ctx.font = '10px system-ui';

      ROWS.forEach((name, i) => {
        const yMid = padT + i * rowH + rowH / 2;
        ctx.fillStyle = '#7b8a9c';
        ctx.fillText(name, 4, yMid + 3);
        ctx.strokeStyle = '#2a323d';
        ctx.beginPath();
        ctx.moveTo(padL, yMid);
        ctx.lineTo(W - padR, yMid);
        ctx.stroke();
      });

      const x0 = tx(0, tr);
      const xTE = tx(te, tr);
      const yRF = padT + 0 * rowH + rowH / 2;
      const ySS = padT + 1 * rowH + rowH / 2;
      const yPE = padT + 2 * rowH + rowH / 2;
      const yADC = padT + 3 * rowH + rowH / 2;
      const amp = rowH * 0.35;

      // RF: a pulse at t=0
      ctx.strokeStyle = '#5dd1c4';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x0, yRF);
      ctx.lineTo(x0, yRF - amp * 1.6);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(x0, yRF - amp * 1.6, 2, 0, Math.PI * 2);
      ctx.fillStyle = '#5dd1c4';
      ctx.fill();

      // Gss: slice-select block during the RF
      ctx.fillStyle = 'rgba(122,162,255,.5)';
      ctx.fillRect(x0 - 6, ySS - amp, 12, amp);

      // Gpe: phase-encode blip just after RF — height steps with peStep each TR
      const peH = amp * 0.8 * peStep; // signed: up or down from the line
      ctx.fillStyle = 'rgba(240,179,91,.7)';
      ctx.fillRect(x0 + 10, yPE - Math.max(0, peH), 8, Math.abs(peH) || 1);

      // ADC: readout window centered at TE
      ctx.fillStyle = 'rgba(93,209,196,.35)';
      ctx.fillRect(xTE - 18, yADC - amp, 36, amp);
      ctx.strokeStyle = '#5dd1c4';
      ctx.lineWidth = 1;
      ctx.strokeRect(xTE - 18, yADC - amp, 36, amp);

      // TE marker (0 → TE)
      ctx.strokeStyle = '#f0b35b';
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(xTE, padT);
      ctx.lineTo(xTE, H - 14);
      ctx.stroke();
      ctx.setLineDash([]);

      // playhead
      const xp = tx(Math.min(cycleTime, tr), tr);
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(xp, padT);
      ctx.lineTo(xp, H - 14);
      ctx.stroke();

      // labels
      ctx.fillStyle = '#9aa7b8';
      ctx.fillText(`TR=${tr.toFixed(1)}s`, W - padR - 56, H - 3);
      ctx.fillStyle = '#f0b35b';
      ctx.fillText(`TE=${te.toFixed(2)}s`, padL, H - 3);
    },
  };
}
