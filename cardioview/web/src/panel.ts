import type { HeartEntry } from './manifest';
import { glbUrl } from './manifest';
import type { HeartViewer } from './viewer';
import { efCategory, efError, fmtMl, fmtPct } from './metrics';

const CARD =
  'position:fixed;top:12px;left:12px;z-index:10;background:rgba(20,24,30,.86);color:#cdd6e0;' +
  'font:13px system-ui,sans-serif;padding:12px 14px;border-radius:10px;border:1px solid #2a323d;' +
  'user-select:none;display:flex;flex-direction:column;gap:10px;min-width:230px;';

/** Builds the control + readout panel and drives the viewer. View glue, no measurement logic. */
export function mountPanel(entries: HeartEntry[], viewer: HeartViewer): void {
  let current = entries[0];
  let phase: 'ED' | 'ES' = 'ED';

  const wrap = document.createElement('div');
  wrap.style.cssText = CARD;

  const title = document.createElement('div');
  title.innerHTML = '<b>cardioview</b> — segmented hearts';
  title.style.cssText = 'font-size:14px;color:#e8edf4;';

  const select = document.createElement('select');
  select.style.cssText = 'background:#11151b;color:#cdd6e0;border:1px solid #2a323d;border-radius:6px;padding:5px;';
  for (const e of entries) {
    const o = document.createElement('option');
    o.value = e.patient;
    o.textContent = `${e.patient}  (${e.group}${e.held_out ? ', held-out' : ''})`;
    select.appendChild(o);
  }

  const phaseRow = document.createElement('div');
  phaseRow.style.cssText = 'display:flex;gap:6px;';
  const edBtn = phaseButton('ED · full');
  const esBtn = phaseButton('ES · empty');
  phaseRow.append(edBtn, esBtn);

  const readout = document.createElement('div');
  readout.style.cssText = 'display:flex;flex-direction:column;gap:3px;line-height:1.5;';

  wrap.append(title, select, phaseRow, readout);
  document.body.appendChild(wrap);

  function refreshReadout(): void {
    const p = current.pred;
    const efErr = efError(current.pred.ef, current.gt.ef);
    readout.innerHTML = `
      <div style="font-size:22px;color:#fff;">LVEF ${fmtPct(p.ef)}
        <span style="font-size:12px;color:#8aa0b6;">${efCategory(p.ef)}</span></div>
      <div><span style="color:#ef5350;">EDV (full)</span> ${fmtMl(p.edv)}
        &nbsp; <span style="color:#5b8def;">ESV (empty)</span> ${fmtMl(p.esv)}</div>
      <div style="color:#8aa0b6;">stroke volume ${fmtMl(p.edv - p.esv)}</div>
      <div style="color:#8aa0b6;border-top:1px solid #2a323d;padding-top:5px;margin-top:3px;">
        GT EF ${fmtPct(current.gt.ef)} &nbsp;·&nbsp; |error| ${efErr.toFixed(1)} pts
        &nbsp;·&nbsp; ${current.held_out ? 'held-out' : '⚠ train-seen'}</div>`;
  }

  function setPhase(p: 'ED' | 'ES', keepCamera: boolean): void {
    phase = p;
    edBtn.dataset.on = String(p === 'ED');
    esBtn.dataset.on = String(p === 'ES');
    edBtn.style.opacity = p === 'ED' ? '1' : '0.5';
    esBtn.style.opacity = p === 'ES' ? '1' : '0.5';
    const file = current.glb[p];
    if (file) void viewer.load(glbUrl(file), keepCamera);
  }

  function selectPatient(patient: string): void {
    current = entries.find((e) => e.patient === patient) ?? entries[0];
    refreshReadout();
    setPhase(phase, false);
  }

  select.addEventListener('change', () => selectPatient(select.value));
  edBtn.addEventListener('click', () => setPhase('ED', true));
  esBtn.addEventListener('click', () => setPhase('ES', true));

  refreshReadout();
  setPhase('ED', false);
}

function phaseButton(label: string): HTMLButtonElement {
  const b = document.createElement('button');
  b.textContent = label;
  b.style.cssText =
    'flex:1;background:#11151b;color:#cdd6e0;border:1px solid #2a323d;border-radius:6px;' +
    'padding:6px;cursor:pointer;';
  return b;
}
