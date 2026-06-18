import type { HeartEntry } from './manifest';
import { glbUrl } from './manifest';
import type { HeartViewer } from './viewer';
import { efCategory, efError, fmtMl, fmtPct } from './metrics';
import { showSpinner, hideSpinner } from './spinner';

const CARD =
  'position:fixed;top:12px;left:12px;z-index:10;background:rgba(20,24,30,.86);color:#cdd6e0;' +
  'font:13px system-ui,sans-serif;padding:12px 14px;border-radius:10px;border:1px solid #2a323d;' +
  'user-select:none;display:flex;flex-direction:column;gap:10px;min-width:248px;';
const FPS = 12;

/** Control + readout panel; drives the viewer (static ED/ES toggle or beating cycle). */
export function mountPanel(entries: HeartEntry[], viewer: HeartViewer): void {
  let current = entries[0];

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

  const controls = document.createElement('div'); // rebuilt per patient (anim vs static)
  controls.style.cssText = 'display:flex;flex-direction:column;gap:6px;';

  const readout = document.createElement('div');
  readout.style.cssText = 'display:flex;flex-direction:column;gap:3px;line-height:1.5;';

  wrap.append(title, select, controls, readout);
  document.body.appendChild(wrap);
  mountLegend();

  function refreshReadout(): void {
    const p = current.pred;
    readout.innerHTML = `
      <div style="font-size:22px;color:#fff;">LVEF ${fmtPct(p.ef)}
        <span style="font-size:12px;color:#8aa0b6;">${efCategory(p.ef)}</span></div>
      <div><span style="color:#e8edf4;">EDV</span> <span style="color:#8aa0b6;">(full)</span> ${fmtMl(p.edv)}
        &nbsp; <span style="color:#e8edf4;">ESV</span> <span style="color:#8aa0b6;">(empty)</span> ${fmtMl(p.esv)}</div>
      <div style="color:#8aa0b6;">stroke volume ${fmtMl(p.edv - p.esv)}</div>
      <div style="color:#8aa0b6;border-top:1px solid #2a323d;padding-top:5px;margin-top:3px;">
        GT EF ${fmtPct(current.gt.ef)} &nbsp;·&nbsp; |error| ${efError(p.ef, current.gt.ef).toFixed(1)} pts
        &nbsp;·&nbsp; ${current.held_out ? 'held-out' : '⚠ train-seen'}</div>`;
  }

  function buildAnimated(e: HeartEntry): void {
    const frames = e.frames!;
    const playBtn = button('❚❚ pause');
    const label = document.createElement('span');
    label.style.cssText = 'color:#8aa0b6;font-size:12px;min-width:78px;text-align:right;';
    const top = row(playBtn, label);

    const scrub = document.createElement('input');
    scrub.type = 'range';
    scrub.min = '0';
    scrub.max = String(frames.length - 1);
    scrub.value = '0';
    scrub.style.width = '100%';

    const jumps = row(jumpButton('ED · full'), jumpButton('ES · empty'));
    controls.append(top, scrub, jumps);

    const setLabel = (i: number) => {
      const tag = i === (e.ed_idx ?? -1) ? ' · ED' : i === (e.es_idx ?? -1) ? ' · ES' : '';
      label.textContent = `frame ${i + 1}/${frames.length}${tag}`;
    };
    const onFrame = (i: number) => {
      scrub.value = String(i);
      setLabel(i);
    };
    const start = () => {
      viewer.play(FPS, onFrame);
      playBtn.textContent = '❚❚ pause';
    };
    const stop = () => {
      viewer.pause();
      playBtn.textContent = '▶ play';
    };

    playBtn.addEventListener('click', () => (viewer.isPlaying ? stop() : start()));
    scrub.addEventListener('input', () => {
      stop();
      const i = Number(scrub.value);
      viewer.showFrame(i);
      setLabel(i);
    });
    jumps.children[0].addEventListener('click', () => jump(e.ed_idx ?? 0));
    jumps.children[1].addEventListener('click', () => jump(e.es_idx ?? 0));
    function jump(i: number) {
      stop();
      viewer.showFrame(i);
      onFrame(i);
    }

    showSpinner();
    void viewer
      .loadSequence(frames.map(glbUrl))
      .then(() => {
        setLabel(0);
        start();
      })
      .finally(hideSpinner);
  }

  function buildStatic(e: HeartEntry): void {
    const edBtn = button('ED · full');
    const esBtn = button('ES · empty');
    controls.append(row(edBtn, esBtn));
    const setPhase = (ph: 'ED' | 'ES') => {
      edBtn.style.opacity = ph === 'ED' ? '1' : '0.5';
      esBtn.style.opacity = ph === 'ES' ? '1' : '0.5';
      const file = e.glb[ph];
      if (file) {
        showSpinner();
        void viewer.load(glbUrl(file)).finally(hideSpinner);
      }
    };
    edBtn.addEventListener('click', () => setPhase('ED'));
    esBtn.addEventListener('click', () => setPhase('ES'));
    setPhase('ED');
  }

  function loadCurrent(): void {
    refreshReadout();
    controls.innerHTML = '';
    if (current.frames && current.frames.length > 1) buildAnimated(current);
    else buildStatic(current);
  }

  select.addEventListener('change', () => {
    current = entries.find((e) => e.patient === select.value) ?? entries[0];
    loadCurrent();
  });

  loadCurrent();
}

// Chamber color key (top-right) — colors match the meshes in viewer.ts.
const LEGEND: [string, string][] = [
  ['#ef5350', 'LV cavity — blood pool'],
  ['#ffca5b', 'LV myocardium — heart muscle'],
  ['#5b8def', 'RV cavity'],
];

function mountLegend(): void {
  const box = document.createElement('div');
  box.style.cssText =
    'position:fixed;top:12px;right:12px;z-index:10;background:rgba(20,24,30,.86);color:#cdd6e0;' +
    'font:12px system-ui,sans-serif;padding:10px 12px;border-radius:10px;border:1px solid #2a323d;' +
    'user-select:none;display:flex;flex-direction:column;gap:6px;';
  const t = document.createElement('div');
  t.textContent = 'chambers';
  t.style.cssText = 'color:#9aa7b8;margin-bottom:2px;';
  box.appendChild(t);
  for (const [color, label] of LEGEND) {
    const r = document.createElement('div');
    r.style.cssText = 'display:flex;align-items:center;gap:8px;';
    r.innerHTML = `<span style="flex:0 0 14px;height:14px;border-radius:3px;background:${color};border:1px solid #3a424d;"></span><span>${label}</span>`;
    box.appendChild(r);
  }
  document.body.appendChild(box);
}

function row(...kids: HTMLElement[]): HTMLDivElement {
  const d = document.createElement('div');
  d.style.cssText = 'display:flex;gap:6px;align-items:center;';
  d.append(...kids);
  return d;
}

function button(label: string): HTMLButtonElement {
  const b = document.createElement('button');
  b.textContent = label;
  b.style.cssText =
    'flex:1;background:#11151b;color:#cdd6e0;border:1px solid #2a323d;border-radius:6px;padding:6px;cursor:pointer;';
  return b;
}

function jumpButton(label: string): HTMLButtonElement {
  const b = button(label);
  b.style.flex = '1';
  return b;
}
