import type { HeartEntry } from './manifest';
import { glbUrl } from './manifest';
import type { HeartViewer } from './viewer';
import { efCategory, efError, fmtMl, fmtPct } from './metrics';
import { showSpinner, hideSpinner } from './spinner';
import { pickDefault } from './select';
import { Segmenter } from './segment';
import { readNifti, frame } from './nifti';
import { chamberPolys } from './mesh';
import { countLabel, volumeMl, TARGET_MM } from './preprocess';

const CARD =
  'position:fixed;top:12px;left:12px;z-index:10;background:rgba(20,24,30,.86);color:#cdd6e0;' +
  'font:13px system-ui,sans-serif;padding:12px 14px;border-radius:10px;border:1px solid #2a323d;' +
  'user-select:none;display:flex;flex-direction:column;gap:9px;min-width:248px;';
const SELECT = 'background:#11151b;color:#cdd6e0;border:1px solid #2a323d;border-radius:6px;padding:5px;width:100%;';
const FPS = 12;
const IMPORT = '__import__';

interface Imported {
  name: string;
  polys: (any | null)[]; // shown frame (ED for a cine, the volume for a single scan)
  readout: string;
  frames?: (any | null)[][]; // segmented cine frames (if 4D) -> beating
  edIdx?: number;
  esIdx?: number;
}

/** Top panel: Model dropdown + Hearts dropdown (each = defaults + import), drives the viewer. */
export function mountPanel(entries: HeartEntry[], viewer: HeartViewer, modelName = 'mnm2'): void {
  const seg = new Segmenter();
  const DEFAULT_MODEL = `models/${modelName}.onnx`;
  let modelSrc: string | Uint8Array = DEFAULT_MODEL;
  let modelKey = '';
  const imported = new Map<string, Imported>();

  const wrap = el('div', CARD);
  const title = el('div', 'font-size:14px;color:#e8edf4;');
  title.innerHTML = '<b>cardioview</b> — segmented hearts';

  const modelSel = el('select', SELECT) as HTMLSelectElement;
  modelSel.append(opt('bundled', `${modelName}.onnx`), opt(IMPORT, 'import .onnx…'));
  const heartSel = el('select', SELECT) as HTMLSelectElement;
  for (const e of entries) heartSel.append(opt(e.patient, `${e.patient}  (${e.group}${e.held_out ? ', held-out' : ''})`));
  heartSel.append(opt(IMPORT, 'import scan (.nii.gz)…'));

  const controls = el('div', 'display:flex;flex-direction:column;gap:6px;');
  const readout = el('div', 'display:flex;flex-direction:column;gap:3px;line-height:1.5;');
  readout.id = 'cv-readout';
  const statusEl = el('div', 'color:#8aa0b6;font-size:11px;min-height:14px;');
  const status = (s: string) => (statusEl.textContent = s);

  const onnxInput = fileInput('.onnx', false, async (fs) => {
    modelSrc = new Uint8Array(await fs[0].arrayBuffer());
    modelKey = '';
    upsertOption(modelSel, fs[0].name, fs[0].name, 'bundled');
    status(`model set: ${fs[0].name}`);
  });
  const scanInput = fileInput('.nii,.nii.gz,.gz', true, (fs) => void onScans(fs));

  wrap.append(title, modelSel, heartSel, controls, readout, statusEl, onnxInput, scanInput);
  document.body.appendChild(wrap);
  mountLegend();

  // ---- model selection ----
  modelSel.addEventListener('change', () => {
    if (modelSel.value === IMPORT) {
      modelSel.value = modelSel.dataset.last || 'bundled';
      onnxInput.click();
    } else {
      modelSel.dataset.last = modelSel.value;
      if (modelSel.value === 'bundled') {
        modelSrc = DEFAULT_MODEL;
        modelKey = '';
      }
    }
  });

  async function ensureModel(): Promise<void> {
    const key = typeof modelSrc === 'string' ? modelSrc : `bytes:${modelSrc.length}`;
    if (modelKey === key) return;
    status('loading model…');
    await seg.load(modelSrc);
    modelKey = key;
  }

  // ---- heart selection ----
  heartSel.addEventListener('change', () => {
    const v = heartSel.value;
    if (v === IMPORT) {
      heartSel.value = heartSel.dataset.last || entries[0]?.patient || '';
      scanInput.click();
    } else {
      heartSel.dataset.last = v;
      if (imported.has(v)) showImported(imported.get(v)!);
      else showCanned(entries.find((e) => e.patient === v)!);
    }
  });

  function showCanned(e: HeartEntry): void {
    controls.innerHTML = '';
    readout.innerHTML = cannedReadout(e);
    if (e.frames && e.frames.length > 1) buildAnimated(e);
    else buildStatic(e);
  }

  function showImported(im: Imported): void {
    controls.innerHTML = '';
    readout.innerHTML = im.readout;
    if (im.frames && im.frames.length > 1) {
      viewer.showSequence(im.frames);
      animControls(im.frames.length, im.edIdx ?? 0, im.esIdx ?? 0);
    } else {
      viewer.showStatic(im.polys);
    }
  }

  /** Play/pause + scrub + ED/ES jump for a sequence already loaded in the viewer; opens paused at ED. */
  function animControls(n: number, edIdx: number, esIdx: number): void {
    const playBtn = button('▶ play');
    const label = el('span', 'color:#8aa0b6;font-size:12px;min-width:78px;text-align:right;') as HTMLSpanElement;
    const scrub = el('input', 'width:100%;') as HTMLInputElement;
    scrub.type = 'range';
    scrub.min = '0';
    scrub.max = String(n - 1);
    const jumps = row(button('ED · full'), button('ES · empty'));
    controls.append(row(playBtn, label), scrub, jumps);

    const setLabel = (i: number) => {
      const tag = i === edIdx ? ' · ED' : i === esIdx ? ' · ES' : '';
      label.textContent = `frame ${i + 1}/${n}${tag}`;
      scrub.value = String(i);
    };
    const stop = () => {
      viewer.pause();
      playBtn.textContent = '▶ play';
    };
    const start = () => {
      viewer.play(FPS, setLabel);
      playBtn.textContent = '❚❚ pause';
    };
    const jump = (i: number) => {
      stop();
      viewer.showFrame(i);
      setLabel(i);
    };
    playBtn.addEventListener('click', () => (viewer.isPlaying ? stop() : start()));
    scrub.addEventListener('input', () => jump(Number(scrub.value)));
    jumps.children[0].addEventListener('click', () => jump(edIdx));
    jumps.children[1].addEventListener('click', () => jump(esIdx));
    jump(edIdx); // open on ED, paused
  }

  async function onScans(files: File[]): Promise<void> {
    showSpinner();
    try {
      await ensureModel();
      let last = '';
      for (const f of files) {
        const v = await readNifti(f);
        const name = uniqueName(f.name, imported);
        if (v.t > 1) await segmentCine(name, f.name, v);
        else await segmentSingle(name, v);
        upsertOption(heartSel, name, `${name} (imported)`, IMPORT);
        last = name;
      }
      if (last) {
        heartSel.value = last;
        heartSel.dataset.last = last;
        showImported(imported.get(last)!);
      }
      status(`imported ${files.length} scan${files.length > 1 ? 's' : ''}`);
    } catch (e: any) {
      status(`error: ${e?.message ?? e}`);
    } finally {
      hideSpinner();
    }
  }

  async function segmentSingle(name: string, v: Awaited<ReturnType<typeof readNifti>>): Promise<void> {
    status(`segmenting ${name}…`);
    const t0 = performance.now();
    const masks = await seg.segmentVolume(frame(v, 0), v.d, v.h, v.w, v.spacingYX);
    const secs = (performance.now() - t0) / 1000;
    const vox = TARGET_MM * TARGET_MM * v.zSpacing;
    const ro = importedReadout(name,
      volumeMl(countLabel(masks, 3), vox), volumeMl(countLabel(masks, 2), vox), volumeMl(countLabel(masks, 1), vox)) +
      `<div style="color:#8aa0b6;">${v.d} slices · ${seg.provider} · ${secs.toFixed(1)} s</div>`;
    imported.set(name, { name, polys: chamberPolys(masks, v.zSpacing), readout: ro });
  }

  async function segmentCine(name: string, label: string, v: Awaited<ReturnType<typeof readNifti>>): Promise<void> {
    const t0 = performance.now();
    const vox = TARGET_MM * TARGET_MM * v.zSpacing;
    const framePolys: (any | null)[][] = [];
    const lv: number[] = [];
    for (let i = 0; i < v.t; i++) {
      status(`segmenting ${label} ${i + 1}/${v.t}…`);
      const masks = await seg.segmentVolume(frame(v, i), v.d, v.h, v.w, v.spacingYX);
      framePolys.push(chamberPolys(masks, v.zSpacing));
      lv.push(volumeMl(countLabel(masks, 3), vox));
    }
    const edIdx = argExtreme(lv, true); // ED = max LV-cavity volume (full)
    const esIdx = argExtreme(lv, false); // ES = min (empty)
    const secs = (performance.now() - t0) / 1000;
    imported.set(name, {
      name, polys: framePolys[edIdx], frames: framePolys, edIdx, esIdx,
      readout: cineReadout(lv[edIdx], lv[esIdx], lv, edIdx, esIdx, v.t, seg.provider, secs),
    });
  }

  // ---- canned heart controls ----
  function buildAnimated(e: HeartEntry): void {
    showSpinner();
    void viewer
      .loadSequence(e.frames!.map(glbUrl))
      .then(() => animControls(e.frames!.length, e.ed_idx ?? 0, e.es_idx ?? 0))
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

  // ---- init ----
  const def = pickDefault(entries);
  if (def) {
    heartSel.value = def.patient;
    heartSel.dataset.last = def.patient;
    showCanned(def);
  }
}

function cannedReadout(e: HeartEntry): string {
  const p = e.pred;
  return `
    <div style="font-size:22px;color:#fff;">LVEF ${fmtPct(p.ef)}
      <span style="font-size:12px;color:#8aa0b6;">${efCategory(p.ef)}</span></div>
    <div><span style="color:#e8edf4;">EDV</span> <span style="color:#8aa0b6;">(full)</span> ${fmtMl(p.edv)}
      &nbsp; <span style="color:#e8edf4;">ESV</span> <span style="color:#8aa0b6;">(empty)</span> ${fmtMl(p.esv)}</div>
    <div style="color:#8aa0b6;">stroke volume ${fmtMl(p.edv - p.esv)}</div>
    <div style="color:#8aa0b6;border-top:1px solid #2a323d;padding-top:5px;margin-top:3px;">
      GT EF ${fmtPct(e.gt.ef)} &nbsp;·&nbsp; |error| ${efError(p.ef, e.gt.ef).toFixed(1)} pts
      &nbsp;·&nbsp; ${e.held_out ? 'held-out' : '⚠ train-seen'}</div>`;
}

function argExtreme(a: number[], max: boolean): number {
  return a.reduce((bi, v, i) => ((max ? v > a[bi] : v < a[bi]) ? i : bi), 0);
}

/** LV-cavity volume vs frame, ED (red) + ES (blue) marked. */
function sparkline(vals: number[], ed: number, es: number): string {
  const w = 224, h = 40, pad = 4;
  const mn = Math.min(...vals), rng = Math.max(...vals) - mn || 1;
  const x = (i: number) => pad + (i * (w - 2 * pad)) / Math.max(1, vals.length - 1);
  const y = (val: number) => h - pad - ((val - mn) / rng) * (h - 2 * pad);
  const pts = vals.map((val, i) => `${x(i).toFixed(1)},${y(val).toFixed(1)}`).join(' ');
  const dot = (i: number, c: string) => `<circle cx="${x(i).toFixed(1)}" cy="${y(vals[i]).toFixed(1)}" r="3" fill="${c}"/>`;
  return `<svg width="${w}" height="${h}" style="display:block;margin:2px 0;">` +
    `<polyline points="${pts}" fill="none" stroke="#7a9bff" stroke-width="1.5"/>${dot(ed, '#ef5350')}${dot(es, '#5b8def')}</svg>` +
    `<div style="color:#8aa0b6;font-size:11px;">LV volume vs frame · ED ● / ES ●</div>`;
}

function cineReadout(edv: number, esv: number, lv: number[], ed: number, es: number, t: number, provider: string, secs: number): string {
  const ef = edv > 0 ? ((edv - esv) / edv) * 100 : NaN;
  return `
    <div style="font-size:22px;color:#fff;">LVEF ${fmtPct(ef)}
      <span style="font-size:12px;color:#8aa0b6;">${efCategory(ef)}</span></div>
    <div><span style="color:#e8edf4;">EDV</span> <span style="color:#8aa0b6;">(full)</span> ${fmtMl(edv)}
      &nbsp; <span style="color:#e8edf4;">ESV</span> <span style="color:#8aa0b6;">(empty)</span> ${fmtMl(esv)}</div>
    ${sparkline(lv, ed, es)}
    <div style="color:#8aa0b6;border-top:1px solid #2a323d;padding-top:5px;margin-top:3px;">
      imported cine · ${t} frames · ${provider} · ${secs.toFixed(1)} s</div>
    <div style="color:#8aa0b6;">regular gated cycle (ACDC) — not arrhythmia</div>`;
}

function importedReadout(name: string, cav: number, myo: number, rv: number): string {
  return `
    <div style="color:#fff;">${name}</div>
    <div><span style="color:#ef5350;">LV cav</span> ${fmtMl(cav)} ·
      <span style="color:#ffca5b;">myo</span> ${fmtMl(myo)} ·
      <span style="color:#5b8def;">RV</span> ${fmtMl(rv)}</div>
    <div style="color:#8aa0b6;">prediction · single volume (no EF — needs ED+ES)</div>`;
}

function uniqueName(name: string, taken: Map<string, unknown>): string {
  if (!taken.has(name)) return name;
  let i = 2;
  while (taken.has(`${name} (${i})`)) i++;
  return `${name} (${i})`;
}

function upsertOption(sel: HTMLSelectElement, value: string, text: string, before: string): void {
  if (![...sel.options].some((o) => o.value === value)) {
    const o = opt(value, text);
    const ref = [...sel.options].find((x) => x.value === before);
    sel.insertBefore(o, ref ?? null);
  }
}

// ---- legend + tiny DOM helpers ----
const LEGEND: [string, string][] = [
  ['#ef5350', 'LV cavity — blood pool'],
  ['#ffca5b', 'LV myocardium — heart muscle'],
  ['#5b8def', 'RV cavity'],
];

function mountLegend(): void {
  const box = el('div',
    'position:fixed;top:12px;right:12px;z-index:10;background:rgba(20,24,30,.86);color:#cdd6e0;' +
    'font:12px system-ui,sans-serif;padding:10px 12px;border-radius:10px;border:1px solid #2a323d;' +
    'user-select:none;display:flex;flex-direction:column;gap:6px;');
  const t = el('div', 'color:#9aa7b8;margin-bottom:2px;');
  t.textContent = 'chambers';
  box.appendChild(t);
  for (const [color, label] of LEGEND) {
    const r = el('div', 'display:flex;align-items:center;gap:8px;');
    r.innerHTML = `<span style="flex:0 0 14px;height:14px;border-radius:3px;background:${color};border:1px solid #3a424d;"></span><span>${label}</span>`;
    box.appendChild(r);
  }
  document.body.appendChild(box);
}

function el(tag: string, css: string): HTMLElement {
  const e = document.createElement(tag);
  e.style.cssText = css;
  return e;
}

function opt(value: string, text: string): HTMLOptionElement {
  const o = document.createElement('option');
  o.value = value;
  o.textContent = text;
  return o;
}

function row(...kids: HTMLElement[]): HTMLDivElement {
  const d = el('div', 'display:flex;gap:6px;align-items:center;') as HTMLDivElement;
  d.append(...kids);
  return d;
}

function button(label: string): HTMLButtonElement {
  const b = el('button', 'flex:1;background:#11151b;color:#cdd6e0;border:1px solid #2a323d;border-radius:6px;padding:6px;cursor:pointer;') as HTMLButtonElement;
  b.textContent = label;
  return b;
}

function fileInput(accept: string, multiple: boolean, onPick: (fs: File[]) => void): HTMLInputElement {
  const input = el('input', 'display:none;') as HTMLInputElement;
  input.type = 'file';
  input.accept = accept;
  input.multiple = multiple;
  input.addEventListener('change', () => {
    const fs = input.files ? [...input.files] : [];
    if (fs.length) onPick(fs);
    input.value = '';
  });
  return input;
}
