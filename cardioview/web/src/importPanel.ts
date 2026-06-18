import type { HeartViewer } from './viewer';
import { Segmenter } from './segment';
import { readNifti } from './nifti';
import { chamberPolys } from './mesh';
import { countLabel, volumeMl, TARGET_MM } from './preprocess';
import { showSpinner, hideSpinner } from './spinner';
import { fmtMl } from './metrics';

const DEFAULT_MODEL = 'models/acdc_aug.onnx';
const CARD =
  'position:fixed;bottom:12px;left:12px;z-index:10;background:rgba(20,24,30,.86);color:#cdd6e0;' +
  'font:12px system-ui,sans-serif;padding:10px 12px;border-radius:10px;border:1px solid #2a323d;' +
  'user-select:none;display:flex;flex-direction:column;gap:8px;max-width:260px;';

/** "Import your own scan": pick a model (bundled or .onnx) + a .nii.gz → segment in-browser → render. */
export function mountImport(viewer: HeartViewer): void {
  const seg = new Segmenter();
  let modelSrc: string | Uint8Array = DEFAULT_MODEL;
  let loadedKey = '';

  const card = document.createElement('div');
  card.style.cssText = CARD;
  card.innerHTML = `
    <div style="color:#e8edf4;"><b>import your own</b></div>
    <label style="color:#8aa0b6;">model
      <select id="cv-model" style="width:100%;background:#11151b;color:#cdd6e0;border:1px solid #2a323d;border-radius:6px;padding:4px;margin-top:3px;">
        <option value="bundled">acdc_aug (bundled)</option>
        <option value="import">import .onnx…</option>
      </select></label>
    <button id="cv-scan" style="background:#11151b;color:#cdd6e0;border:1px solid #2a323d;border-radius:6px;padding:7px;cursor:pointer;">choose scan (.nii.gz)</button>
    <div id="cv-status" style="color:#8aa0b6;">bundled model · drop a heart in</div>
    <div id="cv-vols" style="line-height:1.5;"></div>`;
  document.body.appendChild(card);

  const status = (s: string) => ((card.querySelector('#cv-status') as HTMLElement).textContent = s);
  const vols = card.querySelector('#cv-vols') as HTMLElement;
  const modelSel = card.querySelector('#cv-model') as HTMLSelectElement;
  const onnxInput = fileInput('.onnx', async (f) => {
    modelSrc = new Uint8Array(await f.arrayBuffer());
    loadedKey = '';
    status(`model: ${f.name}`);
  });
  const scanInput = fileInput('.nii,.nii.gz,.gz', (f) => void onScan(f));
  card.append(onnxInput, scanInput);

  modelSel.addEventListener('change', () => {
    if (modelSel.value === 'import') {
      onnxInput.click();
      modelSel.value = 'bundled'; // selection is just a trigger
    } else {
      modelSrc = DEFAULT_MODEL;
      loadedKey = '';
    }
  });
  (card.querySelector('#cv-scan') as HTMLElement).addEventListener('click', () => scanInput.click());

  async function ensureModel(): Promise<void> {
    const key = typeof modelSrc === 'string' ? modelSrc : `bytes:${modelSrc.length}`;
    if (loadedKey === key) return;
    status('loading model…');
    await seg.load(modelSrc);
    loadedKey = key;
  }

  async function onScan(file: File): Promise<void> {
    showSpinner();
    try {
      await ensureModel();
      status('reading scan…');
      const v = await readNifti(file);
      status(`segmenting ${v.d} slices…`);
      const masks = await seg.segmentVolume(v.data, v.d, v.h, v.w, v.spacingYX);
      viewer.showStatic(chamberPolys(masks, v.zSpacing));
      const vox = TARGET_MM * TARGET_MM * v.zSpacing; // mm³ per voxel (resampled in-plane)
      const cav = volumeMl(countLabel(masks, 3), vox);
      const myo = volumeMl(countLabel(masks, 2), vox);
      const rv = volumeMl(countLabel(masks, 1), vox);
      vols.innerHTML =
        `<div style="color:#fff;">${file.name}</div>` +
        `<div><span style="color:#ef5350;">LV cav</span> ${fmtMl(cav)} · ` +
        `<span style="color:#ffca5b;">myo</span> ${fmtMl(myo)} · ` +
        `<span style="color:#5b8def;">RV</span> ${fmtMl(rv)}</div>` +
        `<div style="color:#8aa0b6;">prediction · single volume (no EF — needs ED+ES)</div>`;
      status('done');
    } catch (e: any) {
      status(`error: ${e?.message ?? e}`);
    } finally {
      hideSpinner();
    }
  }
}

function fileInput(accept: string, onPick: (f: File) => void): HTMLInputElement {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = accept;
  input.style.display = 'none';
  input.addEventListener('change', () => {
    const f = input.files?.[0];
    if (f) onPick(f);
    input.value = ''; // allow re-picking the same file
  });
  return input;
}
