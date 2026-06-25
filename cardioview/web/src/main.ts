import { HeartViewer } from './viewer';
import { SliceView } from './sliceview';
import { loadManifest } from './manifest';
import { mountPanel } from './panel';

const viewer = new HeartViewer();
const sliceView = new SliceView();
let hearts: Awaited<ReturnType<typeof loadManifest>>['hearts'] = [];
let model = 'gen'; // fallback only; the real name comes from the manifest below
try {
  const m = await loadManifest();
  hearts = m.hearts;
  model = m.model;
} catch {
  /* no manifest is fine — import still works */
}
mountPanel(hearts, viewer, sliceView, model);
