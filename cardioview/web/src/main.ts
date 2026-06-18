import { HeartViewer } from './viewer';
import { loadManifest } from './manifest';
import { mountPanel } from './panel';

const viewer = new HeartViewer();
let entries: Awaited<ReturnType<typeof loadManifest>> = [];
try {
  entries = await loadManifest();
} catch {
  /* no manifest is fine — import still works */
}
mountPanel(entries, viewer);
