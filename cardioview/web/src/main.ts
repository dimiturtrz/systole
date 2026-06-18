import { HeartViewer } from './viewer';
import { loadManifest } from './manifest';
import { mountPanel } from './panel';
import { mountImport } from './importPanel';

const viewer = new HeartViewer();
mountImport(viewer); // upload-your-own-scan path (independent of the canned hearts)

try {
  const entries = await loadManifest();
  if (entries.length > 0) mountPanel(entries, viewer);
} catch {
  /* no manifest is fine — the import panel still works */
}
