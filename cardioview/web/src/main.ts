import { HeartViewer } from './viewer';
import { loadManifest } from './manifest';
import { mountPanel } from './panel';

const viewer = new HeartViewer();
const entries = await loadManifest();
if (entries.length === 0) {
  document.body.insertAdjacentHTML('beforeend',
    '<div style="color:#cdd6e0;padding:20px;">No hearts in manifest. Run cardioview/export_web.py.</div>');
} else {
  mountPanel(entries, viewer);
}
