// Sync exported cardioview assets from the out-of-repo home into the gitignored public/ dirs that
// vite serves. Runs on predev/prebuild (and standalone via `npm run sync`).
//
// Single external home = <data>/meshes/cardioview/  (export_web.py writes it: glb/manifest/slices +
// models/<name>.onnx). Its path is owned by core.config — resolved here via one python call (single
// source, no duplicated path logic), overridable with $CARDIOVIEW_ASSETS. Never fatal: if assets
// aren't there yet (fresh clone, no export run), dev still works in import-only mode.
import { execSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, readdirSync, rmSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const webRoot = resolve(here, '..');
const repoRoot = resolve(webRoot, '..', '..');

function assetsPath() {
  if (process.env.CARDIOVIEW_ASSETS) return process.env.CARDIOVIEW_ASSETS;
  try {
    const root = execSync(
      'uv run python -c "from core.config import data_root; print(data_root(\'meshes\'))"',
      { cwd: repoRoot, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] },
    ).trim();
    return root ? join(root, 'cardioview') : '';
  } catch {
    return ''; // no uv/python (e.g. pure web deploy) — rely on already-built dist assets
  }
}

const src = assetsPath();
if (!src || !existsSync(src)) {
  console.warn(`[sync-assets] no external assets at "${src || '(unresolved)'}" — skipping ` +
    `(set $CARDIOVIEW_ASSETS or run \`python -m cardioview.export_web\`; import-only dev still works).`);
  process.exit(0);
}

// public/{data,models} are a derived cache — mirror the external home exactly (drop stale orphans).
// data payload = everything except the models/ subdir -> public/data
const dataDst = join(webRoot, 'public', 'data');
rmSync(dataDst, { recursive: true, force: true });
mkdirSync(dataDst, { recursive: true });
for (const e of readdirSync(src, { withFileTypes: true })) {
  if (e.isDirectory() && e.name === 'models') continue;
  cpSync(join(src, e.name), join(dataDst, e.name), { recursive: true });
}

// models -> public/models
const modelsSrc = join(src, 'models');
if (existsSync(modelsSrc)) {
  const modelsDst = join(webRoot, 'public', 'models');
  rmSync(modelsDst, { recursive: true, force: true });
  mkdirSync(modelsDst, { recursive: true });
  cpSync(modelsSrc, modelsDst, { recursive: true });
}

console.log(`[sync-assets] ${src} -> public/{data,models}`);
