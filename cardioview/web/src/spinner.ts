// Centered loading spinner overlay, shown while glTF frames load.

let el: HTMLDivElement | null = null;
let depth = 0; // ref-count so overlapping loads don't hide early

function ensure(): HTMLDivElement {
  if (el) return el;
  const style = document.createElement('style');
  style.textContent = '@keyframes cv-spin{to{transform:rotate(360deg)}}';
  document.head.appendChild(style);
  el = document.createElement('div');
  el.style.cssText =
    'position:fixed;inset:0;z-index:20;display:none;align-items:center;justify-content:center;' +
    'pointer-events:none;flex-direction:column;gap:12px;';
  el.innerHTML =
    '<div style="width:46px;height:46px;border:4px solid #2a323d;border-top-color:#5b8def;' +
    'border-radius:50%;animation:cv-spin 0.9s linear infinite;"></div>' +
    '<div style="color:#8aa0b6;font:13px system-ui,sans-serif;">loading heart…</div>';
  document.body.appendChild(el);
  return el;
}

export function showSpinner(): void {
  depth++;
  ensure().style.display = 'flex';
}

export function hideSpinner(): void {
  depth = Math.max(0, depth - 1);
  if (depth === 0 && el) el.style.display = 'none';
}
