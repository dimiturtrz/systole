/** Minimal DOM overlay: a stack of labeled sliders. View-layer UI, no physics. */
export interface SliderSpec {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  fmt?: (v: number) => string;
  onChange: (v: number) => void;
}

export function mountControls(specs: SliderSpec[]): void {
  const wrap = document.createElement('div');
  wrap.style.cssText =
    'position:fixed;top:12px;left:12px;z-index:10;background:rgba(20,24,30,.82);' +
    'color:#cdd6e0;font:13px system-ui,sans-serif;padding:8px 12px;border-radius:8px;' +
    'border:1px solid #2a323d;user-select:none;display:flex;flex-direction:column;gap:8px;';

  for (const spec of specs) {
    const fmt = spec.fmt ?? ((v: number) => String(v));
    const row = document.createElement('div');
    const label = document.createElement('label');
    label.textContent = `${spec.label}: ${fmt(spec.value)}`;
    label.style.cssText = 'display:block;margin-bottom:4px;';
    const input = document.createElement('input');
    input.type = 'range';
    input.min = String(spec.min);
    input.max = String(spec.max);
    input.step = String(spec.step);
    input.value = String(spec.value);
    input.style.width = '180px';
    input.addEventListener('input', () => {
      const v = parseFloat(input.value);
      label.textContent = `${spec.label}: ${fmt(v)}`;
      spec.onChange(v);
    });
    row.appendChild(label);
    row.appendChild(input);
    wrap.appendChild(row);
  }
  document.body.appendChild(wrap);
}
