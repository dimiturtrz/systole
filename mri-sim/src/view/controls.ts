/** Minimal DOM overlay: a speed slider. View-layer UI, no physics. */
export function mountSpeedSlider(initial: number, onChange: (v: number) => void): void {
  const wrap = document.createElement('div');
  wrap.style.cssText =
    'position:fixed;top:12px;left:12px;z-index:10;background:rgba(20,24,30,.82);' +
    'color:#cdd6e0;font:13px system-ui,sans-serif;padding:8px 12px;border-radius:8px;' +
    'border:1px solid #2a323d;user-select:none;';

  const label = document.createElement('label');
  label.textContent = `Speed: ${initial.toFixed(2)}×`;
  label.style.cssText = 'display:block;margin-bottom:6px;';

  const input = document.createElement('input');
  input.id = 'speed-slider';
  input.type = 'range';
  input.min = '0.05';
  input.max = '2';
  input.step = '0.05';
  input.value = String(initial);
  input.style.width = '170px';
  input.addEventListener('input', () => {
    const v = parseFloat(input.value);
    label.textContent = `Speed: ${v.toFixed(2)}×`;
    onChange(v);
  });

  wrap.appendChild(label);
  wrap.appendChild(input);
  document.body.appendChild(wrap);
}
