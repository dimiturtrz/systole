// Legend overlay (top-right). View-only — explains what the scene encodes. Colors mirror
// SpinScene (transverse-mag ramp) and Presenter.freqColor (Larmor gradient) so they stay honest.

interface LegendRow {
  swatch: string; // CSS background for the chip
  text: string;
}

const ROWS: LegendRow[] = [
  { swatch: 'linear-gradient(90deg,#4d73d9,#ffa626)', text: 'spin tilt: <b>rest</b> → <b>tipped</b> (transverse signal)' },
  { swatch: 'linear-gradient(90deg,#0059ff,#805980,#ff5900)', text: 'gradient on: <b>low</b> ↔ <b>high</b> Larmor (position along the active gradient)' },
  { swatch: '#66e6ff', text: '<b>slice plane</b> = RF transmit slab — flashes as it excites' },
  { swatch: '#80808c', text: '<b>B₀</b> main-field axis' },
];

export function mountLegend(): void {
  const wrap = document.createElement('div');
  wrap.style.cssText =
    'position:fixed;top:12px;right:12px;z-index:10;background:rgba(20,24,30,.82);' +
    'color:#cdd6e0;font:12px system-ui,sans-serif;padding:8px 12px;border-radius:8px;' +
    'border:1px solid #2a323d;user-select:none;max-width:260px;display:flex;flex-direction:column;gap:6px;';

  const title = document.createElement('div');
  title.textContent = 'legend';
  title.style.cssText = 'color:#9aa7b8;margin-bottom:2px;';
  wrap.appendChild(title);

  const arrow = document.createElement('div');
  arrow.style.cssText = 'display:flex;align-items:center;gap:8px;';
  arrow.innerHTML = '<span style="color:#7a9bff;font-size:15px;">↑•</span><span>arrow = a proton spin (dot = tail)</span>';
  wrap.appendChild(arrow);

  for (const r of ROWS) {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;align-items:center;gap:8px;';
    const chip = document.createElement('span');
    chip.style.cssText = `flex:0 0 26px;height:12px;border-radius:3px;background:${r.swatch};border:1px solid #3a424d;`;
    const label = document.createElement('span');
    label.innerHTML = r.text;
    row.appendChild(chip);
    row.appendChild(label);
    wrap.appendChild(row);
  }
  document.body.appendChild(wrap);
}
