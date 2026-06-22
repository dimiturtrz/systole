import type { Vols } from './metrics';

export interface HeartEntry {
  patient: string;
  group: string; // ACDC pathology group (DCM, HCM, ...)
  held_out: boolean; // was this patient outside the model's training set?
  source: 'pred' | 'gt';
  pred: Vols;
  gt: Vols;
  glb: { ED?: string; ES?: string };
  frames?: string[]; // beating-cycle glb files, in time order (if animated)
  ed_idx?: number; // index into frames for ED (full)
  es_idx?: number; // index into frames for ES (empty)
  slices?: string[]; // per-frame baked slice strips (PNG: R=gray, G=label), aligned with `frames`
  sliceD?: number; // slices per frame (each strip is sliceD * SIZE tall)
}

const DATA = 'data';

export interface Manifest {
  model: string; // bundled model name -> models/<model>.onnx
  hearts: HeartEntry[];
}

export async function loadManifest(): Promise<Manifest> {
  const r = await fetch(`${DATA}/manifest.json`);
  if (!r.ok) throw new Error(`manifest fetch failed: ${r.status}`);
  const j = await r.json();
  return Array.isArray(j) ? { model: 'acdc_aug', hearts: j } : j; // back-compat with array form
}

export const glbUrl = (file: string): string => `${DATA}/${file}`;
