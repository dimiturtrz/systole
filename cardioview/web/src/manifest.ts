import type { Vols } from './metrics';

export interface HeartEntry {
  patient: string;
  group: string; // ACDC pathology group (DCM, HCM, ...)
  held_out: boolean; // was this patient outside the model's training set?
  source: 'pred' | 'gt';
  pred: Vols;
  gt: Vols;
  glb: { ED?: string; ES?: string };
}

const DATA = 'data';

export async function loadManifest(): Promise<HeartEntry[]> {
  const r = await fetch(`${DATA}/manifest.json`);
  if (!r.ok) throw new Error(`manifest fetch failed: ${r.status}`);
  return r.json();
}

export const glbUrl = (file: string): string => `${DATA}/${file}`;
