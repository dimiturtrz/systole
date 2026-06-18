// Chamber definitions + color classification — pure, unit-tested. glTF bakes each chamber's
// color; we recover which chamber an imported mesh is by that color (opacity is dropped by
// glTF, so it's re-applied here).

export interface Chamber {
  name: string;
  color: [number, number, number];
  opacity: number;
}

export const CHAMBERS: Chamber[] = [
  { name: 'LV cavity', color: [0.94, 0.33, 0.31], opacity: 1.0 }, // slot 0 — red
  { name: 'myocardium', color: [1.0, 0.79, 0.36], opacity: 0.32 }, // slot 1 — gold (translucent)
  { name: 'RV cavity', color: [0.36, 0.56, 0.93], opacity: 1.0 }, // slot 2 — blue
];

/** Map an imported mesh's baked color to a chamber slot (0/1/2), or -1 if unrecognized. */
export function chamberSlot(r: number, g: number, b: number): number {
  if (r > 0.6 && g < 0.5 && b < 0.5) return 0; // LV cavity (red)
  if (r > 0.6 && g > 0.55 && b < 0.55) return 1; // myocardium (gold)
  if (b > 0.55 && b > r) return 2; // RV cavity (blue)
  return -1;
}
