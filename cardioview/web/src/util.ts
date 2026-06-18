/** Wrap an index into [0, n) — handles negatives, so the cycle loops cleanly both ways. */
export function wrapIndex(i: number, n: number): number {
  if (n <= 0) return 0;
  return ((i % n) + n) % n;
}
