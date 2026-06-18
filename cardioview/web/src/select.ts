import type { HeartEntry } from './manifest';

/** Default heart to open on: a genuinely normal (NOR group) one, else the highest-EF. */
export function pickDefault(entries: HeartEntry[]): HeartEntry {
  return (
    entries.find((e) => e.group === 'NOR') ??
    entries.reduce((a, b) => (b.pred.ef > a.pred.ef ? b : a), entries[0])
  );
}
