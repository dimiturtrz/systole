# Data-scarcity signal-test: does physics-synth augmentation pay at low real-data budget?

**Date:** 2026-07-18 · **Scope:** cross-task (synth-as-augmentation, `pwih`/`tmfi`) · **Beads:**
cardiac-seg-tmfi / wqmh (harness) / pgb9 (this test) · **Status:** signal-test decided — Dice lever
refuted, EF lever = real signal, harness confound surfaced.

The cheapest keep/kill for the scarcity-crossover epic (`tmfi`): at a small real-data budget K, does
adding the physics-synth pool (`anatomy_mode=mix`) beat real-only? Two K points (3, 10), two arms each,
quick single-seed. Real-only vs real-K + synth pool, identical ACDC val early-stop, identical frozen
cross-vendor test (Canon+GE+cmrxmotion, n=147). Baselines already banked: full-real ≈ 0.847, zero-real
synth ≈ 0.613.

## Results

| K real subj | real-only TEST Dice | mix TEST Dice | ΔDice | real EF MAE | mix EF MAE |
|-------------|---------------------|---------------|-------|-------------|------------|
| 3           | ≥0.627 (unconverged)| 0.630         | ~0    | 22.9%       | **13.0%**  |
| 10          | 0.735               | 0.717         | −0.018| 19.8%       | **12.3%**  |
| ~495 (full) | ~0.847              | —             | —     | ~3.5%       | —          |
| 0 (synth)   | 0.613               | —             | —     | —           | —          |

## What it says

1. **Dice: synth-aug adds nothing at scarcity — no crossover down to K=3.** At K=10 real-only *beats*
   mix by 0.018 (within the ±0.02–0.03 single-seed floor); at K=3 they tie (0.627 vs 0.630) *and the
   real-only number is a lower bound* (see confound). The scarcity thesis — "when real is scarce, synth
   props up Dice" — **is not supported**: real data is efficient enough that even 3 subjects match the
   synth-mix on segmentation Dice.

2. **Real-data efficiency is the real headline.** K=10 real subjects → **0.735** cross-vendor Dice; K=3
   → **≥0.627**. Both already beat pure zero-real synth (0.613). Cross-vendor Dice saturates in a
   handful of real subjects — 50× more data (K=10→495) buys only 0.735→0.847. This reframes synth's
   value: it does **not** compete with even a tiny real set on Dice; its value is zero-annotation
   domains + the twin, not rescuing a scarce-but-present real set.

3. **EF is the one positive signal.** Synth-aug cuts EF MAE by ~7–10 pp at both K (mix 12–13% vs
   real-only 19.8–22.9%), consistently, even where Dice is flat. Mechanism: the synth pool's anatomical
   / volume diversity regularizes the volume estimate (EF is a volume ratio) where boundary Dice has
   already saturated. Single-seed — needs a seed-confirmed replication before it's hardened, but it is
   the clearest lead here and the clinically-relevant endpoint.

## The confound (a harness finding, filed)

Epoch-based training + early-stop is **not comparable across K**: batches/epoch scales with the train
set. At K=3 the real-only arm sees **1 batch/epoch** (~1 gradient step); the mix arm sees 166. Under
`patience=30` epochs the low-K real arm early-stopped at val 0.204 (ep165) — but rerun with
`patience=120, epochs=900` it climbed monotonically to val 0.632 and **hit the 900-epoch ceiling still
rising, never converged**. So the tiny-batch low-K arms are gradient-**step**-starved; their reported
Dice are lower bounds. A trustworthy scarcity curve needs a **gradient-step budget** (equal total
steps across arms), not an epoch budget. That the lower-bound real-only numbers *already match* mix only
strengthens the Dice verdict, but the exact curve can't be drawn until the harness is step-budgeted.

## Verdict — keep / kill

- **KILLED (as a Dice lever):** the `tmfi` scarcity-crossover thesis for *segmentation Dice*. No synth-
  aug Dice gain > noise at any K tested; real data saturates Dice by K≤3–10. The Dice-curve children
  (`6ui5`/`hlqh`) rest on a premise this refutes — deprioritize unless the step-budget harness reopens
  a lower-K regime.
- **KEPT (real lead):** synth-aug **EF** benefit at scarcity (−7 to −10 pp, consistent across K) →
  reframe `tmfi` from "Dice crossover" to "synth-aug regularizes EF under scarcity"; confirm with seeds.
- **KEPT (infra finding):** step-budgeted training is a prerequisite for any honest cross-K comparison
  (filed as a wqmh follow-up). Reusable `train_subjects` K-cap knob landed + unit-tested (bd wqmh).
- **Bottom line:** even 3–10 real subjects beat zero-real synth on Dice and synth-aug adds no Dice on
  top — but synth-aug meaningfully improves EF at scarcity. Synth's Dice value is unseen/zero-annotation
  domains, not augmenting a scarce real set.

Runs (quick, single-seed, gitignored): `runs/scar_{real,mix}_k{3,10}`, `runs/scar_real_k3_long`.
