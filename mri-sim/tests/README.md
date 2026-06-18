# Tests — the pyramid

```
        e2e / visual smoke     scripts/shoot.mjs (puppeteer)   ← thin, MANUAL, not in CI
      integration              tests/integration/*.test.ts     ← vitest, headless
   unit                        tests/unit/*.test.ts            ← vitest, headless (base)
```

## unit/ (vitest, no browser)
Model pieces in isolation — `SpinSystem`, later `Simulator` (precession, RF tip,
T1/T2), FFT helper. Fast, deterministic. **Physics correctness lives here.**
`npm test`

## integration/ (vitest, no browser)
Engine pieces working together — chiefly the **honest roundtrip**:
phantom → spins accrue gradient phase → k-space (Σ density·e^{iφ}) → inverse FFT →
**recovers the phantom** within tolerance. Also presenter↔model via a **mock view**
(needs the small DI refactor: presenter depends on a `SpinView` interface, real
vtk `SpinScene` injected in the app, fake injected in tests). Added at M1/M4.
`npm test`

## e2e / visual smoke (puppeteer — manual)
`npm run smoke` → loads the dev server headless, captures console + a screenshot
(`debug-shot.png`). Catches **render** bugs a unit test can't (e.g. invisible
glyphs). Requires `npm run dev` running. Kept manual — not worth automating in CI.

---
Why split: the M0 bug was *rendering* (invisible glyphs) → only a screenshot caught
it. Engine bugs (physics/FFT) are caught far faster + cheaper by unit/integration
tests. Don't reach for puppeteer to test the engine.
