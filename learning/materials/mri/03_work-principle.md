# 03 · Work principle — how an image is actually made

Goal: locate the signal in 3D. The trick is to make **position determine the signal's
frequency and phase**, using the gradient coils as "rulers." Three gradient **jobs**
(same x/y/z coils, used at three different times):

1. **Slice select** — pick the plane.
2. **Phase encode** — locate one in-plane axis.
3. **Frequency encode (readout)** — locate the other in-plane axis.

## 1. Slice selection (pick the plane)
- Turn on a gradient (say z) → Larmor frequency now **varies along z**.
- Fire an **RF pulse with a narrow frequency band** → only protons whose local
  frequency matches are at **resonance** → **only that one slab tips.**
- **Key geometry:** the field varies along the gradient direction **g**, so all points
  at the same value of g share the same frequency → they form a **flat plane
  perpendicular to g**. RF *center frequency* picks **which** plane; RF *bandwidth* (+
  gradient strength) picks **thickness**. Combine x/y/z coils → **g in any direction**
  → **oblique / double-oblique** slices (needed for the tilted heart).
- A **sinc-shaped** RF pulse gives a clean rectangular slab. Multi-slice imaging
  **interleaves** slices (and leaves small gaps) to avoid cross-talk; **multiband/SMS**
  excites several slices at once (separated later by multiple receive coils).
- The slice gradient is **on during the RF**, then **off**.

## 2. The signal & the echo (how protons "answer")
- The RF tips M into the transverse plane; M then **precesses** at the Larmor
  frequency. A **rotating magnetization is a tiny rotating magnet**, and a rotating
  magnet next to the coil **induces a voltage** in it (Faraday's law — a **dynamo**).
  **Not** a reflection, and **nothing crosses the air**: the field is already in
  space; its *change* (from the spinning) pushes the coil's own electrons.
- **Only spinning (transverse) magnetization induces signal.** Still magnetization
  (along B₀) = constant field = no change = no signal. (Same reason B₀ itself induces
  nothing — it's constant.)
- Right after the pulse, all spins are **in phase** → strong signal (the **FID**),
  which then **decays as they dephase** (T2 = irreversible; **T2\*** = extra dephasing
  from field inhomogeneity, which is **reversible**).
- **The echo:** a **180° pulse** reverses the reversible (T2\*) dephasing — like
  runners who spread out, then all turn around and re-cross the start line together —
  so the signal **re-converges into an echo at time TE**, read at a moment you choose.
  Spin echo measures **true T2**. The faster cousin, **gradient echo** (refocus with a
  gradient instead of a 180°), stays T2\*-weighted and is what fast/**cine cardiac**
  imaging uses. **TR** = time between excitations; **TE** = pulse→echo readout time.

## 3. In-plane encoding (find position within the plane)
One read with the slice gradient off = the coil hears **one note = the sum of the
whole plane**: no "where." You fix that by making position change the signal, then
decoding with Fourier. A gradient is a **single ruler** — it labels **one axis at a
time**, and **only while it's on**, so you use it twice, two ways:

- **Frequency encode (readout):** turn a gradient (x) **ON during sampling** → each
  column spins at a **different pitch** → the recorded waveform is a **mix of pitches**
  → **Fourier splits them** → whole **x-axis in one read.** Cheap.
- **Phase encode:** *before* the read, briefly pulse a gradient (y) → each row gets a
  **position-dependent phase head-start**, then off → all rows back to the same pitch
  but **phase-stamped** by y. One read can't unmix them, so you **repeat** the whole
  cycle, stepping the phase strength each time. To resolve **N rows you need N steps**;
  the adjacent-row phase increment per step is **360°/N** (the discrete Fourier basis).

**Why the slice gradient is off during the read:** it points along the slice axis
(z), which is useless now (slice already picked, and it's thin). You **re-aim the
ruler** to an in-plane axis. Two live gradients at once would blend axes ambiguously —
one ruler, one axis at a time.

**Why it's predictable despite quantum randomness:** you encode the **net
magnetization** of each region (deterministic, Bloch), not chaotic individual protons.
A gradient G applied for time t rotates a region's net vector by a known angle
`γ·G·y·t` ∝ position.

## The full chain (one line of data)
1. **Slice-select gradient ON + RF** → tip the plane.
2. Slice-select **OFF**.
3. **Phase-encode pulse** (one strength) → stamp the y-axis → off.
4. **Readout gradient ON + sample** → one waveform → (after Fourier) one x-line, with
   this read's y-stamp.
5. Wait **TR**, repeat from 1 with a **different phase-encode strength** → next line.

Do this for all phase steps → the data grid (**k-space**, see
[04_k-space.md](04_k-space.md)) → 2D Fourier → image (see
[05_to-image.md](05_to-image.md)).

## Timing, and why the heart is special (bridge to A2)
- One k-space line per **TR**. Full image = N_y lines × TR. Fast bSSFP (TR≈3 ms) →
  ~0.8 s; slow spin-echo (TR≈600 ms) → ~2.5 min.
- The heart moves a lot in ~0.6 s → naive acquisition blurs. Solution: exploit that
  the heartbeat is **periodic**. **ECG-gate** to the R-wave; grab a **few lines per
  beat** at a given cardiac phase, repeat at the **same phase across ~10–20 beats**
  (one breath-hold) → one frozen image of that phase. Repeat for ~20–30 phases →
  **cine** movie of one representative beat (each frame stitched from many beats).
- **Retrospective gating** records continuously and afterward bins each line by its
  **fraction of the actual R-R interval**, so it copes with **variable rhythm**.
  Severe **arrhythmia** still breaks the "every beat the same" assumption → artifacts
  → use rejection or **real-time MRI**. (Full A2 writeup to come.)
