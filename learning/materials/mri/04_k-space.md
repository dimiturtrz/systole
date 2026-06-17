# 04 · k-space — the raw data grid

## What it is
Every read (one phase-encode step) produces one **line** of a 2D grid called
**k-space**. You stack all the lines:
- **along a line** (the readout / frequency direction) = **k_x** info,
- **across the lines** (the phase-encode steps) = **k_y** info.

Each cell holds a **complex value** (magnitude + phase) — MRI keeps phase because
**phase is where position lives** (that's why acquisition is **complex I/Q**, not real).

**k-space is the 2D Fourier transform of the image.** It is *not* the image, and *not*
a spectrogram (see below) — it's the **recipe** the image is built from.

## How to read it (the geometry)
Each k-space cell corresponds to **one 2D "stripe pattern"** (a spatial-frequency
component) in the image:
- **Distance from the center** = how **fine** the stripes are → near center = broad,
  slow variations (overall shapes & contrast); far out = tight stripes (edges, fine
  detail). *That's why the center is always the brightest part of k-space — every image
  is mostly broad shapes.*
- **Angle from center** = the **orientation** of the stripes.
- **Complex value** = **how much** of that stripe pattern (magnitude) and **where its
  bright bands sit** (phase).

So don't read k-space as a picture — it's a **parts list**: "use this much of this
stripe pattern, shifted here."

## Resolution & FOV — a Fourier duality
The two domains are **inversely linked**, two separate knobs:
- **How far out you sample (k_max — the outer lines)** ↔ **image resolution** (fine
  detail). Skip the outer lines → blurrier but faster.
- **How finely you space samples (Δk)** ↔ **field of view (FOV)**. Space them too far
  apart (undersample) → FOV shrinks → **wraparound aliasing** (the Nyquist limit).

Practical reads:
- **Number of lines (matrix size)** = number of pixels / resolution; image needn't be
  square (N_x and N_y independent; **N_y drives scan time**).
- **Center of k-space** → contrast & shapes; **edges** → sharp detail.

## DSP parallels (for the audio-minded)
- The receive chain rhymes with **PDM→PCM**: oversampling sigma-delta ADC + decimation
  — but MRI samples are **complex (I/Q)**, demodulated from the Larmor carrier.
- **Within one read:** a genuine **time-domain waveform**, sampled N_x times, **FFT →
  N_x positions**. Pure acoustics. Sample count ↔ resolution; sample rate ↔ FOV.
- **Across reads:** each repetition adds one sample along **k_y**. So you're sampling a
  2D frequency space: N_x along k_x (within a read) × N_y along k_y (across reads).
- **Nyquist carries over:** undersample → aliasing, which in MRI shows as the image
  **wrapping around** (fold-over).

## NOT a spectrogram
Tempting (both are 2D Fourier-ish), but different:

| | Spectrogram | k-space |
|---|---|---|
| Axes | time × frequency | spatial-freq × spatial-freq (both Fourier) |
| Values | magnitude (phase discarded) | **complex** (phase kept — it's essential) |
| Read directly? | yes ("which freq, when") | no — must **inverse-2D-FFT** to an image |
| Time axis? | yes | **none** |

k-space has **no time axis**; the whole multi-minute acquisition collapses into **one
static 2D image**. (Want time back → repeat per cardiac phase = **cine**.)

→ Turning this grid into the picture: [05_to-image.md](05_to-image.md).
