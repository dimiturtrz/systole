# 05 · From k-space to image — the Fourier transform

## Fourier transform, the core idea
**Any signal can be built by adding up pure waves** — each with an amplitude and a
phase. The Fourier transform finds the **recipe**: how much of each frequency is in the
signal.
- **Forward:** signal → recipe (amount + phase per frequency).
- **Inverse:** recipe → signal (scale each wave by its amount, **add them all up**).

Audio version: a chord = sum of pure tones; FT tells you which notes & how loud;
inverse plays them together → the chord back.

**How it extracts one frequency:** multiply the signal by a **test wave** of that
frequency and **sum**. If present, they line up → big number; if absent, it averages
to **zero** (cancels). So FT = *correlate against each test wave; strong match = that
frequency is present.* (Same "only the match responds" logic as resonance.) The **FFT**
is just a fast algorithm to do this (N·log N).

## In 2D — images are sums of stripe patterns
For an image, the building blocks are **2D stripe patterns**: brightness waving across
the picture at some spacing and orientation.
- **Low spatial frequency** = broad, slow variation → shapes, contrast.
- **High spatial frequency** = fine, rapid variation → edges, detail.

Any image = a **sum of thousands of these stripe patterns**, each weighted.

## k-space → image
k-space **is** that recipe (see [04_k-space.md](04_k-space.md)). Each cell = one stripe
pattern; its position gives the stripe's **fineness + orientation**, its complex value
gives **amount (magnitude) + positioning (phase)**.

**Inverse 2D Fourier transform = assemble the recipe:** take every stripe pattern,
**scale** it by its k-space magnitude, **shift** it by its k-space phase, and **add
them all together.** The pile-up **is** the image. (2D FFT is separable: 1D FFT every
row, then every column.)

## Why phase is non-negotiable
A sine and a cosine of the same frequency have equal strength but different **phase** —
which puts their bright bands in different **places**. Magnitude = *how strong*; phase
= *where features land*. Strip the phase and you keep the energy but lose the
arrangement → no recognizable image. (Classic demo: swap two photos' magnitude and
phase — the result looks like whichever photo the **phase** came from.) This is why
MRI insists on **complex (I/Q)** data.

## What the pixel values mean
Each output pixel's brightness = the signal from that (x,y) location = **proton
density × T1/T2 weighting** (set by TR/TE). Bright = lots of favorable-relaxation
protons there.

## One-paragraph summary
k-space is a grid of little vectors; each is a recipe entry — "add this much of one
stripe pattern, positioned by this phase." Center = broad shapes/contrast, edges =
fine detail. The **inverse 2D Fourier transform** overlays all the weighted stripe
patterns, and the sum **is** the image. The FFT is just the fast way to compute "how
much of each pattern" (forward) or "add them back up" (inverse), via the
multiply-by-a-test-wave-and-sum trick.

---
*Acquisition timing (TR, scan time) and the moving-heart problem (ECG gating, cine)
are covered at the end of [03_work-principle.md](03_work-principle.md), bridging into
the cardiac (A2) materials.*
