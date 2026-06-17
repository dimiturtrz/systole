# MRI — intro & big picture

## Why this lane
The project measures **cardiac function (ejection fraction)** from images. For the
MRI lane, the images come from cardiac MRI. Before segmenting hearts or computing EF,
you have to understand **where the pixels come from** — how an MRI machine turns
protons in the body into a 2D image, and why cardiac imaging is special. That's what
these materials cover (the "A" phase of the curriculum).

## The one-paragraph picture
A giant superconducting magnet aligns the hydrogen protons in your body. A
radio-frequency pulse, tuned to resonance, tips them so they spin sideways; spinning
protons act like tiny rotating magnets and **induce a signal in a nearby coil**.
Magnetic-field **gradients** make the signal's frequency and phase depend on
**position**, so the recorded signal encodes *where* things are. The raw recordings
fill a grid called **k-space**, and a **2D Fourier transform** turns k-space into the
image. The heart moves, so cardiac MRI **synchronizes to the ECG** and stitches the
data together across many heartbeats.

## The five pieces (and why in this order)
1. **Machine physics** — the hardware that makes and shapes the fields (magnet,
   gradients, RF, cooling). You can't follow the method without knowing the parts.
2. **Proton physics** — why hydrogen, what "spin" is, resonance, and the two
   relaxation clocks (T1/T2) that create contrast.
3. **Work principle** — the actual method: select a slice, encode position, get the
   signal back (the echo), repeat. This is the core.
4. **k-space** — what the pile of raw recordings *is* (the Fourier domain of the image).
5. **To image** — how the Fourier transform assembles k-space into a picture, plus
   why timing makes the moving heart hard.

## How to use these
Read top to bottom; each builds on the last. They're written as explanations, not
reference cards — the goal is *understanding*, so the "why" is foregrounded. When a
piece clicks, ask to be **quizzed** on it; wrong answers get logged honestly (the
ramp is real). External videos/reading are in [orientation.md](orientation.md); the
ordered plan with status is in [curriculum.md](curriculum.md).

**Honest scope:** this is a deliberate ramp into medical imaging from an
audio/signal-ML background. The DSP parallels (Fourier, sampling, Nyquist, I/Q) are
leaned on heavily because they transfer directly; the clinical/anatomical specifics
are being learned.
