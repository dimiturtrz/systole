# 01 · Machine physics — the scanner

The active, programmable parts are the **gradient coils** and the **RF coil**. The
**main magnet** and **shim coils** are static. Everything else is structure.

## The main field B₀
- **B₀** = the main static magnetic field. Symbol `B`, subscript `0` = the baseline.
  Strength **1.5 T or 3 T** — about **60,000× Earth's field**. On 24/7.
- It's made by a **superconducting solenoid**: superconducting wire wound in many
  loops around the bore. The **winding geometry** makes the field point straight down
  the bore axis (head-to-foot, conventionally **z**) and be uniform inside. Direction
  follows the current's circulation (right-hand rule).
- **Persistent current.** Superconductors have **zero electrical resistance** when
  cold, so once you ramp current into the loop it **circulates forever with no power
  supply and no heat** (`I²R = 0`). The field is a normal electromagnet field from
  that current; the energy just **sits stored in the field**. Analogy: a frictionless
  flywheel — no motor needed because nothing drains it.
- **Steady current → steady field.** B₀ does **not** rotate or oscillate, even though
  the current never stops flowing. The things that move are the RF pulse, the
  gradients, and the precessing protons — not B₀.

## Cooling, and turning it off
- The magnet sits in a **cryostat** of **liquid helium (~4 K, −269 °C)** — that's what
  keeps the wire superconducting. Modern scanners use a **cryocooler** (cold head +
  compressor) that re-condenses helium → little/no refilling, *as long as it runs*.
- **There's no on/off switch.** Two ways to remove the field:
  - **Ramp down** (planned): reconnect a supply, drive the current back down over ~an
    hour. For servicing/moving.
  - **Quench** (emergency): force part of the wire out of superconductivity → it
    gains resistance → the huge current dies in seconds, dumping energy as heat →
    helium boils off and vents. Fast but expensive. Used in real emergencies (e.g. a
    ferromagnetic object pins someone to the bore).

## Gradient coils (x, y, z) — the position rulers
- Three coil sets. Each makes the **field vary linearly along its own axis**:
  total field `B(x) = B₀ + G·x`. The **linear ramp shape is built into the winding**;
  the only thing you *program* is **G** (the slope), via the **current** — one number
  per coil, per moment.
- Combine currents in several coils → one **gradient vector** pointing in any
  direction → lets you tilt encoding to **any oblique plane** (key for the heart).
- They switch on/off fast and forcefully → **the loud banging** you hear.

## RF coil (transceiver) — excite & listen
- Transmits the **radio-frequency pulse** (to tip protons) and **receives** the
  signal they induce back. Often split: a big **body coil** transmits, small **local
  coils** placed on the patient receive (closer = better signal).
- It's an **antenna = coil** — same thing. "Transceiver" = transmit + receive.

## Shim coils
- Fine-tune B₀ to be as **uniform** as possible (correcting magnet imperfections).

## The "sandwich" (patient → outward)
1. **Bore liner** — cosmetic/protective cover (no imaging role).
2. **RF coil** — innermost active layer (must be near the patient).
3. **Gradient coils** — the position rulers (the banging).
4. **Shim coils** — uniformity.
5. **Main superconducting magnet** — outermost, in the helium **cryostat**.

Why this order: things that interact with the patient sit closest (RF nearest,
gradients spanning the bore); the uniform B₀ fills everything, so its big cold magnet
lives outside.

## Active vs static — and the controller
- **Static:** main magnet + shims (set once).
- **Active/programmable:** gradients + RF — the **pulse sequence**. A real-time
  controller (FPGA/DSP class) generates the precisely-timed waveforms: RF envelope
  (→ DAC → RF amp), three gradient waveforms (→ DACs → gradient amps), and ADC
  sampling — all locked to a **common clock** (which also gives transmit/receive a
  shared phase reference). The scanner is essentially a precision, phase-locked,
  multi-channel arbitrary-waveform generator + digitizer.
