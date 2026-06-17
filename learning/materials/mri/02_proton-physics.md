# 02 · Proton physics — what we're imaging

## Spin is not electron-only
**Spin** is an intrinsic property of most particles. Electrons, protons, neutrons are
all **spin-½**. A hydrogen nucleus **is a single proton**, and that's what MRI images —
hence **N**MR (*nuclear* magnetic resonance).

**Proton vs electron spin:** same spin (½), but a proton's **magnetic moment is ~660×
weaker** (it's ~1836× heavier; magnetism ∝ charge/mass), so it precesses at **MHz**
(~64 MHz at 1.5 T, ~128 MHz at 3 T) vs the electron's GHz.

**Why image protons (hydrogen), not electrons:**
- **Availability:** electrons are almost all **paired** (opposite spins cancel) → most
  tissue has no net electronic spin. Hydrogen nuclei (single, unpaired protons) are
  everywhere — the body is ~60% water.
- **Frequency:** proton resonance is in the convenient, non-ionizing radio band.
- (Nuclei with even numbers of both protons and neutrons have paired spins → **zero
  net spin → MRI-invisible**. ¹H, one lone proton, has spin ½ → visible.)

## In the magnet: Zeeman splitting & equilibrium
- A proton's spin gives it a tiny magnetic moment. In B₀ it has **two energy levels**:
  **aligned (low)** and **anti-aligned (high)**. The gap `ΔE = γℏB₀` is **proportional
  to field strength**, and it corresponds **exactly to the Larmor frequency**
  (`ΔE = h·f`).
- **Equilibrium / "default" state:** each proton **precesses** around B₀ at the Larmor
  frequency, at **random phase**, with a slight **excess in the aligned state**.
  Because phases are random, the sideways components cancel → the net magnetization
  **M points along z** (no transverse part). M is what we manipulate and measure.
- We never track individual protons (quantum, random). We track **M, the net
  magnetization** of each region — it's the sum, and it behaves **classically and
  predictably** (Bloch equations). That predictability is why MRI works.

## Resonance — how a tiny pulse tips M (and why it's safe)
- The RF pulse is a tiny oscillating field **B₁** (microtesla) — it can't overpower B₀
  (teslas). It works by **resonance**: oscillating **exactly at the Larmor frequency**,
  each little push is timed to the precession and **accumulates** — like pushing a
  swing at its natural rhythm. Off-resonance → pushes cancel → nothing.
- (Formally: in the "rotating frame" spinning at the Larmor frequency, B₀'s effect
  vanishes and only B₁ remains, slowly rotating M. Same as the swing picture.)
- **Energy scales (why it's safe):** an RF photon at 64 MHz is ~**2.6×10⁻⁷ eV** —
  ~100,000× too weak to break even a hydrogen bond (~0.2 eV), ~18,000,000× too weak
  for a covalent bond (~4.8 eV). MRI RF **cannot break bonds or ionize** — it only
  reorients nuclear spins. **Non-ionizing.** Its only tissue effect is slight heating
  (monitored as SAR). (Contrast: X-ray/CT photons carry keV–MeV → ionizing.)

## T1 and T2 — the contrast clocks
After the pulse, two **different** relaxations happen at once:
- **T1 (longitudinal recovery):** M regrows back along z. Usually the **slower** one
  (hundreds of ms–seconds).
- **T2 (transverse decay):** the sideways signal dies as spins **dephase** (lose phase
  sync). Usually **faster** (tens–hundreds of ms). Always **T2 ≤ T1**.

Every tissue has **both** a T1 and a T2 (plus a proton density). The image is a
**blend** of all three; you choose where on that spectrum to sit with the **timing
knobs**:
- **T1-weighted:** short TR, short TE.
- **T2-weighted:** long TR, long TE.
- **Proton-density-weighted:** long TR, short TE.

(TR = time between excitations; TE = time from pulse to readout. Defined more in
[03_work-principle.md](03_work-principle.md).)
