# ATRAC1 Encoder/Decoder — Mathematical Reference

## Overview

ATRAC1 compresses 512 PCM samples into 212 bytes (1696 bits) per frame.
Sample rate: 44100 Hz → each frame = 11.6 ms of audio.
Compression ratio: 512 × 2 bytes = 1024 bytes → 212 bytes ≈ **5:1**.

---

## Encoding Pipeline

```
PCM[512] → QMF Analysis → MDCT → Scale → Quantize → Bitstream (212 bytes)
```

### Step 1: QMF Analysis (Subband Decomposition)

The 512-sample frame is split into 3 frequency bands using Quadrature Mirror Filters:

```
         QMF₁ (512→256+256)
        /                    \
   mid+low                   high (256 samples)
      |
   QMF₂ (256→128+128)
  /              \
low (128)      mid (128)
```

**Math**: The QMF is a 48-tap polyphase filter. For each pair of output samples:

```
lower[j/2] = Σᵢ₌₀²³ w[2i] · x[47+j-2i]  +  Σᵢ₌₀²³ w[2i+1] · x[46+j-2i]
upper[j/2] = Σᵢ₌₀²³ w[2i] · x[47+j-2i]  -  Σᵢ₌₀²³ w[2i+1] · x[46+j-2i]
```

where `w[48]` are the QMF window coefficients and `x` includes 46 samples of history.

**Result**: low[128], mid[128], hi[256]

---

### Step 2: Transient Detection

For each band, detect sudden energy changes to decide window mode:

```
E[k] = 19 · log₁₀(RMS(HP(band[k·S .. (k+1)·S])))
```

where `HP` is a 21-tap high-pass FIR filter and `S` is the short block size.

**Decision**:
- If `E[k] - E[k-1] > 16 dB` (rising) → **transient detected**
- If `E[k-1] - E[k] > 20 dB` (falling) → **transient detected**
- Transient → use **short windows** (multiple small MDCTs)
- No transient → use **long window** (one large MDCT)

---

### Step 3: MDCT (Time → Frequency)

Each band is transformed from time-domain to frequency-domain:

```
X[k] = Σₙ₌₀²ᴺ⁻¹ x[n] · cos(π/N · (n + ½ + N/2) · (k + ½))
```

**Window sizes** (long mode):
- Low band: 128 samples → MDCT(256) → 128 coefficients
- Mid band: 128 samples → MDCT(256) → 128 coefficients
- High band: 256 samples → MDCT(512) → 256 coefficients

**Short mode**: Multiple 64-point MDCTs per band.

**Total**: Always 512 spectral coefficients.

The implementation uses an FFT-based fast algorithm:
1. Pre-rotate input with twiddle factors: `twiddle[i] = √(scale/N) · e^(j·(ω·i + α))`
2. N/4-point complex FFT
3. Post-rotate output with same twiddles

---

### Step 4: Scaling (Normalization)

The 512 coefficients are divided into **52 BFUs** (Basic Functional Units) of varying sizes.

For each BFU `b`:
1. Find `max_abs = max(|X[i]|)` over all coefficients in the BFU
2. Find the smallest scale factor `SF[b]` from a table of 64 values where `SF[b] ≥ max_abs`:
   ```
   ScaleTable[i] = 2^(i/3 - 21),  i = 0..63
   ```
3. Normalize: `V[i] = X[i] / SF[b]`, clamped to `[-0.99999, 0.99999]`
4. Store the 6-bit scale factor index `sf_idx[b]`

---

### Step 5: Psychoacoustic Model

Determines how many bits each BFU deserves based on human hearing:

**Absolute Threshold of Hearing (ATH)**: Below this level, sounds are inaudible.
```
ATH(f) ≈ lookup_table(f) - 100 - f²·0.015  [dB SPL]
```

**Scale Factor Spread**: Measures tonal vs noise character.
```
σ = stddev(sf_idx[0..N]) / N
spread = min(σ, 14) / 14    ∈ [0, 1]
```
- spread ≈ 1 → tonal (concentrated energy, needs more bits)
- spread ≈ 0 → noisy (spread energy, needs fewer bits)

**Loudness Tracking**: Exponential moving average.
```
L[t] = 0.98 · L[t-1] + 0.01 · (L_left + L_right)
```

---

### Step 6: Bit Allocation

Each BFU gets a **word length** (0-15 bits per coefficient). Total bits must fit in 1696 bits.

```
bits[b] = FixedTable[b] + adjust(spread, shift, ATH[b], loudness)
```

The `shift` parameter is found by **binary search** to hit the target bit budget:
```
while (max_shift - min_shift > 0.1):
    shift = (max_shift + min_shift) / 2
    total = Σ bits[b] · specs_per_block[b]
    if total > target: increase shift (fewer bits)
    else: decrease shift (more bits)
```

A **BitsBooster** then redistributes spare bits to important BFUs.

---

### Step 7: Quantization

Each scaled coefficient is quantized to an integer mantissa:

```
mantissa[i] = round_ties_even(V[i] · (2^(wordlen-1) - 1))
```

**Critical**: Uses banker's rounding (round-half-to-even), matching C's `lrint()` with `FE_TONEAREST`.

---

### Step 8: Bitstream Packing

The 212-byte frame is packed as:

```
┌──────────────────────────────────────────────────┐
│ Block size mode     8 bits  (2+2+2+2 per band)  │
│ BFU amount index    3 bits  (indexes into table) │
│ Reserved            5 bits  (zeros)              │
│ Word lengths        4 bits × N_bfu              │
│ Scale factors       6 bits × N_bfu              │
│ Mantissas           variable bits per BFU        │
│ Padding             fill to 1696 bits            │
└──────────────────────────────────────────────────┘
```

---

## Decoding Pipeline (Inverse)

```
Bitstream → Parse → Dequantize → IMDCT → QMF Synthesis → PCM[512]
```

### Dequantization

```
X[i] = ScaleTable[sf_idx] · mantissa[i] / (2^(wordlen-1) - 1)
```

### Inverse MDCT

```
x[n] = (1/N) · Σₖ₌₀ᴺ⁻¹ X[k] · cos(π/N · (n + ½ + N/2) · (k + ½))
```

With overlap-add of consecutive frames (TDAC — Time Domain Aliasing Cancellation).

### QMF Synthesis

Reconstructs 512 PCM samples from 3 bands (reverse of analysis).

---

## Key Numbers

| Parameter | Value |
|-----------|-------|
| Samples per frame | 512 |
| Frame size (bytes) | 212 |
| Frame size (bits) | 1696 |
| Sample rate | 44100 Hz |
| Max BFUs | 52 |
| Frequency bands | 3 (low/mid/high) |
| Scale factors | 64 (6-bit index) |
| Max word length | 15 bits |
| Bitrate | ~292 kbps (stereo) |

## BFU Layout

```
Band    BFUs      Specs    Frequency Range
Low     0-19      128      0 - 5.5 kHz
Mid     20-35     128      5.5 - 11 kHz
High    36-51     256      11 - 22.05 kHz
Total   52        512
```
