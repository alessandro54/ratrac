//! ATRAC1 bit allocation and frame writing.
//!
//! This module decides how many bits each BFU (Basic Functional Unit) gets
//! within the 212-byte frame budget. The pipeline has 7 steps:
//!
//! 1. **Initial allocation**: Binary search on a `shift` parameter to find
//!    word lengths that fit the bit budget. Uses a blend of scale-factor-based
//!    and fixed-table-based allocation, weighted by spectral `spread`.
//!
//! 2. **BFU count optimization**: If trailing BFUs are all zero, reduce the
//!    BFU count to save header overhead (fewer word length + scale factor fields).
//!
//! 3. **Bit boost**: Redistribute leftover bits to important BFUs (band
//!    transition zones) using the `BitsBooster`.
//!
//! 4. **Safety cap**: Trim any over-allocation caused by the boost phase
//!    (the 0→2 bit jump costs double).
//!
//! 5. **Greedy fill**: Use any remaining slack bits on the highest-energy
//!    BFU that can accept them.
//!
//! 6. **Analysis-by-Synthesis (AbS)**: Measure physical-domain quantization
//!    noise vs. the ATH masking threshold per BFU. Robin Hood bits from
//!    over-masked (inaudible noise) to under-masked (audible noise) BFUs.
//!
//! 7. **Bitstream write**: Pack the final word lengths, scale factors, and
//!    quantized mantissas into the 212-byte frame.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use crate::atrac1::{
    BFU_AMOUNT_TAB, BITS_PER_BFU_AMOUNT_TAB_IDX, BITS_PER_IDSF, BITS_PER_IDWL, BLOCKS_PER_BAND,
    BlockSizeMod, MAX_BFUS, NUM_QMF, SOUND_UNIT_SIZE, SPECS_PER_BLOCK, SPECS_START_LONG,
    bfu_to_band,
};
use crate::bitstream::{BitStream, make_sign};
use crate::psychoacoustic::{analyze_scale_factor_spread, calc_ath};
use crate::scaler::ScaledBlock;
use crate::util::to_int;

// ─── Named Constants ─────────────────────────────────────────────────────────

/// Maximum word length per BFU (15 bits per mantissa + 1 sign = 16).
const MAX_WORD_LENGTH: u32 = 16;
/// Minimum useful word length (below 2, coefficients are zeroed).
const MIN_USEFUL_WORD_LENGTH: u32 = 2;
/// Fixed overhead bits for the binary search budget calculation.
/// Includes: block_size(8) + bfu_idx(3) + reserved(5) + footer(24) = 40.
const FIXED_OVERHEAD_BITS: usize = 8 + 3 + 5 + 24;
/// Footer padding bits (subtracted again for the safety cap / greedy fill).
const FOOTER_BITS: usize = 24;
/// Binary search convergence threshold for the `shift` parameter.
const SHIFT_CONVERGENCE: f32 = 0.1;
/// Binary search bounds and starting point.
const SHIFT_INITIAL: f32 = 3.0;
const SHIFT_MAX: f32 = 15.0;
const SHIFT_MIN: f32 = -3.0;
/// How many bits of slack the binary search allows below the target.
/// Lower = more bits used per frame = better quality.
const BIT_BUDGET_SLACK: usize = 16;
/// Scale factor index divisor in the allocation formula.
const SF_INDEX_DIVISOR: f32 = 3.2;
/// Number of AbS (Analysis-by-Synthesis) refinement iterations.
const ABS_ITERATIONS: usize = 6;
/// AbS only swaps if worst NMR is at least this many times the best NMR.
const ABS_NMR_RATIO: f32 = 10.0;
/// AbS allows this much bit-budget slack per swap (safety cap fixes overflow).
const ABS_BUDGET_SLACK: i32 = 20;

// ─── Static Tables ───────────────────────────────────────────────────────────

/// Default bits per BFU for long (single) MDCT windows.
/// Low band gets 6-7 bits, mid gets 5-6, high gets 0-4 (less audible).
const FIXED_BIT_ALLOC_TABLE_LONG: [u32; MAX_BFUS] = [
    7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 4, 4, 4, 3, 3, 3, 3, 3, 3, 2, 1, 1, 1, 1, 0, 0, 0,
];

/// Default bits per BFU for short (multiple) MDCT windows.
const FIXED_BIT_ALLOC_TABLE_SHORT: [u32; MAX_BFUS] = [
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// Which BFUs can receive boosted bits (1 = eligible).
/// Targets band transition zones where extra bits improve quality most.
const BIT_BOOST_MASK: [u32; MAX_BFUS] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// Per-BFU ATH (Absolute Threshold of Hearing) in linear scale.
/// Computed once from the 512-bin ATH curve by taking the minimum
/// threshold within each BFU's frequency range.
static ATH_LONG: LazyLock<Vec<f32>> = LazyLock::new(|| {
    let ath_spec = calc_ath(512, 44100);
    let mut ath_long = Vec::with_capacity(MAX_BFUS);
    for band in 0..NUM_QMF {
        for bfu in BLOCKS_PER_BAND[band] as usize..BLOCKS_PER_BAND[band + 1] as usize {
            let start = SPECS_START_LONG[bfu] as usize;
            let count = SPECS_PER_BLOCK[bfu] as usize;
            let min_ath = ath_spec[start..start + count]
                .iter()
                .fold(999.0f32, |a, &b| a.min(b));
            ath_long.push(10.0_f32.powf(0.1 * min_ath));
        }
    }
    ath_long
});

// ─── BitsBooster ─────────────────────────────────────────────────────────────

/// Redistributes leftover bits to eligible BFUs after the initial allocation.
///
/// The boost mask targets band transition zones (BFUs 18-22 and 32-38)
/// where extra bits reduce audible artifacts at frequency boundaries.
///
/// Note: When a BFU goes from 0→2 bits, it costs `specs_per_block × 2`
/// (double), because word length 1 is not useful. This can overshoot
/// the budget, which is why the safety cap runs after boosting.
pub struct BitsBooster {
    /// Map from bit cost (specs_per_block) → list of eligible BFU indices.
    bits_boost_map: BTreeMap<u32, Vec<u32>>,
    max_bits_per_iteration: u32,
    min_key: u32,
}

impl Default for BitsBooster {
    fn default() -> Self {
        Self::new()
    }
}

impl BitsBooster {
    pub fn new() -> Self {
        let mut map: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for i in 0..MAX_BFUS {
            if BIT_BOOST_MASK[i] == 0 {
                continue;
            }
            map.entry(SPECS_PER_BLOCK[i]).or_default().push(i as u32);
        }
        let max_bits = map.keys().last().copied().unwrap_or(0);
        let min_key = map.keys().next().copied().unwrap_or(0);
        Self {
            bits_boost_map: map,
            max_bits_per_iteration: max_bits,
            min_key,
        }
    }

    /// Distribute `target - cur` surplus bits to eligible BFUs.
    /// Returns remaining unused surplus.
    pub fn apply_boost(&self, bits_per_block: &mut Vec<u32>, cur: u32, target: u32) -> u32 {
        let mut surplus = target - cur;
        let key = surplus.min(self.max_bits_per_iteration);

        let entries: Vec<(u32, u32)> = self
            .bits_boost_map
            .range(..=key)
            .flat_map(|(&bits, positions)| positions.iter().map(move |&pos| (bits, pos)))
            .collect();

        if entries.is_empty() {
            return surplus;
        }

        while surplus >= self.min_key {
            let mut done = true;
            for &(cost, pos) in &entries {
                let pos = pos as usize;
                if pos >= bits_per_block.len() || bits_per_block[pos] == MAX_WORD_LENGTH as u32 {
                    continue;
                }
                // Going from 0→2 costs double (word length 1 is useless)
                let increment = if bits_per_block[pos] != 0 { 1 } else { 2 };
                let total_cost = cost * increment;
                if bits_per_block[pos] == 0 && total_cost > surplus {
                    continue;
                }
                if total_cost > surplus {
                    continue;
                }
                bits_per_block[pos] += increment;
                surplus -= total_cost;
                done = false;
            }
            if done {
                break;
            }
        }

        surplus
    }
}

// ─── Bitstream Writer ────────────────────────────────────────────────────────

/// Pack a complete ATRAC1 frame (212 bytes) from bit allocations and scaled blocks.
///
/// Frame layout:
/// ```text
/// [block_size: 8] [bfu_idx: 3] [reserved: 5]
/// [word_lengths: 4 × N] [scale_factors: 6 × N]
/// [mantissas: variable] [padding to 212 bytes]
/// ```
pub fn write_bitstream(
    bits_per_block: &[u32],
    scaled_blocks: &[ScaledBlock],
    bfu_amount_idx: u32,
    block_size: &BlockSizeMod,
) -> Vec<u8> {
    let mut bs = BitStream::new();

    // ── Header ──
    bs.write((2 - block_size.log_count[0]) as u32, 2); // low band window mode
    bs.write((2 - block_size.log_count[1]) as u32, 2); // mid band window mode
    bs.write((3 - block_size.log_count[2]) as u32, 2); // high band window mode
    bs.write(0, 2); // unused
    bs.write(bfu_amount_idx, BITS_PER_BFU_AMOUNT_TAB_IDX);
    bs.write(0, 2); // reserved
    bs.write(0, 3); // reserved

    // ── Word lengths (4 bits each, stored as wl-1) ──
    for &wl in bits_per_block {
        bs.write(if wl > 0 { wl - 1 } else { 0 }, BITS_PER_IDWL);
    }

    // ── Scale factors (6 bits each) ──
    for i in 0..bits_per_block.len() {
        bs.write(scaled_blocks[i].scale_factor_index as u32, BITS_PER_IDSF);
    }

    // ── Quantized mantissas ──
    for i in 0..bits_per_block.len() {
        let wl = bits_per_block[i] as usize;
        if wl < MIN_USEFUL_WORD_LENGTH as usize {
            continue;
        }
        let mul = ((1u32 << (wl - 1)) - 1) as f32;
        for &val in &scaled_blocks[i].values {
            let mantissa = to_int(val * mul);
            bs.write(make_sign(mantissa, wl as u32) as u32, wl);
        }
    }

    // ── Footer + padding ──
    bs.write(0, 8);
    bs.write(0, 8);
    bs.write(0, 8);
    let mut bytes = bs.get_bytes().to_vec();
    bytes.resize(SOUND_UNIT_SIZE, 0);
    bytes
}

// ─── Bit Allocation Core ─────────────────────────────────────────────────────

/// Calculate how many data bits are available for a given BFU count.
/// How many data bits are available for mantissas, given a BFU count.
/// This accounts for all overhead: header, BFU count, word lengths,
/// scale factors, reserved bits, and footer padding.
fn data_bits_available(bfu_count: usize) -> usize {
    SOUND_UNIT_SIZE * 8 - FIXED_OVERHEAD_BITS - bfu_count * (BITS_PER_IDWL + BITS_PER_IDSF)
}

/// Count total data bits consumed by an allocation.
fn count_data_bits(alloc: &[u32]) -> u32 {
    alloc
        .iter()
        .enumerate()
        .map(|(i, &b)| SPECS_PER_BLOCK[i] * b)
        .sum()
}

/// Propose a bit allocation for each BFU based on the `shift` parameter.
///
/// The formula blends two strategies:
/// - **Scale-factor-driven** (tonal signals): `sf_index / 3.2` — louder BFUs get more bits.
/// - **Fixed-table-driven** (noisy signals): predetermined allocation per BFU position.
///
/// The `spread` parameter (0=noise-like, 1=tone-like) controls the blend.
/// BFUs below the ATH threshold get zero bits (inaudible).
fn propose_allocation(
    scaled_blocks: &[ScaledBlock],
    bfu_num: usize,
    spread: f32,
    shift: f32,
    block_size: &BlockSizeMod,
    loudness: f32,
) -> Vec<u32> {
    let ath_long = &*ATH_LONG;
    let mut bits = vec![0u32; bfu_num];

    for i in 0..bfu_num {
        let band = bfu_to_band(i as u32) as usize;
        let short_block = block_size.log_count[band] != 0;
        let fixed = if short_block {
            FIXED_BIT_ALLOC_TABLE_SHORT[i]
        } else {
            FIXED_BIT_ALLOC_TABLE_LONG[i]
        };

        // Skip BFUs below the hearing threshold (long blocks only)
        let ath = ath_long[i] * loudness;
        if !short_block && scaled_blocks[i].max_energy < ath {
            bits[i] = 0;
            continue;
        }

        // Blend: spread × (energy-based) + (1-spread) × (fixed table) - shift
        let raw = spread * (scaled_blocks[i].scale_factor_index as f32 / SF_INDEX_DIVISOR)
            + (1.0 - spread) * fixed as f32
            - shift;

        let clamped = raw as i32;
        if clamped > MAX_WORD_LENGTH as i32 {
            bits[i] = MAX_WORD_LENGTH;
        } else if clamped < MIN_USEFUL_WORD_LENGTH as i32 {
            bits[i] = 0;
        } else {
            bits[i] = clamped as u32;
        }
    }

    bits
}

/// Find the highest BFU amount index that still has non-zero trailing BFUs.
/// This lets us reduce the BFU count to save header bits.
fn get_max_used_bfu_id(bits_per_block: &[u32]) -> u32 {
    let mut idx: u32 = 7;
    loop {
        let bfu_num = BFU_AMOUNT_TAB[idx as usize] as usize;
        if bfu_num > bits_per_block.len() {
            idx -= 1;
        } else if idx != 0 {
            let mut i: usize = 0;
            let mut bfu_num = bfu_num;
            while idx > 0 && bits_per_block[bfu_num - 1 - i] == 0 {
                i += 1;
                let step =
                    (BFU_AMOUNT_TAB[idx as usize] - BFU_AMOUNT_TAB[idx as usize - 1]) as usize;
                if i >= step {
                    idx -= 1;
                    bfu_num -= i;
                    i = 0;
                }
            }
            break;
        } else {
            break;
        }
    }
    idx
}

/// Trim over-allocated bits from highest-numbered BFUs to fit within budget.
fn apply_safety_cap(bits_per_block: &mut [u32], max_data_bits: u32) {
    let total = count_data_bits(bits_per_block);
    if total <= max_data_bits {
        return;
    }

    let mut excess = total - max_data_bits;
    for i in (0..bits_per_block.len()).rev() {
        if excess == 0 {
            break;
        }
        while bits_per_block[i] > 0 && excess > 0 {
            let freed = SPECS_PER_BLOCK[i];
            if freed <= excess {
                bits_per_block[i] -= 1;
                excess -= freed;
            } else {
                break;
            }
        }
    }
}

/// Fill any remaining bit budget with +1 to the BFU that benefits most.
/// Prioritizes high-energy, low-bit BFUs (biggest quality improvement per bit).
fn fill_remaining_bits(bits_per_block: &mut [u32], scaled_blocks: &[ScaledBlock], budget: u32) {
    let mut remaining = budget.saturating_sub(count_data_bits(bits_per_block));

    while remaining > 0 {
        let mut best: Option<(usize, f32)> = None;
        for i in 0..bits_per_block.len() {
            if bits_per_block[i] >= MAX_WORD_LENGTH || bits_per_block[i] < MIN_USEFUL_WORD_LENGTH {
                continue;
            }
            if SPECS_PER_BLOCK[i] > remaining {
                continue;
            }
            let energy = scaled_blocks[i].max_energy;
            if energy <= 0.0 {
                continue;
            }
            let score = energy / bits_per_block[i] as f32;
            match best {
                None => best = Some((i, score)),
                Some((_, prev)) if score > prev => best = Some((i, score)),
                _ => {}
            }
        }
        match best {
            Some((i, _)) => {
                bits_per_block[i] += 1;
                remaining -= SPECS_PER_BLOCK[i];
            }
            None => break,
        }
    }
}

// ─── Analysis-by-Synthesis (AbS) ─────────────────────────────────────────────

/// Compute the physical-domain quantization noise for a single scaled value.
///
/// "Physical domain" means we multiply the error by the scale factor to get
/// the true amplitude of the noise. This ensures loud BFUs are weighted more
/// heavily than quiet ones (a 1-bit error on a loud guitar matters more than
/// a 1-bit error on a barely-audible hiss).
fn physical_noise_power(scaled_val: f32, word_length: u32, sf_index: u8) -> f32 {
    let sf = crate::atrac1::SCALE_TABLE[sf_index as usize];

    if word_length < MIN_USEFUL_WORD_LENGTH {
        let phys = scaled_val * sf;
        return phys * phys;
    }

    let mul = ((1u32 << (word_length - 1)) - 1) as f32;
    let quantized = to_int(scaled_val * mul) as f32 / mul;
    let phys_error = (scaled_val - quantized) * sf;
    phys_error * phys_error
}

/// Measure physical noise and signal power per BFU.
fn measure_bfu_noise(bits_per_block: &[u32], scaled_blocks: &[ScaledBlock]) -> Vec<(f32, f32)> {
    let n = bits_per_block.len();
    let mut result = vec![(0.0f32, 0.0f32); n];
    let scale_table = &*crate::atrac1::SCALE_TABLE;

    for i in 0..n {
        let sf = scale_table[scaled_blocks[i].scale_factor_index as usize];
        let mut noise = 0.0f32;
        let mut signal = 0.0f32;
        for &val in &scaled_blocks[i].values {
            noise +=
                physical_noise_power(val, bits_per_block[i], scaled_blocks[i].scale_factor_index);
            signal += (val * sf) * (val * sf);
        }
        result[i] = (noise, signal);
    }

    result
}

/// Approximate bark frequency for a BFU center.
/// Uses the Traunmüller formula: bark = 26.81 * f / (1960 + f) - 0.53
fn bfu_bark(bfu: usize) -> f32 {
    let freq =
        (SPECS_START_LONG[bfu] as f32 + SPECS_PER_BLOCK[bfu] as f32 / 2.0) * 44100.0 / 1024.0;
    26.81 * freq / (1960.0 + freq) - 0.53
}

/// Bark-scale spreading: loud BFUs raise the masking threshold of neighbors.
///
/// The human inner ear's basilar membrane vibrates broadly — a loud 1 kHz tone
/// masks quiet sounds at 0.9 and 1.1 kHz. We model this by "spreading" each
/// BFU's signal energy to its neighbors with frequency-dependent decay:
/// - Upper slope: -25 dB/bark (masking falls off quickly above the masker)
/// - Lower slope: -10 dB/bark (masking extends further below the masker)
/// - Masking offset: -14 dB (noise floor sits below the masker)
fn compute_spread_masking(scaled_blocks: &[ScaledBlock], n: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; n];

    for i in 0..n {
        let energy = scaled_blocks[i].max_energy;
        if energy <= 0.0 {
            continue;
        }
        let masker_db = 10.0 * energy.log10();
        let bark_i = bfu_bark(i);

        for j in 0..n {
            if i == j {
                continue;
            }
            let bark_dist = bfu_bark(j) - bark_i;

            // Asymmetric spreading: easier to mask above than below
            let spread_db = if bark_dist > 0.0 {
                masker_db - 25.0 * bark_dist // upper slope
            } else {
                masker_db - 10.0 * bark_dist.abs() // lower slope
            };

            // The masking floor is ~18 dB below the masker (conservative)
            let mask_db = spread_db - 18.0;

            if mask_db > -100.0 {
                let mask_linear = 10.0_f32.powf(mask_db / 10.0);
                if mask_linear > mask[j] {
                    mask[j] = mask_linear;
                }
            }
        }
    }

    mask
}

/// Spectral Flatness Measure (SFM) for a set of BFU energies.
///
/// SFM = geometric_mean / arithmetic_mean of the energy values.
/// - SFM ≈ 1.0 → noise-like (flat spectrum, like cymbals)
/// - SFM ≈ 0.0 → tonal (peaked spectrum, like a flute)
///
/// For tonal frames, quantization noise is easily audible (stricter NMR).
/// For noisy frames, noise hides in the chaos (relaxed NMR).
fn spectral_flatness(scaled_blocks: &[ScaledBlock], n: usize) -> f32 {
    if n == 0 {
        return 0.5;
    }

    let energies: Vec<f64> = (0..n)
        .map(|i| (scaled_blocks[i].max_energy as f64).max(1e-20))
        .collect();

    let arithmetic_mean = energies.iter().sum::<f64>() / n as f64;
    let log_sum: f64 = energies.iter().map(|&e| e.ln()).sum::<f64>() / n as f64;
    let geometric_mean = log_sum.exp();

    let sfm = (geometric_mean / arithmetic_mean.max(1e-20)) as f32;
    sfm.clamp(0.0, 1.0)
}

/// Robin Hood bit reallocation based on Noise-to-Mask Ratio (NMR).
///
/// For each BFU, NMR = physical_noise / masking_threshold.
/// The masking threshold combines:
/// - ATH (absolute threshold of hearing)
/// - Bark-scale spreading (loud neighbors raise the threshold)
/// - Spectral flatness (noisy frames get relaxed thresholds)
///
/// The loop steals 1 bit from the most over-masked BFU and gives it
/// to the most under-masked one, up to `ABS_ITERATIONS` times.
fn abs_reallocate(
    bits_per_block: &mut [u32],
    scaled_blocks: &[ScaledBlock],
    ath_long: &[f32],
    loudness: f32,
) {
    let n = bits_per_block.len();
    if n < 2 {
        return;
    }

    // Spectral flatness: tonal frames need stricter masking
    let sfm = spectral_flatness(scaled_blocks, n);

    // Only apply bark spreading for noise-like content (SFM > 0.3).
    // For tonal content (classical, clean instruments), spreading steals
    // bits from quiet melodic lines — counterproductive.
    let spread_mask = if sfm > 0.2 {
        compute_spread_masking(scaled_blocks, n)
    } else {
        vec![0.0f32; n] // no spreading for tonal content
    };

    // Tonality factor: 1.0 for tonal (strict), up to 3.0 for noisy (relaxed)
    let tonality_boost = 1.0 + 2.0 * sfm;

    for _ in 0..ABS_ITERATIONS {
        let noise = measure_bfu_noise(bits_per_block, scaled_blocks);

        // Combined masking threshold per BFU:
        // max(ATH, spreading) × loudness × tonality_boost
        let nmr: Vec<f32> = (0..n)
            .map(|i| {
                let ath = ath_long[i] * loudness;
                let combined_mask = ath.max(spread_mask[i]) * tonality_boost;
                noise[i].0 / combined_mask.max(1e-15)
            })
            .collect();

        // Find the BFU with the worst (highest) NMR — needs more bits
        let worst = (0..n)
            .filter(|&i| {
                bits_per_block[i] >= MIN_USEFUL_WORD_LENGTH && bits_per_block[i] < MAX_WORD_LENGTH
            })
            .max_by(|&a, &b| nmr[a].partial_cmp(&nmr[b]).unwrap());

        // Find the BFU with the best (lowest) NMR — can donate bits
        let donor = (0..n)
            .filter(|&i| bits_per_block[i] >= 3)
            .min_by(|&a, &b| nmr[a].partial_cmp(&nmr[b]).unwrap());

        match (worst, donor) {
            (Some(w), Some(d)) if w != d && nmr[w] > nmr[d] * ABS_NMR_RATIO => {
                // Strict budget check: only swap if total stays within frame capacity
                let budget_change = SPECS_PER_BLOCK[w] as i32 - SPECS_PER_BLOCK[d] as i32;
                let current_total = count_data_bits(bits_per_block);
                let max_budget = data_bits_available(bits_per_block.len()) as u32;
                if current_total as i32 + budget_change <= max_budget as i32 {
                    bits_per_block[w] += 1;
                    bits_per_block[d] -= 1;
                } else {
                    break;
                }
            }
            _ => break,
        }
    }
}

// ─── Main Entry Point ────────────────────────────────────────────────────────

/// Allocate bits and write one ATRAC1 frame (212 bytes).
///
/// This is the main encoder entry point for a single channel's spectral data.
/// Returns `(frame_bytes, num_bfus_used)`.
pub fn write_frame(
    scaled_blocks: &[ScaledBlock],
    block_size: &BlockSizeMod,
    loudness: f32,
    bfu_idx_const: u32,
    fast_bfu_num_search: bool,
) -> (Vec<u8>, u32) {
    let auto_bfu = bfu_idx_const == 0;
    let mut bfu_idx: u32 = if bfu_idx_const > 0 {
        bfu_idx_const - 1
    } else {
        7 // start with maximum BFU count
    };

    // Analyze spectral characteristics to choose allocation strategy
    let sf_indices: Vec<u8> = scaled_blocks.iter().map(|b| b.scale_factor_index).collect();
    let spread = analyze_scale_factor_spread(&sf_indices);
    let booster = BitsBooster::new();

    let mut bits_per_block = vec![0u32; BFU_AMOUNT_TAB[bfu_idx as usize] as usize];
    let target_bits;
    let mut cur_bits;

    // ── Step 1-2: Binary search for optimal shift + BFU count ──
    loop {
        bits_per_block.resize(BFU_AMOUNT_TAB[bfu_idx as usize] as usize, 0);
        let budget = data_bits_available(bits_per_block.len());
        let max_bits = budget;
        let min_bits = budget - BIT_BUDGET_SLACK;

        let mut max_shift = SHIFT_MAX;
        let mut min_shift = SHIFT_MIN;
        let mut shift = SHIFT_INITIAL;
        let mut bfu_num_changed = false;

        loop {
            let alloc = propose_allocation(
                scaled_blocks,
                BFU_AMOUNT_TAB[bfu_idx as usize] as usize,
                spread,
                shift,
                block_size,
                loudness,
            );
            let bits_used = count_data_bits(&alloc);

            if bits_used < min_bits as u32 {
                // Under budget — try to use more bits
                if max_shift - min_shift < SHIFT_CONVERGENCE {
                    // Converged: accept this allocation
                    if auto_bfu {
                        let used_id = get_max_used_bfu_id(&alloc);
                        if used_id < bfu_idx {
                            bfu_num_changed = true;
                            bfu_idx = if fast_bfu_num_search {
                                used_id
                            } else {
                                bfu_idx - 1
                            };
                        }
                    }
                    if !bfu_num_changed {
                        bits_per_block = alloc;
                    }
                    cur_bits = bits_used;
                    break;
                }
                max_shift = shift;
                shift -= (shift - min_shift) / 2.0;
            } else if bits_used > max_bits as u32 {
                // Over budget — reduce bits
                min_shift = shift;
                shift += (max_shift - shift) / 2.0;
            } else {
                // Within budget — accept
                if auto_bfu {
                    let used_id = get_max_used_bfu_id(&alloc);
                    if used_id < bfu_idx {
                        bfu_num_changed = true;
                        bfu_idx = if fast_bfu_num_search {
                            used_id
                        } else {
                            bfu_idx - 1
                        };
                    }
                }
                if !bfu_num_changed {
                    bits_per_block = alloc;
                }
                cur_bits = bits_used;
                break;
            }
        }

        if !bfu_num_changed {
            target_bits = budget as u32;
            break;
        }
    }

    // ── Step 3: Boost ──
    booster.apply_boost(&mut bits_per_block, cur_bits, target_bits);

    // ── Step 4: Safety cap ──
    // The binary search targets data_bits_available (includes footer space).
    // For the safety cap and greedy fill, we subtract footer to be safe.
    let max_data = (data_bits_available(bits_per_block.len()) - FOOTER_BITS) as u32;
    apply_safety_cap(&mut bits_per_block, max_data);

    // ── Step 5: Greedy fill ──
    fill_remaining_bits(&mut bits_per_block, scaled_blocks, max_data);

    // ── Step 6: Analysis-by-Synthesis ──
    abs_reallocate(&mut bits_per_block, scaled_blocks, &ATH_LONG, loudness);

    // ── Step 7: Write bitstream ──
    let frame = write_bitstream(&bits_per_block, scaled_blocks, bfu_idx, block_size);
    (frame, BFU_AMOUNT_TAB[bfu_idx as usize])
}

#[cfg(test)]
#[path = "../tests/atrac1_bitalloc_tests.rs"]
mod tests;
