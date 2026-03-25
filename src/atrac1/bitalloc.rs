use std::collections::BTreeMap;

use crate::atrac1::{
    BFU_AMOUNT_TAB, BITS_PER_BFU_AMOUNT_TAB_IDX, BITS_PER_IDSF, BITS_PER_IDWL, BLOCKS_PER_BAND,
    BlockSizeMod, MAX_BFUS, NUM_QMF, SOUND_UNIT_SIZE, SPECS_PER_BLOCK, SPECS_START_LONG,
    bfu_to_band,
};
use crate::bitstream::{BitStream, make_sign};
use crate::psychoacoustic::{analyze_scale_factor_spread, calc_ath};
use crate::scaler::ScaledBlock;
use crate::util::to_int;

// --- Static tables ---

const FIXED_BIT_ALLOC_TABLE_LONG: [u32; MAX_BFUS] = [
    7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 4, 4, 4, 3, 3, 3, 3, 3, 3, 2, 1, 1, 1, 1, 0, 0, 0,
];

const FIXED_BIT_ALLOC_TABLE_SHORT: [u32; MAX_BFUS] = [
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0,
];

const BIT_BOOST_MASK: [u32; MAX_BFUS] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

// --- BitsBooster ---

/// Redistributes spare bits to important BFUs using the BitBoostMask.
pub struct BitsBooster {
    /// (bits_needed, bfu_position) sorted by bits_needed
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
            let n_bits = SPECS_PER_BLOCK[i];
            map.entry(n_bits).or_default().push(i as u32);
        }
        let max_bits = map.keys().last().copied().unwrap_or(0);
        let min_key = map.keys().next().copied().unwrap_or(0);
        Self {
            bits_boost_map: map,
            max_bits_per_iteration: max_bits,
            min_key,
        }
    }

    /// Apply boost: distribute surplus bits from `cur` up to `target`.
    /// Returns remaining surplus.
    pub fn apply_boost(&self, bits_per_block: &mut Vec<u32>, cur: u32, target: u32) -> u32 {
        let mut surplus = target - cur;
        let key = surplus.min(self.max_bits_per_iteration);

        // Collect entries up to key (upper_bound equivalent)
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
            for &(cur_bits, cur_pos) in &entries {
                if cur_pos as usize >= bits_per_block.len() {
                    break;
                }
                if bits_per_block[cur_pos as usize] == 16 {
                    continue;
                }
                let n_bits_per_spec = if bits_per_block[cur_pos as usize] != 0 {
                    1
                } else {
                    2
                };
                if bits_per_block[cur_pos as usize] == 0 && cur_bits * 2 > surplus {
                    continue;
                }
                if cur_bits * n_bits_per_spec > surplus {
                    continue;
                }
                bits_per_block[cur_pos as usize] += n_bits_per_spec;
                surplus -= cur_bits * n_bits_per_spec;
                done = false;
            }
            if done {
                break;
            }
        }

        surplus
    }
}

// --- ATH for long blocks (lazy init) ---

use std::sync::LazyLock;

static ATH_LONG: LazyLock<Vec<f32>> = LazyLock::new(|| {
    let ath_spec = calc_ath(512, 44100);
    let mut ath_long = Vec::with_capacity(MAX_BFUS);
    for band in 0..NUM_QMF {
        let start_bfu = BLOCKS_PER_BAND[band] as usize;
        let end_bfu = BLOCKS_PER_BAND[band + 1] as usize;
        for bfu in start_bfu..end_bfu {
            let spec_start = SPECS_START_LONG[bfu] as usize;
            let num_specs = SPECS_PER_BLOCK[bfu] as usize;
            let mut x = 999.0f32;
            for line in spec_start..spec_start + num_specs {
                x = x.min(ath_spec[line]);
            }
            x = 10.0_f32.powf(0.1 * x);
            ath_long.push(x);
        }
    }
    ath_long
});

// --- BitStreamWriter ---

/// Write an ATRAC1 bitstream frame (212 bytes).
/// Returns the raw bytes of the frame.
pub fn write_bitstream(
    bits_per_block: &[u32],
    scaled_blocks: &[ScaledBlock],
    bfu_amount_idx: u32,
    block_size: &BlockSizeMod,
) -> Vec<u8> {
    let mut bs = BitStream::new();

    // Block size mode (8 bits)
    bs.write((2 - block_size.log_count[0]) as u32, 2);
    bs.write((2 - block_size.log_count[1]) as u32, 2);
    bs.write((3 - block_size.log_count[2]) as u32, 2);
    bs.write(0, 2);

    // BFU amount index (3 bits)
    bs.write(bfu_amount_idx, BITS_PER_BFU_AMOUNT_TAB_IDX);

    // Reserved (5 bits)
    bs.write(0, 2);
    bs.write(0, 3);

    // Word lengths (4 bits each): encode as wl > 0 ? wl-1 : 0
    for &wl in bits_per_block {
        let tmp = if wl > 0 { wl - 1 } else { 0 };
        bs.write(tmp, BITS_PER_IDWL);
    }

    // Scale factors (6 bits each)
    for i in 0..bits_per_block.len() {
        bs.write(scaled_blocks[i].scale_factor_index as u32, BITS_PER_IDSF);
    }

    // Quantized mantissas
    for i in 0..bits_per_block.len() {
        let word_length = bits_per_block[i] as usize;
        if word_length == 0 || word_length == 1 {
            continue;
        }

        let multiple = ((1u32 << (word_length - 1)) - 1) as f32;
        for &val in &scaled_blocks[i].values {
            let tmp = to_int(val * multiple);
            bs.write(make_sign(tmp, word_length as u32) as u32, word_length);
        }
    }

    // Footer padding (24 bits)
    bs.write(0, 8);
    bs.write(0, 8);
    bs.write(0, 8);

    // Pad the byte buffer to SOUND_UNIT_SIZE
    let mut bytes = bs.get_bytes().to_vec();
    bytes.resize(SOUND_UNIT_SIZE, 0);
    bytes
}

// --- Psychoacoustic masking ---

// --- Bit allocation algorithm ---

fn calc_bits_allocation(
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
        let fix = if short_block {
            FIXED_BIT_ALLOC_TABLE_SHORT[i]
        } else {
            FIXED_BIT_ALLOC_TABLE_LONG[i]
        };

        let ath = ath_long[i] * loudness;

        if !short_block && scaled_blocks[i].max_energy < ath {
            bits[i] = 0;
        } else {
            let tmp = spread * (scaled_blocks[i].scale_factor_index as f32 / 3.2)
                + (1.0 - spread) * fix as f32
                - shift;
            let tmp = tmp as i32;
            if tmp > 16 {
                bits[i] = 16;
            } else if tmp < 2 {
                bits[i] = 0;
            } else {
                bits[i] = tmp as u32;
            }
        }
    }

    bits
}

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
                if i >= (BFU_AMOUNT_TAB[idx as usize] - BFU_AMOUNT_TAB[idx as usize - 1]) as usize {
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

/// Simulate exact quantization → dequantization noise in PHYSICAL domain.
/// Uses the scale factor to convert from normalized [-1,1] back to true amplitude.
/// Returns squared physical noise power.
fn physical_noise_power(scaled_val: f32, word_length: u32, sf_index: u8) -> f32 {
    let scale_table = &*crate::atrac1::SCALE_TABLE;
    let sf = scale_table[sf_index as usize];

    if word_length < 2 {
        // Zero bits: decoder outputs 0.0. Error = entire physical signal.
        let phys = scaled_val * sf;
        return phys * phys;
    }

    // Bit-exact encoder math (matches write_bitstream)
    let mul = ((1u32 << (word_length - 1)) - 1) as f32;
    let quantized_int = to_int(scaled_val * mul);

    // Bit-exact decoder math (matches dequantiser)
    let reconstructed = quantized_int as f32 / mul;

    // Error in physical domain
    let phys_error = (scaled_val - reconstructed) * sf;
    phys_error * phys_error
}

/// Compute per-BFU physical noise power and signal power.
/// Returns (noise_power, signal_power) per BFU.
fn compute_bfu_physical_noise(
    bits_per_block: &[u32],
    scaled_blocks: &[ScaledBlock],
) -> Vec<(f32, f32)> {
    let n = bits_per_block.len();
    let mut result = vec![(0.0f32, 0.0f32); n];
    let scale_table = &*crate::atrac1::SCALE_TABLE;

    for i in 0..n {
        let sf = scale_table[scaled_blocks[i].scale_factor_index as usize];
        let mut noise = 0.0f32;
        let mut signal = 0.0f32;
        for &val in &scaled_blocks[i].values {
            noise += physical_noise_power(val, bits_per_block[i], scaled_blocks[i].scale_factor_index);
            let phys = val * sf;
            signal += phys * phys;
        }
        result[i] = (noise, signal);
    }

    result
}

/// Analysis-by-Synthesis bit reallocation using physical-domain NMR.
/// Steals bits from BFUs where noise is well below the masking threshold,
/// gives them to BFUs where noise is closest to (or above) the threshold.
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

    for _ in 0..6 {
        let phys = compute_bfu_physical_noise(bits_per_block, scaled_blocks);

        // Compute NMR: physical_noise / masking_threshold
        // NMR > 1.0 = noise above threshold (audible!)
        // NMR < 0.01 = noise deeply buried (can steal bits)
        let nmr: Vec<f32> = (0..n)
            .map(|i| {
                let mask = (ath_long[i] * loudness).max(1e-15);
                phys[i].0 / mask
            })
            .collect();

        // Find worst BFU (highest NMR)
        let mut worst_bfu = None;
        let mut max_nmr = f32::MIN;
        for i in 0..n {
            if bits_per_block[i] >= 16 || bits_per_block[i] < 2 {
                continue;
            }
            if nmr[i] > max_nmr {
                max_nmr = nmr[i];
                worst_bfu = Some(i);
            }
        }

        // Find best donor (lowest NMR)
        let mut best_bfu = None;
        let mut min_nmr = f32::MAX;
        for i in 0..n {
            if bits_per_block[i] < 3 {
                continue;
            }
            if nmr[i] < min_nmr {
                min_nmr = nmr[i];
                best_bfu = Some(i);
            }
        }

        match (worst_bfu, best_bfu) {
            (Some(w), Some(b)) if w != b => {
                // Only swap if worst has at least 10x more NMR than best
                if max_nmr < min_nmr * 10.0 {
                    break;
                }

                // Budget check: ensure total doesn't exceed max
                let cost = SPECS_PER_BLOCK[w] as i32;
                let freed = SPECS_PER_BLOCK[b] as i32;
                let budget_change = cost - freed;

                // Allow ±20 bits slack (safety cap handles overflow later)
                if budget_change <= 20 {
                    bits_per_block[w] += 1;
                    bits_per_block[b] -= 1;
                } else {
                    break;
                }
            }
            _ => break,
        }
    }
}

/// Full bit allocation + bitstream write for one ATRAC1 frame.
/// Returns (frame_bytes, num_bfus_used).
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
        7
    };

    let sf_indices: Vec<u8> = scaled_blocks.iter().map(|b| b.scale_factor_index).collect();
    let spread = analyze_scale_factor_spread(&sf_indices);
    let booster = BitsBooster::new();

    let mut bits_per_block = vec![0u32; BFU_AMOUNT_TAB[bfu_idx as usize] as usize];
    let target_bits;
    let mut cur_bits;

    loop {
        bits_per_block.resize(BFU_AMOUNT_TAB[bfu_idx as usize] as usize, 0);

        let bits_available = SOUND_UNIT_SIZE * 8
            - BITS_PER_BFU_AMOUNT_TAB_IDX
            - 32
            - 2
            - 3
            - bits_per_block.len() * (BITS_PER_IDWL + BITS_PER_IDSF);

        let max_bits = bits_available;
        let min_bits = bits_available - 16;

        let mut max_shift: f32 = 15.0;
        let mut min_shift: f32 = -3.0;
        let mut shift: f32 = 3.0;

        let mut bfu_num_changed = false;

        loop {
            let tmp_alloc = calc_bits_allocation(
                scaled_blocks,
                BFU_AMOUNT_TAB[bfu_idx as usize] as usize,
                spread,
                shift,
                block_size,
                loudness,
            );

            let mut bits_used: u32 = 0;
            for (i, &b) in tmp_alloc.iter().enumerate() {
                bits_used += SPECS_PER_BLOCK[i] * b;
            }

            if bits_used < min_bits as u32 {
                if max_shift - min_shift < 0.1 {
                    if auto_bfu {
                        let used_id = get_max_used_bfu_id(&tmp_alloc);
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
                        bits_per_block = tmp_alloc;
                    }
                    cur_bits = bits_used;
                    break;
                }
                max_shift = shift;
                shift -= (shift - min_shift) / 2.0;
            } else if bits_used > max_bits as u32 {
                min_shift = shift;
                shift += (max_shift - shift) / 2.0;
            } else {
                if auto_bfu {
                    let used_id = get_max_used_bfu_id(&tmp_alloc);
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
                    bits_per_block = tmp_alloc;
                }
                cur_bits = bits_used;
                break;
            }
        }

        if !bfu_num_changed {
            target_bits = bits_available as u32;
            break;
        }
    }

    booster.apply_boost(&mut bits_per_block, cur_bits, target_bits);

    // Safety: verify total bits don't exceed frame capacity.
    // The boost can overshoot if BFUs go from 0→2 (double cost).
    let total_data: u32 = bits_per_block
        .iter()
        .enumerate()
        .map(|(i, &b)| SPECS_PER_BLOCK[i] * b)
        .sum();
    let max_data = (SOUND_UNIT_SIZE * 8
        - BITS_PER_BFU_AMOUNT_TAB_IDX
        - 32
        - 2
        - 3
        - bits_per_block.len() * (BITS_PER_IDWL + BITS_PER_IDSF)
        - 24) as u32; // subtract footer too
    if total_data > max_data {
        // Over budget — trim highest-numbered BFUs with bits until we fit
        let mut excess = total_data - max_data;
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

    // Greedy fill: use any remaining bits on the highest-energy BFU that fits
    let total_after: u32 = bits_per_block
        .iter()
        .enumerate()
        .map(|(i, &b)| SPECS_PER_BLOCK[i] * b)
        .sum();
    let mut remaining = max_data.saturating_sub(total_after);
    while remaining > 0 {
        // Find BFU where +1 bit gives the most benefit:
        // prioritize low-bit BFUs with signal (going from 2→3 bits = 50% more precision)
        let mut best: Option<(usize, f32)> = None;
        for i in 0..bits_per_block.len() {
            if bits_per_block[i] >= 16 || bits_per_block[i] < 2 {
                continue;
            }
            let cost = SPECS_PER_BLOCK[i];
            if cost > remaining {
                continue;
            }
            let energy = scaled_blocks[i].max_energy;
            if energy <= 0.0 {
                continue;
            }
            // Score: energy / current_bits — high energy + low bits = most benefit
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

    // Analysis-by-Synthesis: NMR-driven bit reallocation.
    // Measure quantization noise vs masking threshold per BFU,
    // then Robin Hood bits from over-masked to under-masked BFUs.
    let ath_long = &*ATH_LONG;
    abs_reallocate(&mut bits_per_block, scaled_blocks, ath_long, loudness);

    let frame = write_bitstream(&bits_per_block, scaled_blocks, bfu_idx, block_size);
    (frame, BFU_AMOUNT_TAB[bfu_idx as usize])
}

#[cfg(test)]
#[path = "../tests/atrac1_bitalloc_tests.rs"]
mod tests;
