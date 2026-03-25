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

        let threshold = ath;

        if !short_block && scaled_blocks[i].max_energy < threshold {
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
        let min_bits = bits_available - 110;

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

    let frame = write_bitstream(&bits_per_block, scaled_blocks, bfu_idx, block_size);
    (frame, BFU_AMOUNT_TAB[bfu_idx as usize])
}

#[cfg(test)]
#[path = "../tests/atrac1_bitalloc_tests.rs"]
mod tests;
