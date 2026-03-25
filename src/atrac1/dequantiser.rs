//! ATRAC1 dequantiser: reconstructs spectral coefficients from a bitstream.
//!
//! The decoder reads an AEA frame (212 bytes) and reconstructs 512 spectral
//! coefficients. Each BFU (Basic Functional Unit) has:
//! - A word length (0-15 bits per coefficient)
//! - A scale factor index (6-bit, indexes into ScaleTable)
//! - Quantized mantissas (variable bits)
//!
//! Reconstruction formula per coefficient:
//! ```text
//! spec[i] = ScaleTable[sf_idx] × (1 / (2^(wl-1) - 1)) × MakeSign(mantissa, wl)
//! ```

use crate::atrac1::{
    BFU_AMOUNT_TAB, BLOCKS_PER_BAND, BlockSizeMod, MAX_BFUS, NUM_QMF, SCALE_TABLE, SPECS_PER_BLOCK,
    SPECS_START_LONG, SPECS_START_SHORT,
};
use crate::bitstream::{BitStream, make_sign};

/// Dequantise an ATRAC1 frame from a bitstream into 512 spectral coefficients.
///
/// Reads from the bitstream:
/// - BFU count (3 bits → index into BfuAmountTab)
/// - Skip 5 reserved bits (2 + 3)
/// - Word lengths (4 bits × numBFUs)
/// - Scale factor indices (6 bits × numBFUs)
/// - Quantized mantissas (variable bits)
///
/// Reconstructs: `specs[i] = ScaleTable[sf] * (1 / (2^(wl-1) - 1)) * MakeSign(read(wl), wl)`
pub fn dequant(stream: &mut BitStream, bs: &BlockSizeMod, specs: &mut [f32; 512]) {
    let num_bfus = BFU_AMOUNT_TAB[stream.read(3) as usize] as usize;
    stream.read(2); // skip reserved
    stream.read(3); // skip reserved

    let mut word_lens = [0u32; MAX_BFUS];
    let mut id_scale_factors = [0u32; MAX_BFUS];

    for i in 0..num_bfus {
        word_lens[i] = stream.read(4);
    }

    for i in 0..num_bfus {
        id_scale_factors[i] = stream.read(6);
    }

    // Unused BFUs get zero word length and scale factor
    for i in num_bfus..MAX_BFUS {
        word_lens[i] = 0;
        id_scale_factors[i] = 0;
    }

    let scale_table = &*SCALE_TABLE;

    for band in 0..NUM_QMF {
        let start_bfu = BLOCKS_PER_BAND[band] as usize;
        let end_bfu = BLOCKS_PER_BAND[band + 1] as usize;

        for bfu in start_bfu..end_bfu {
            let num_specs = SPECS_PER_BLOCK[bfu] as usize;
            // C++: !!wordLens[bfu] + wordLens[bfu] → if wl>0 then wl+1, else 0
            let word_len = if word_lens[bfu] != 0 {
                word_lens[bfu] + 1
            } else {
                0
            } as usize;
            let scale_factor = scale_table[id_scale_factors[bfu] as usize];
            let start_pos = if bs.log_count[band] != 0 {
                SPECS_START_SHORT[bfu] as usize
            } else {
                SPECS_START_LONG[bfu] as usize
            };

            if word_len > 0 {
                let max_quant = 1.0 / ((1u32 << (word_len - 1)) - 1) as f32;
                for i in 0..num_specs {
                    let raw = stream.read(word_len) as i32;
                    specs[start_pos + i] =
                        scale_factor * max_quant * make_sign(raw, word_len as u32) as f32;
                }
            } else {
                for i in 0..num_specs {
                    specs[start_pos + i] = 0.0;
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "../tests/atrac1_dequantiser_tests.rs"]
mod tests;
