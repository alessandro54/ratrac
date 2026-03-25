use super::*;
use crate::atrac1::{BlockSizeMod, SOUND_UNIT_SIZE, SPECS_PER_BLOCK};
use crate::bitstream::BitStream;
use crate::scaler::{ScaledBlock, Scaler};

fn make_dummy_scaled_blocks(n: usize, sf_idx: u8) -> Vec<ScaledBlock> {
    (0..n)
        .map(|i| ScaledBlock {
            scale_factor_index: sf_idx,
            values: vec![0.1; SPECS_PER_BLOCK[i] as usize],
            max_energy: 0.01,
        })
        .collect()
}

// --- BitsBooster ---

#[test]
fn test_bits_booster_new() {
    let booster = BitsBooster::new();
    assert!(booster.max_bits_per_iteration > 0);
    assert!(booster.min_key > 0);
}

#[test]
fn test_bits_booster_apply_no_surplus() {
    let booster = BitsBooster::new();
    let mut bits = vec![4u32; 20];
    let surplus = booster.apply_boost(&mut bits, 100, 100);
    assert_eq!(0, surplus);
}

#[test]
fn test_bits_booster_apply_with_surplus() {
    let booster = BitsBooster::new();
    let mut bits = vec![4u32; 52];
    let initial_sum: u32 = bits
        .iter()
        .zip(SPECS_PER_BLOCK.iter())
        .map(|(&b, &s)| b * s)
        .sum();
    let target = initial_sum + 50;
    let remaining = booster.apply_boost(&mut bits, initial_sum, target);
    // Some bits should have been redistributed
    let new_sum: u32 = bits
        .iter()
        .zip(SPECS_PER_BLOCK.iter())
        .map(|(&b, &s)| b * s)
        .sum();
    assert!(new_sum >= initial_sum, "Sum should not decrease");
    assert_eq!(remaining, target - new_sum);
}

// --- WriteBitStream ---

#[test]
fn test_write_bitstream_frame_size() {
    let block_size = BlockSizeMod::new();
    let scaled_blocks = make_dummy_scaled_blocks(20, 30);
    let bits_per_block = vec![4u32; 20];

    let frame = write_bitstream(&bits_per_block, &scaled_blocks, 0, &block_size);
    assert_eq!(
        SOUND_UNIT_SIZE,
        frame.len(),
        "Frame must be exactly 212 bytes"
    );
}

#[test]
fn test_write_bitstream_zero_wordlens() {
    let block_size = BlockSizeMod::new();
    let scaled_blocks = make_dummy_scaled_blocks(20, 30);
    let bits_per_block = vec![0u32; 20];

    let frame = write_bitstream(&bits_per_block, &scaled_blocks, 0, &block_size);
    assert_eq!(SOUND_UNIT_SIZE, frame.len());
}

#[test]
fn test_write_bitstream_all_bfus() {
    let block_size = BlockSizeMod::new();
    let scaled_blocks = make_dummy_scaled_blocks(52, 30);
    let bits_per_block = vec![3u32; 52];

    let frame = write_bitstream(&bits_per_block, &scaled_blocks, 7, &block_size);
    assert_eq!(SOUND_UNIT_SIZE, frame.len());
}

// --- Write + Dequant roundtrip ---

#[test]
fn test_write_dequant_roundtrip() {
    use crate::atrac1::dequantiser::dequant;

    let block_size = BlockSizeMod::new();
    let scaled_blocks = make_dummy_scaled_blocks(20, 40);
    let bits_per_block = vec![6u32; 20];

    let frame = write_bitstream(&bits_per_block, &scaled_blocks, 0, &block_size);

    // Read back
    let mut bs = BitStream::from_bytes(&frame);
    let parsed_bs = BlockSizeMod::from_bitstream(&mut bs);

    let mut specs = [0.0f32; 512];
    dequant(&mut bs, &parsed_bs, &mut specs);

    // Verify non-zero specs exist in the first 128 positions (BFUs 0-19 = low band)
    let has_nonzero = specs[..128].iter().any(|&x| x != 0.0);
    assert!(has_nonzero, "Dequantised specs should have non-zero values");
}

// --- Full write_frame ---

#[test]
fn test_write_frame_produces_valid_frame() {
    let block_size = BlockSizeMod::new();
    let scaled_blocks = make_dummy_scaled_blocks(52, 30);

    let (frame, num_bfus) = write_frame(&scaled_blocks, &block_size, 0.5, 0, false);
    assert_eq!(SOUND_UNIT_SIZE, frame.len());
    assert!(num_bfus <= 52);
    assert!(num_bfus >= 20); // minimum from BfuAmountTab
}

#[test]
fn test_write_frame_with_fixed_bfu() {
    let block_size = BlockSizeMod::new();
    let scaled_blocks = make_dummy_scaled_blocks(52, 30);

    // bfu_idx_const = 1 → uses BfuAmountTab[0] = 20 BFUs
    let (frame, num_bfus) = write_frame(&scaled_blocks, &block_size, 0.5, 1, false);
    assert_eq!(SOUND_UNIT_SIZE, frame.len());
    assert_eq!(20, num_bfus);
}

#[test]
fn test_write_frame_short_windows() {
    let block_size = BlockSizeMod::from_flags(true, true, true);
    let scaled_blocks = make_dummy_scaled_blocks(52, 25);

    let (frame, num_bfus) = write_frame(&scaled_blocks, &block_size, 0.5, 0, false);
    assert_eq!(SOUND_UNIT_SIZE, frame.len());
    assert!(num_bfus >= 20);
}

#[test]
fn test_write_frame_silence() {
    let block_size = BlockSizeMod::new();
    let scaled_blocks: Vec<ScaledBlock> = (0..52)
        .map(|i| ScaledBlock {
            scale_factor_index: 0,
            values: vec![0.0; SPECS_PER_BLOCK[i] as usize],
            max_energy: 0.0,
        })
        .collect();

    let (frame, _) = write_frame(&scaled_blocks, &block_size, 0.0, 0, false);
    assert_eq!(SOUND_UNIT_SIZE, frame.len());
}

#[test]
fn test_write_frame_dequant_roundtrip() {
    use crate::atrac1::dequantiser::dequant;

    let block_size = BlockSizeMod::new();
    let scaler = Scaler::new();

    // Create a known spectrum
    let mut specs_in = vec![0.0f32; 512];
    for i in 0..512 {
        specs_in[i] = 0.3 * (i as f32 * 0.05).sin();
    }

    let scaled_blocks = scaler.scale_frame(&specs_in, &block_size);
    let (frame, _) = write_frame(&scaled_blocks, &block_size, 0.5, 0, false);

    // Decode
    let mut bs = BitStream::from_bytes(&frame);
    let parsed_bs = BlockSizeMod::from_bitstream(&mut bs);
    let mut specs_out = [0.0f32; 512];
    dequant(&mut bs, &parsed_bs, &mut specs_out);

    // Check reconstruction: not bit-exact (lossy!) but should be correlated
    let mut correlation = 0.0f64;
    let mut e_in = 0.0f64;
    let mut e_out = 0.0f64;
    for i in 0..512 {
        correlation += specs_in[i] as f64 * specs_out[i] as f64;
        e_in += (specs_in[i] as f64).powi(2);
        e_out += (specs_out[i] as f64).powi(2);
    }
    let norm_corr = correlation / (e_in.sqrt() * e_out.sqrt() + 1e-30);
    assert!(
        norm_corr > 0.8,
        "Normalized correlation {norm_corr} too low — encode/decode mismatch"
    );
}
