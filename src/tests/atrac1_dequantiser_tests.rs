use super::*;
use crate::atrac1::{BFU_AMOUNT_TAB, BlockSizeMod, SPECS_PER_BLOCK};
use crate::bitstream::BitStream;

/// Helper: build a minimal bitstream for a frame with given BFU count,
/// word lengths, scale factors, and all-zero mantissas.
fn build_frame_bitstream(
    bfu_amount_idx: u32,
    word_lens: &[u32],
    scale_factors: &[u32],
    block_size: &BlockSizeMod,
) -> BitStream {
    let mut bs = BitStream::new();

    // Block size mode: 2+2+2+2 bits
    bs.write((2 - block_size.log_count[0]) as u32, 2);
    bs.write((2 - block_size.log_count[1]) as u32, 2);
    bs.write((3 - block_size.log_count[2]) as u32, 2);
    bs.write(0, 2); // unused

    // BFU amount index (3 bits)
    bs.write(bfu_amount_idx, 3);
    // Reserved (2 + 3 bits)
    bs.write(0, 2);
    bs.write(0, 3);

    let num_bfus = BFU_AMOUNT_TAB[bfu_amount_idx as usize] as usize;

    // Word lengths (4 bits each)
    for i in 0..num_bfus {
        bs.write(word_lens.get(i).copied().unwrap_or(0), 4);
    }

    // Scale factors (6 bits each)
    for i in 0..num_bfus {
        bs.write(scale_factors.get(i).copied().unwrap_or(0), 6);
    }

    // Mantissas: all zeros
    for i in 0..num_bfus {
        let wl = word_lens.get(i).copied().unwrap_or(0);
        let actual_wl = if wl > 0 { wl + 1 } else { 0 };
        let num_specs = SPECS_PER_BLOCK[i] as usize;
        for _ in 0..num_specs {
            if actual_wl > 0 {
                bs.write(0, actual_wl as usize);
            }
        }
    }

    bs
}

#[test]
fn test_dequant_all_zero_wordlens() {
    let block_size = BlockSizeMod::new();
    let num_bfus = BFU_AMOUNT_TAB[0] as usize; // 20 BFUs

    let word_lens = vec![0u32; num_bfus];
    let scale_factors = vec![30u32; num_bfus];

    let mut bs = build_frame_bitstream(0, &word_lens, &scale_factors, &block_size);

    // Skip block size mode (already written, read it to advance)
    bs.reset_read_pos();
    let parsed_bs = BlockSizeMod::from_bitstream(&mut bs);
    let _ = parsed_bs;

    // Reset and dequant from start (dequant reads after block size mode)
    bs.reset_read_pos();
    // Skip block size (8 bits)
    bs.read(8);

    let mut specs = [0.0f32; 512];
    dequant(&mut bs, &block_size, &mut specs);

    // All word lengths are 0, so all specs should be 0
    for (i, &v) in specs.iter().enumerate() {
        assert_eq!(0.0, v, "specs[{i}] should be 0 with zero word lengths");
    }
}

#[test]
fn test_dequant_produces_correct_bfu_count() {
    // Use BFU amount index 7 → 52 BFUs (maximum)
    let block_size = BlockSizeMod::new();
    let num_bfus = 52;

    let word_lens = vec![0u32; num_bfus];
    let scale_factors = vec![0u32; num_bfus];

    let mut bs = build_frame_bitstream(7, &word_lens, &scale_factors, &block_size);
    bs.reset_read_pos();
    bs.read(8); // skip block size

    let mut specs = [0.0f32; 512];
    dequant(&mut bs, &block_size, &mut specs);

    // Should not panic, all specs should be 0
    for &v in &specs {
        assert_eq!(0.0, v);
    }
}

#[test]
fn test_dequant_nonzero_mantissa() {
    let block_size = BlockSizeMod::new();
    let mut bs = BitStream::new();

    // Block size: all long
    bs.write(2, 2);
    bs.write(2, 2);
    bs.write(3, 2);
    bs.write(0, 2);

    // BFU amount index = 0 → 20 BFUs
    bs.write(0, 3);
    bs.write(0, 2);
    bs.write(0, 3); // reserved

    // Set word length = 2 for BFU 0, rest = 0
    // wl=2 → actual_wl = 3, maxQuant = 1/(2^2 - 1) = 1/3
    bs.write(2, 4); // BFU 0: wl=2
    for _ in 1..20 {
        bs.write(0, 4); // rest: wl=0
    }

    // Scale factor = 63 for BFU 0 (ScaleTable[63] = 1.0), rest = 0
    bs.write(63, 6);
    for _ in 1..20 {
        bs.write(0, 6);
    }

    // BFU 0 has 8 specs, actual_wl=3 → write 8 × 3-bit mantissas
    // Write mantissa = 1 (positive) for first spec
    bs.write(1, 3);
    // Write mantissa = 0b101 = 5 → MakeSign(5, 3) = -3 for second spec
    bs.write(5, 3);
    // Rest zeros
    for _ in 2..8 {
        bs.write(0, 3);
    }

    bs.reset_read_pos();
    bs.read(8); // skip block size

    let mut specs = [0.0f32; 512];
    dequant(&mut bs, &block_size, &mut specs);

    // BFU 0 starts at SpecsStartLong[0] = 0
    // ScaleTable[63] = 1.0, maxQuant = 1/3
    // specs[0] = 1.0 * (1/3) * MakeSign(1, 3) = 1/3 * 1 ≈ 0.333
    let expected_0 = 1.0 * (1.0 / 3.0) * 1.0;
    assert!(
        (specs[0] - expected_0).abs() < 1e-6,
        "specs[0] = {}, expected {expected_0}",
        specs[0]
    );

    // specs[1] = 1.0 * (1/3) * MakeSign(5, 3) = (1/3) * (-3) = -1.0
    let expected_1 = 1.0 * (1.0 / 3.0) * (-3.0);
    assert!(
        (specs[1] - expected_1).abs() < 1e-6,
        "specs[1] = {}, expected {expected_1}",
        specs[1]
    );

    // specs[2..8] should be 0 (mantissa = 0 → MakeSign(0,3) = 0)
    for i in 2..8 {
        assert_eq!(0.0, specs[i], "specs[{i}] should be 0");
    }
}

#[test]
fn test_dequant_short_window() {
    let block_size = BlockSizeMod::from_flags(true, false, false);
    let mut bs = BitStream::new();

    // Block size: low=short, mid=long, hi=long
    bs.write(0, 2); // low: 2-0=2 (short)
    bs.write(2, 2); // mid: 2-2=0 (long)
    bs.write(3, 2); // hi: 3-3=0 (long)
    bs.write(0, 2);

    // BFU amount = 0 → 20 BFUs, all wl=0
    bs.write(0, 3);
    bs.write(0, 2);
    bs.write(0, 3);
    for _ in 0..20 {
        bs.write(0, 4);
    }
    for _ in 0..20 {
        bs.write(0, 6);
    }

    bs.reset_read_pos();
    bs.read(8); // skip block size

    let mut specs = [0.0f32; 512];
    dequant(&mut bs, &block_size, &mut specs);

    // Should not panic — short window uses SpecsStartShort for low band
    for &v in &specs {
        assert_eq!(0.0, v);
    }
}

#[test]
fn test_dequant_scale_factor_effect() {
    // Two frames with same mantissas but different scale factors should differ
    let block_size = BlockSizeMod::new();

    let make_frame = |sf: u32| -> [f32; 512] {
        let mut bs = BitStream::new();
        bs.write(2, 2);
        bs.write(2, 2);
        bs.write(3, 2);
        bs.write(0, 2);
        bs.write(0, 3);
        bs.write(0, 2);
        bs.write(0, 3);

        // BFU 0: wl=1 → actual_wl=2, maxQuant=1/1=1
        bs.write(1, 4);
        for _ in 1..20 {
            bs.write(0, 4);
        }

        bs.write(sf, 6);
        for _ in 1..20 {
            bs.write(0, 6);
        }

        // 8 mantissas of 2 bits each, all = 1
        for _ in 0..8 {
            bs.write(1, 2);
        }

        bs.reset_read_pos();
        bs.read(8);

        let mut specs = [0.0f32; 512];
        dequant(&mut bs, &block_size, &mut specs);
        specs
    };

    let specs_low_sf = make_frame(20);
    let specs_high_sf = make_frame(50);

    // Higher scale factor → larger dequantized values
    assert!(
        specs_high_sf[0].abs() > specs_low_sf[0].abs(),
        "Higher SF should produce larger values: {} vs {}",
        specs_high_sf[0],
        specs_low_sf[0]
    );
}
