use super::*;
use crate::bitstream::BitStream;

// --- Constant table validation ---

#[test]
fn test_specs_per_block_sum_equals_512() {
    let sum: u32 = SPECS_PER_BLOCK.iter().sum();
    assert_eq!(512, sum);
}

#[test]
fn test_specs_per_block_band_sums() {
    // Low band (BFUs 0..20): 128 specs
    let low: u32 = SPECS_PER_BLOCK[0..20].iter().sum();
    assert_eq!(128, low);

    // Mid band (BFUs 20..36): 128 specs
    let mid: u32 = SPECS_PER_BLOCK[20..36].iter().sum();
    assert_eq!(128, mid);

    // High band (BFUs 36..52): 256 specs
    let hi: u32 = SPECS_PER_BLOCK[36..52].iter().sum();
    assert_eq!(256, hi);
}

#[test]
fn test_specs_start_long_contiguous() {
    // Within each band, SpecsStartLong[i] + SpecsPerBlock[i] == SpecsStartLong[i+1]
    for band in 0..3 {
        let start = BLOCKS_PER_BAND[band] as usize;
        let end = BLOCKS_PER_BAND[band + 1] as usize;
        for i in start..end - 1 {
            assert_eq!(
                SPECS_START_LONG[i] + SPECS_PER_BLOCK[i],
                SPECS_START_LONG[i + 1],
                "Contiguity broken at BFU {i}"
            );
        }
    }
}

#[test]
fn test_specs_start_long_band_boundaries() {
    assert_eq!(0, SPECS_START_LONG[0]);      // low starts at 0
    assert_eq!(128, SPECS_START_LONG[20]);   // mid starts at 128
    assert_eq!(256, SPECS_START_LONG[36]);   // high starts at 256
}

#[test]
fn test_specs_start_long_last_bfu_end() {
    // Last BFU ends at 512
    let last = MAX_BFUS - 1;
    assert_eq!(512, SPECS_START_LONG[last] + SPECS_PER_BLOCK[last]);
}

#[test]
fn test_blocks_per_band() {
    assert_eq!([0, 20, 36, 52], BLOCKS_PER_BAND);
}

#[test]
fn test_bfu_amount_tab() {
    assert_eq!([20, 28, 32, 36, 40, 44, 48, 52], BFU_AMOUNT_TAB);
    // All values are within valid BFU range
    for &v in &BFU_AMOUNT_TAB {
        assert!(v <= MAX_BFUS as u32);
    }
}

// --- Scale table ---

#[test]
fn test_scale_table_length() {
    assert_eq!(64, SCALE_TABLE.len());
}

#[test]
fn test_scale_table_formula() {
    // ScaleTable[i] = 2^(i/3 - 21)
    // Spot-check several indices
    let table = &*SCALE_TABLE;

    // i=0: 2^(-21) = 4.76837158e-7
    assert!((table[0] - 2.0_f64.powf(-21.0) as f32).abs() < 1e-13);

    // i=63: 2^(63/3 - 21) = 2^0 = 1.0
    assert!((table[63] - 1.0).abs() < 1e-7);

    // i=21*3 = 63 -> 2^0 = 1.0 (same as above)
    // i=42: 2^(42/3 - 21) = 2^(14-21) = 2^(-7) = 0.0078125
    assert!((table[42] - 0.0078125).abs() < 1e-9);

    // i=30: 2^(10 - 21) = 2^(-11) = 0.000488281...
    assert!((table[30] - 2.0_f64.powf(-11.0) as f32).abs() < 1e-10);
}

#[test]
fn test_scale_table_monotonically_increasing() {
    let table = &*SCALE_TABLE;
    for i in 1..64 {
        assert!(table[i] > table[i - 1], "ScaleTable not monotonic at index {i}");
    }
}

// --- Sine window ---

#[test]
fn test_sine_window_length() {
    assert_eq!(32, SINE_WINDOW.len());
}

#[test]
fn test_sine_window_formula() {
    let win = &*SINE_WINDOW;

    // i=0: sin(0.5 * PI/64) = sin(PI/128)
    let expected_0 = (0.5_f64 * std::f64::consts::PI / 64.0).sin() as f32;
    assert!((win[0] - expected_0).abs() < 1e-7);

    // i=15: sin(15.5 * PI/64)
    let expected_15 = (15.5_f64 * std::f64::consts::PI / 64.0).sin() as f32;
    assert!((win[15] - expected_15).abs() < 1e-7);

    // i=31: sin(31.5 * PI/64)
    let expected_31 = (31.5_f64 * std::f64::consts::PI / 64.0).sin() as f32;
    assert!((win[31] - expected_31).abs() < 1e-7);
}

#[test]
fn test_sine_window_range() {
    let win = &*SINE_WINDOW;
    for (i, &v) in win.iter().enumerate() {
        assert!(v > 0.0 && v <= 1.0, "SineWindow[{i}] = {v} out of range (0, 1]");
    }
}

// --- bfu_to_band ---

#[test]
fn test_bfu_to_band() {
    // Low band: 0..20
    for i in 0..20 {
        assert_eq!(0, bfu_to_band(i), "BFU {i} should be band 0");
    }
    // Mid band: 20..36
    for i in 20..36 {
        assert_eq!(1, bfu_to_band(i), "BFU {i} should be band 1");
    }
    // High band: 36..52
    for i in 36..52 {
        assert_eq!(2, bfu_to_band(i), "BFU {i} should be band 2");
    }
}

// --- BlockSizeMod ---

#[test]
fn test_block_size_mod_default() {
    let bsm = BlockSizeMod::new();
    assert_eq!([0, 0, 0], bsm.log_count);
    assert!(!bsm.short_win(0));
    assert!(!bsm.short_win(1));
    assert!(!bsm.short_win(2));
}

#[test]
fn test_block_size_mod_from_flags_all_long() {
    let bsm = BlockSizeMod::from_flags(false, false, false);
    assert_eq!([0, 0, 0], bsm.log_count);
}

#[test]
fn test_block_size_mod_from_flags_all_short() {
    let bsm = BlockSizeMod::from_flags(true, true, true);
    assert_eq!([2, 2, 3], bsm.log_count);
    assert!(bsm.short_win(0));
    assert!(bsm.short_win(1));
    assert!(bsm.short_win(2));
}

#[test]
fn test_block_size_mod_from_flags_mixed() {
    let bsm = BlockSizeMod::from_flags(true, false, true);
    assert_eq!([2, 0, 3], bsm.log_count);
    assert!(bsm.short_win(0));
    assert!(!bsm.short_win(1));
    assert!(bsm.short_win(2));
}

#[test]
fn test_block_size_mod_from_bitstream_all_long() {
    // All long: low=2(2-2=0), mid=2(2-2=0), hi=3(3-3=0), skip=0
    // Write: 2 in 2 bits, 2 in 2 bits, 3 in 2 bits, 0 in 2 bits
    let mut bs = BitStream::new();
    bs.write(2, 2); // low: 2 -> 2-2=0
    bs.write(2, 2); // mid: 2 -> 2-2=0
    bs.write(3, 2); // hi: 3 -> 3-3=0
    bs.write(0, 2); // unused
    bs.reset_read_pos();

    let bsm = BlockSizeMod::from_bitstream(&mut bs);
    assert_eq!([0, 0, 0], bsm.log_count);
}

#[test]
fn test_block_size_mod_from_bitstream_all_short() {
    // All short: low=0(2-0=2), mid=0(2-0=2), hi=0(3-0=3), skip=0
    let mut bs = BitStream::new();
    bs.write(0, 2); // low: 0 -> 2-0=2
    bs.write(0, 2); // mid: 0 -> 2-0=2
    bs.write(0, 2); // hi: 0 -> 3-0=3
    bs.write(0, 2); // unused
    bs.reset_read_pos();

    let bsm = BlockSizeMod::from_bitstream(&mut bs);
    assert_eq!([2, 2, 3], bsm.log_count);
}

#[test]
fn test_block_size_mod_bitstream_roundtrip() {
    // Write a block size mode as the encoder would, then parse it back
    // Encoder writes: (2 - log_count[0]), (2 - log_count[1]), (3 - log_count[2]), 0
    let original = BlockSizeMod::from_flags(true, false, true); // [2, 0, 3]

    let mut bs = BitStream::new();
    bs.write((2 - original.log_count[0]) as u32, 2);
    bs.write((2 - original.log_count[1]) as u32, 2);
    bs.write((3 - original.log_count[2]) as u32, 2);
    bs.write(0, 2);
    bs.reset_read_pos();

    let parsed = BlockSizeMod::from_bitstream(&mut bs);
    assert_eq!(original.log_count, parsed.log_count);
}

// --- Encode settings ---

#[test]
fn test_encode_settings_default() {
    let settings = Atrac1EncodeSettings::default();
    assert_eq!(0, settings.bfu_idx_const);
    assert!(!settings.fast_bfu_num_search);
    assert_eq!(WindowMode::Auto, settings.window_mode);
    assert_eq!(0, settings.window_mask);
}
