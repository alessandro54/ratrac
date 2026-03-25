use super::*;
use crate::atrac1::BlockSizeMod;

/// SNR-based epsilon matching C++ CalcEps.
fn calc_eps(magn: f32) -> f32 {
    let snr: f32 = -114.0;
    magn * 10.0_f32.powf(snr / 20.0)
}

/// Check 128-sample band: a[i] ≈ 4 * b[i+32] for i in 0..96.
/// Matches C++ CheckResult128.
fn check_result_128(a: &[f32], b: &[f32]) {
    let m = a.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
    let eps = calc_eps(m);

    for i in 0..96 {
        let expected = a[i];
        let actual = 4.0 * b[i + 32];
        assert!(
            (expected - actual).abs() <= eps,
            "128-band mismatch at {i}: expected {expected}, got {actual} (4*b[{}]={actual}), eps={eps}",
            i + 32
        );
    }
}

/// Check 256-sample band: a[i] ≈ 2 * b[i+32] for i in 0..192.
/// Matches C++ CheckResult256.
fn check_result_256(a: &[f32], b: &[f32]) {
    let m = a.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
    let eps = calc_eps(m);

    for i in 0..192 {
        let expected = a[i];
        let actual = 2.0 * b[i + 32];
        assert!(
            (expected - actual).abs() <= eps,
            "256-band mismatch at {i}: expected {expected}, got {actual} (2*b[{}]={actual}), eps={eps}",
            i + 32
        );
    }
}

// --- Translated from atracdenc_ut.cpp ---

#[test]
fn test_atrac1_mdct_long_enc_dec() {
    let mut mdct = Atrac1Mdct::new();

    let mut low = vec![0.0f32; 256 + 16];
    let mut mid = vec![0.0f32; 256 + 16];
    let mut hi = vec![0.0f32; 512 + 16];
    let mut specs = vec![0.0f32; 512];

    let mut low_res = vec![0.0f32; 256 + 16];
    let mut mid_res = vec![0.0f32; 256 + 16];
    let mut hi_res = vec![0.0f32; 512 + 16];

    for i in 0..128 {
        low[i] = i as f32;
        mid[i] = i as f32;
    }
    for i in 0..256 {
        hi[i] = i as f32;
    }

    let block_size = BlockSizeMod::from_flags(false, false, false);

    mdct.mdct(&mut specs, &mut low, &mut mid, &mut hi, &block_size);
    mdct.imdct(
        &mut specs,
        &block_size,
        &mut low_res,
        &mut mid_res,
        &mut hi_res,
    );

    check_result_128(&low[..256], &low_res);
    check_result_128(&mid[..256], &mid_res);
    check_result_256(&hi[..512], &hi_res);
}

#[test]
fn test_atrac1_mdct_short_enc_dec() {
    let mut mdct = Atrac1Mdct::new();

    let mut low = vec![0.0f32; 256 + 16];
    let mut mid = vec![0.0f32; 256 + 16];
    let mut hi = vec![0.0f32; 512 + 16];
    let mut specs = vec![0.0f32; 512];

    let mut low_res = vec![0.0f32; 256 + 16];
    let mut mid_res = vec![0.0f32; 256 + 16];
    let mut hi_res = vec![0.0f32; 512 + 16];

    for i in 0..128 {
        low[i] = i as f32;
        mid[i] = i as f32;
    }
    // Save copies since short window MDCT modifies the input buffer
    let low_copy = low.clone();
    let mid_copy = mid.clone();

    for i in 0..256 {
        hi[i] = i as f32;
    }
    let hi_copy = hi.clone();

    let block_size = BlockSizeMod::from_flags(true, true, true);

    mdct.mdct(&mut specs, &mut low, &mut mid, &mut hi, &block_size);
    mdct.imdct(
        &mut specs,
        &block_size,
        &mut low_res,
        &mut mid_res,
        &mut hi_res,
    );

    check_result_128(&low_copy[..256], &low_res);
    check_result_128(&mid_copy[..256], &mid_res);
    check_result_256(&hi_copy[..512], &hi_res);
}

// --- Additional tests ---

#[test]
fn test_atrac1_mdct_zeros() {
    let mut mdct = Atrac1Mdct::new();
    let mut low = vec![0.0f32; 256 + 16];
    let mut mid = vec![0.0f32; 256 + 16];
    let mut hi = vec![0.0f32; 512 + 16];
    let mut specs = vec![0.0f32; 512];

    let block_size = BlockSizeMod::new();
    mdct.mdct(&mut specs, &mut low, &mut mid, &mut hi, &block_size);

    for (i, &v) in specs.iter().enumerate() {
        assert!(
            v.abs() < 1e-10,
            "specs[{i}] should be ~0 for zero input, got {v}"
        );
    }
}

#[test]
fn test_atrac1_mdct_specs_finite() {
    let mut mdct = Atrac1Mdct::new();
    let mut low = vec![0.0f32; 256 + 16];
    let mut mid = vec![0.0f32; 256 + 16];
    let mut hi = vec![0.0f32; 512 + 16];
    let mut specs = vec![0.0f32; 512];

    // Fill with some signal
    for i in 0..128 {
        low[i] = (i as f32 * 0.1).sin();
        mid[i] = (i as f32 * 0.2).cos();
    }
    for i in 0..256 {
        hi[i] = (i as f32 * 0.05).sin();
    }

    let block_size = BlockSizeMod::new();
    mdct.mdct(&mut specs, &mut low, &mut mid, &mut hi, &block_size);

    for (i, &v) in specs.iter().enumerate() {
        assert!(v.is_finite(), "specs[{i}] is not finite");
    }
}

#[test]
fn test_atrac1_imdct_finite() {
    let mut mdct = Atrac1Mdct::new();
    let mut specs = vec![0.0f32; 512];
    for i in 0..512 {
        specs[i] = (i as f32 * 0.01).sin() * 0.5;
    }

    let mut low = vec![0.0f32; 256 + 16];
    let mut mid = vec![0.0f32; 256 + 16];
    let mut hi = vec![0.0f32; 512 + 16];
    let block_size = BlockSizeMod::new();

    mdct.imdct(&mut specs, &block_size, &mut low, &mut mid, &mut hi);

    for i in 0..128 {
        assert!(low[i].is_finite(), "low[{i}] not finite");
        assert!(mid[i].is_finite(), "mid[{i}] not finite");
    }
    for i in 0..256 {
        assert!(hi[i].is_finite(), "hi[{i}] not finite");
    }
}

#[test]
fn test_atrac1_mdct_mixed_windows() {
    let mut mdct = Atrac1Mdct::new();
    let mut low = vec![0.0f32; 256 + 16];
    let mut mid = vec![0.0f32; 256 + 16];
    let mut hi = vec![0.0f32; 512 + 16];
    let mut specs = vec![0.0f32; 512];

    for i in 0..128 {
        low[i] = i as f32;
        mid[i] = i as f32;
    }
    for i in 0..256 {
        hi[i] = i as f32;
    }

    // Mixed: low=short, mid=long, hi=short
    let block_size = BlockSizeMod::from_flags(true, false, true);
    mdct.mdct(&mut specs, &mut low, &mut mid, &mut hi, &block_size);

    for (i, &v) in specs.iter().enumerate() {
        assert!(v.is_finite(), "specs[{i}] not finite in mixed mode");
    }
}
