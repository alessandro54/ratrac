use super::*;

// --- ATH formula ---

#[test]
fn test_ath_tab_length() {
    assert_eq!(140, ATH_TAB.len());
}

#[test]
fn test_ath_formula_1khz() {
    // 1 kHz is in the most sensitive range of hearing
    let val = ath_formula_frank(1000.0);
    // Should be around 3.12 dB (tab index ~80: 312 millibel)
    assert!(
        val > 2.0 && val < 5.0,
        "ATH at 1kHz = {val}, expected ~3.12"
    );
}

#[test]
fn test_ath_formula_100hz() {
    let val = ath_formula_frank(100.0);
    // Should be around 26.58 dB (tab[40] = 2658 millibel)
    assert!(
        val > 20.0 && val < 30.0,
        "ATH at 100Hz = {val}, expected ~26.58"
    );
}

#[test]
fn test_ath_formula_10khz() {
    let val = ath_formula_frank(10000.0);
    // Should be around 14.79 dB (tab[120] = 1479 millibel)
    assert!(
        val > 10.0 && val < 20.0,
        "ATH at 10kHz = {val}, expected ~14.79"
    );
}

#[test]
fn test_ath_formula_4khz() {
    // 4 kHz is near the ear canal resonance, most sensitive
    let val = ath_formula_frank(4000.0);
    // tab index ~104: around -513 to -476 millibel → negative dB
    assert!(
        val < 0.0,
        "ATH at 4kHz = {val}, expected negative (most sensitive)"
    );
}

#[test]
fn test_ath_formula_clamping() {
    // Below 10 Hz should clamp to 10 Hz
    let val_low = ath_formula_frank(1.0);
    let val_10 = ath_formula_frank(10.0);
    assert!((val_low - val_10).abs() < 0.01, "Should clamp to 10 Hz");

    // Above 29853 Hz should clamp
    let val_high = ath_formula_frank(50000.0);
    let val_max = ath_formula_frank(29853.0);
    assert!(
        (val_high - val_max).abs() < 0.01,
        "Should clamp to 29853 Hz"
    );
}

#[test]
fn test_ath_formula_monotonic_low_range() {
    // ATH should generally decrease from 10 Hz to ~4 kHz (ear gets more sensitive)
    let v_10 = ath_formula_frank(10.0);
    let v_100 = ath_formula_frank(100.0);
    let v_1000 = ath_formula_frank(1000.0);
    let v_4000 = ath_formula_frank(4000.0);
    assert!(v_10 > v_100, "10Hz should be less sensitive than 100Hz");
    assert!(v_100 > v_1000, "100Hz should be less sensitive than 1kHz");
    assert!(v_1000 > v_4000, "1kHz should be less sensitive than 4kHz");
}

// --- CalcATH ---

#[test]
fn test_calc_ath_length() {
    let ath = calc_ath(256, 44100);
    assert_eq!(256, ath.len());
}

#[test]
fn test_calc_ath_all_finite() {
    let ath = calc_ath(512, 44100);
    for (i, &v) in ath.iter().enumerate() {
        assert!(v.is_finite(), "calc_ath[{i}] is not finite: {v}");
    }
}

#[test]
fn test_calc_ath_low_bins_higher_than_mid() {
    // Low frequency bins should have higher threshold than mid-range (less sensitive)
    let ath = calc_ath(256, 44100);
    // Bin 0 (~86 Hz) should be higher than bin ~50 (~4 kHz range)
    assert!(
        ath[0] > ath[50],
        "Low freq threshold {} should be > mid freq threshold {}",
        ath[0],
        ath[50]
    );
}

// --- Loudness curve ---

#[test]
fn test_loudness_curve_length() {
    let curve = create_loudness_curve(256);
    assert_eq!(256, curve.len());
}

#[test]
fn test_loudness_curve_all_finite_positive() {
    let curve = create_loudness_curve(512);
    for (i, &v) in curve.iter().enumerate() {
        assert!(v.is_finite(), "loudness_curve[{i}] not finite");
        assert!(v > 0.0, "loudness_curve[{i}] should be positive, got {v}");
    }
}

#[test]
fn test_loudness_curve_peak_in_midrange() {
    // Loudness weighting should peak in the 2-5 kHz range
    let curve = create_loudness_curve(512);
    let max_idx = curve
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    // Bin to frequency: f = (i+3) * 0.5 * 44100 / 512
    let peak_freq = (max_idx + 3) as f32 * 0.5 * 44100.0 / 512.0;
    assert!(
        peak_freq > 1000.0 && peak_freq < 6000.0,
        "Peak at {peak_freq} Hz, expected 1-6 kHz"
    );
}

// --- Scale factor spread ---

#[test]
fn test_spread_uniform() {
    // All same scale factors → sigma=0 → spread=0
    let indices = vec![30u8; 20];
    let spread = analyze_scale_factor_spread(&indices);
    assert!(
        (spread - 0.0).abs() < 1e-6,
        "Uniform should give 0, got {spread}"
    );
}

#[test]
fn test_spread_varied() {
    // Highly varied scale factors → high spread
    let indices: Vec<u8> = (0..20).map(|i| (i * 3) as u8).collect(); // 0,3,6,...,57
    let spread = analyze_scale_factor_spread(&indices);
    assert!(
        spread > 0.5,
        "Varied factors should give high spread, got {spread}"
    );
}

#[test]
fn test_spread_clamped_to_one() {
    // Extreme variation should clamp at 1.0
    let indices: Vec<u8> = vec![0, 63, 0, 63, 0, 63, 0, 63, 0, 63];
    let spread = analyze_scale_factor_spread(&indices);
    assert!(spread <= 1.0, "Spread should be <= 1.0, got {spread}");
}

#[test]
fn test_spread_range() {
    // Any input should return [0, 1]
    let indices: Vec<u8> = (0..52).map(|i| (i % 64) as u8).collect();
    let spread = analyze_scale_factor_spread(&indices);
    assert!(
        (0.0..=1.0).contains(&spread),
        "Spread {spread} out of [0,1]"
    );
}

// --- Loudness tracking ---

#[test]
fn test_track_loudness_stereo() {
    // Starting from 0, with l0=1, l1=1: 0.98*0 + 0.01*(1+1) = 0.02
    let result = track_loudness_stereo(0.0, 1.0, 1.0);
    assert!((result - 0.02).abs() < 1e-6);
}

#[test]
fn test_track_loudness_stereo_convergence() {
    // Repeated tracking with constant input should converge
    let mut loud = 0.0f32;
    for _ in 0..1000 {
        loud = track_loudness_stereo(loud, 1.0, 1.0);
    }
    // Should converge to 0.01*(1+1)/0.02 = 1.0
    assert!(
        (loud - 1.0).abs() < 0.01,
        "Should converge to 1.0, got {loud}"
    );
}

#[test]
fn test_track_loudness_mono() {
    let result = track_loudness_mono(0.0, 1.0);
    assert!((result - 0.02).abs() < 1e-6);
}

#[test]
fn test_track_loudness_mono_convergence() {
    let mut loud = 0.0f32;
    for _ in 0..1000 {
        loud = track_loudness_mono(loud, 1.0);
    }
    assert!(
        (loud - 1.0).abs() < 0.01,
        "Should converge to 1.0, got {loud}"
    );
}
