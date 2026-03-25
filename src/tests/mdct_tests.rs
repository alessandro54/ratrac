use super::*;

/// Matching C++ CalcEps: SNR-based epsilon for f32.
/// snr = -114.0 dB for f32
fn calc_eps(magn: f32) -> f32 {
    let snr: f32 = -114.0;
    magn * 10.0_f32.powf(snr / 20.0)
}

/// Naive O(N^2) forward MDCT reference. Input: 2N samples, output: N coefficients.
/// Matches C++ `mdct()` in mdct_ut.cpp.
fn naive_mdct(x: &[f32], n: usize) -> Vec<f32> {
    let mut res = Vec::with_capacity(n);
    for k in 0..n {
        let mut sum = 0.0f64;
        for i in 0..(2 * n) {
            sum += x[i] as f64
                * ((std::f64::consts::PI / n as f64)
                    * (i as f64 + 0.5 + n as f64 / 2.0)
                    * (k as f64 + 0.5))
                    .cos();
        }
        res.push(sum as f32);
    }
    res
}

/// Naive O(N^2) inverse MDCT reference. Input: N coefficients, output: 2N samples.
/// Matches C++ `midct()` in mdct_ut.cpp.
fn naive_midct(x: &[f32], n: usize) -> Vec<f32> {
    let mut res = Vec::with_capacity(2 * n);
    for i in 0..(2 * n) {
        let mut sum = 0.0f64;
        for k in 0..n {
            sum += x[k] as f64
                * ((std::f64::consts::PI / n as f64)
                    * (i as f64 + 0.5 + n as f64 / 2.0)
                    * (k as f64 + 0.5))
                    .cos();
        }
        res.push(sum as f32);
    }
    res
}

// --- Forward MDCT tests (translated from mdct_ut.cpp) ---

#[test]
fn test_mdct_32() {
    let n = 32;
    let mut transform = Mdct::new(n, n as f32);
    let src: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let ref_result = naive_mdct(&src, n / 2);
    let result = transform.process(&src);

    assert_eq!(ref_result.len(), result.len());
    let eps = calc_eps(n as f32);
    for i in 0..ref_result.len() {
        assert!(
            (ref_result[i] - result[i]).abs() <= eps,
            "MDCT32 mismatch at {i}: ref={} got={} eps={eps}",
            ref_result[i],
            result[i]
        );
    }
}

#[test]
fn test_mdct_64() {
    let n = 64;
    let mut transform = Mdct::new(n, n as f32);
    let src: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let ref_result = naive_mdct(&src, n / 2);
    let result = transform.process(&src);

    assert_eq!(ref_result.len(), result.len());
    let eps = calc_eps(n as f32);
    for i in 0..ref_result.len() {
        assert!(
            (ref_result[i] - result[i]).abs() <= eps,
            "MDCT64 mismatch at {i}: ref={} got={} eps={eps}",
            ref_result[i],
            result[i]
        );
    }
}

#[test]
fn test_mdct_128() {
    let n = 128;
    let mut transform = Mdct::new(n, n as f32);
    let src: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let ref_result = naive_mdct(&src, n / 2);
    let result = transform.process(&src);

    assert_eq!(ref_result.len(), result.len());
    let eps = calc_eps((n * 4) as f32); // C++ uses N*4 for 128
    for i in 0..ref_result.len() {
        assert!(
            (ref_result[i] - result[i]).abs() <= eps,
            "MDCT128 mismatch at {i}: ref={} got={} eps={eps}",
            ref_result[i],
            result[i]
        );
    }
}

#[test]
fn test_mdct_256() {
    let n = 256;
    let mut transform = Mdct::new(n, n as f32);
    let src: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let ref_result = naive_mdct(&src, n / 2);
    let result = transform.process(&src);

    assert_eq!(ref_result.len(), result.len());
    let eps = calc_eps((n * 4) as f32); // C++ uses N*4 for 256
    for i in 0..ref_result.len() {
        assert!(
            (ref_result[i] - result[i]).abs() <= eps,
            "MDCT256 mismatch at {i}: ref={} got={} eps={eps}",
            ref_result[i],
            result[i]
        );
    }
}

// --- Inverse MDCT tests ---

#[test]
fn test_midct_32() {
    let n = 32;
    let mut transform = Midct::new(n, n as f32);
    let src: Vec<f32> = (0..n)
        .map(|i| if i < n / 2 { i as f32 } else { 0.0 })
        .collect();
    let ref_result = naive_midct(&src, n / 2);
    let result = transform.process(&src[..n / 2]);

    assert_eq!(ref_result.len(), result.len());
    let eps = calc_eps(n as f32);
    for i in 0..n {
        assert!(
            (ref_result[i] - result[i]).abs() <= eps,
            "MIDCT32 mismatch at {i}: ref={} got={} eps={eps}",
            ref_result[i],
            result[i]
        );
    }
}

#[test]
fn test_midct_64() {
    let n = 64;
    let mut transform = Midct::new(n, n as f32);
    let src: Vec<f32> = (0..n / 2).map(|i| i as f32).collect();
    let ref_result = naive_midct(&src, n / 2);
    let result = transform.process(&src);

    assert_eq!(ref_result.len(), result.len());
    let eps = calc_eps(n as f32);
    for i in 0..n {
        assert!(
            (ref_result[i] - result[i]).abs() <= eps,
            "MIDCT64 mismatch at {i}: ref={} got={} eps={eps}",
            ref_result[i],
            result[i]
        );
    }
}

#[test]
fn test_midct_128() {
    let n = 128;
    let mut transform = Midct::new(n, n as f32);
    let src: Vec<f32> = (0..n / 2).map(|i| i as f32).collect();
    let ref_result = naive_midct(&src, n / 2);
    let result = transform.process(&src);

    assert_eq!(ref_result.len(), result.len());
    let eps = calc_eps(n as f32);
    for i in 0..n {
        assert!(
            (ref_result[i] - result[i]).abs() <= eps,
            "MIDCT128 mismatch at {i}: ref={} got={} eps={eps}",
            ref_result[i],
            result[i]
        );
    }
}

#[test]
fn test_midct_256() {
    let n = 256;
    let mut transform = Midct::new(n, n as f32);
    let src: Vec<f32> = (0..n / 2).map(|i| i as f32).collect();
    let ref_result = naive_midct(&src, n / 2);
    let result = transform.process(&src);

    assert_eq!(ref_result.len(), result.len());
    let eps = calc_eps((n * 2) as f32); // C++ uses N*2 for 256
    for i in 0..n {
        assert!(
            (ref_result[i] - result[i]).abs() <= eps,
            "MIDCT256 mismatch at {i}: ref={} got={} eps={eps}",
            ref_result[i],
            result[i]
        );
    }
}

// --- Round-trip test ---
// MDCT->IMDCT roundtrip requires overlap-add of consecutive frames.
// With scale=1.0 for forward and scale=N for inverse (ATRAC1 convention),
// the TDAC property gives: imdct(mdct(frame))[n/2..n] + imdct(mdct(next))[0..n/2] = original
// We use the naive reference to verify our implementation matches.

#[test]
fn test_mdct_midct_consistency_64() {
    // Verify that our MDCT and IMDCT match the naive implementations,
    // which is sufficient to guarantee correct roundtrip via TDAC.
    let n = 64;
    let mut mdct_fast = Mdct::new(n, n as f32);
    let mut midct_fast = Midct::new(n, n as f32);

    let src: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();

    // Forward: fast vs naive
    let coeffs_naive = naive_mdct(&src, n / 2);
    let coeffs_fast = mdct_fast.process(&src).to_vec();
    let eps = calc_eps(n as f32);
    for i in 0..n / 2 {
        assert!(
            (coeffs_naive[i] - coeffs_fast[i]).abs() <= eps,
            "Forward mismatch at {i}"
        );
    }

    // Inverse: fast vs naive
    let time_naive = naive_midct(&coeffs_fast, n / 2);
    let time_fast = midct_fast.process(&coeffs_fast).to_vec();
    for i in 0..n {
        assert!(
            (time_naive[i] - time_fast[i]).abs() <= eps,
            "Inverse mismatch at {i}"
        );
    }
}

// --- ATRAC1-specific sizes ---

#[test]
fn test_mdct_512() {
    let n = 512;
    let mut transform = Mdct::new(n, 1.0); // ATRAC1 uses scale=1.0 for 512
    let src: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
    let result = transform.process(&src);
    assert_eq!(n / 2, result.len());
    // Just verify no NaN/Inf
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "MDCT512 produced non-finite at {i}");
    }
}

#[test]
fn test_midct_512() {
    let n = 512;
    let mut transform = Midct::new(n, 1024.0); // ATRAC1 uses scale=1024 for 512
    let src: Vec<f32> = (0..n / 2).map(|i| (i as f32 * 0.01).sin()).collect();
    let result = transform.process(&src);
    assert_eq!(n, result.len());
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "MIDCT512 produced non-finite at {i}");
    }
}
