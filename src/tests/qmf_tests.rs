use super::*;

#[test]
fn test_tap_half_coefficients() {
    // Verify all 24 coefficients are present and non-zero
    assert_eq!(24, TAP_HALF.len());
    for (i, &v) in TAP_HALF.iter().enumerate() {
        assert!(v.is_finite(), "TAP_HALF[{i}] is not finite");
    }
    // Spot-check first and last
    assert!((TAP_HALF[0] - (-0.00001461907)).abs() < 1e-10);
    assert!((TAP_HALF[23] - 0.46424159).abs() < 1e-7);
}

#[test]
fn test_qmf_window_symmetry() {
    let qmf = Qmf::new(512);
    // Window should be symmetric: window[i] == window[47-i]
    for i in 0..24 {
        assert_eq!(
            qmf.qmf_window[i], qmf.qmf_window[47 - i],
            "QMF window not symmetric at {i}"
        );
    }
}

#[test]
fn test_qmf_window_values() {
    let qmf = Qmf::new(512);
    // window[i] = TapHalf[i] * 2.0
    for i in 0..24 {
        let expected = TAP_HALF[i] * 2.0;
        assert!(
            (qmf.qmf_window[i] - expected).abs() < 1e-10,
            "QMF window[{i}] mismatch"
        );
    }
}

#[test]
fn test_analysis_zeros() {
    let mut qmf = Qmf::new(512);
    let input = vec![0.0f32; 512];
    let mut lower = vec![0.0f32; 256];
    let mut upper = vec![0.0f32; 256];

    qmf.analysis(&input, &mut lower, &mut upper);

    for i in 0..256 {
        assert_eq!(0.0, lower[i], "lower[{i}] should be zero");
        assert_eq!(0.0, upper[i], "upper[{i}] should be zero");
    }
}

#[test]
fn test_synthesis_zeros() {
    let mut qmf = Qmf::new(512);
    let lower = vec![0.0f32; 256];
    let upper = vec![0.0f32; 256];
    let mut output = vec![0.0f32; 512];

    qmf.synthesis(&mut output, &lower, &upper);

    for i in 0..512 {
        assert_eq!(0.0, output[i], "output[{i}] should be zero");
    }
}

#[test]
fn test_analysis_output_sizes() {
    let mut qmf = Qmf::new(512);
    let input = vec![1.0f32; 512];
    let mut lower = vec![0.0f32; 256];
    let mut upper = vec![0.0f32; 256];

    qmf.analysis(&input, &mut lower, &mut upper);

    // Just check we get finite outputs
    for i in 0..256 {
        assert!(lower[i].is_finite(), "lower[{i}] not finite");
        assert!(upper[i].is_finite(), "upper[{i}] not finite");
    }
}

#[test]
fn test_analysis_256() {
    let mut qmf = Qmf::new(256);
    let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut lower = vec![0.0f32; 128];
    let mut upper = vec![0.0f32; 128];

    qmf.analysis(&input, &mut lower, &mut upper);

    for i in 0..128 {
        assert!(lower[i].is_finite(), "lower[{i}] not finite");
        assert!(upper[i].is_finite(), "upper[{i}] not finite");
    }
}

#[test]
fn test_analysis_synthesis_roundtrip_512() {
    // After a few warm-up frames, analysis->synthesis should reconstruct
    // the signal with reasonable fidelity (perfect reconstruction requires
    // many frames for the filter to settle).
    let mut qmf_a = Qmf::new(512);
    let mut qmf_s = Qmf::new(512);

    let mut lower = vec![0.0f32; 256];
    let mut upper = vec![0.0f32; 256];
    let mut output = vec![0.0f32; 512];

    // Generate a test signal
    let signal: Vec<f32> = (0..512).map(|i| (i as f32 * 0.05).sin()).collect();

    // Warm up with several frames
    for _ in 0..10 {
        qmf_a.analysis(&signal, &mut lower, &mut upper);
        qmf_s.synthesis(&mut output, &lower, &upper);
    }

    // After warmup, check reconstruction quality
    qmf_a.analysis(&signal, &mut lower, &mut upper);
    qmf_s.synthesis(&mut output, &lower, &upper);

    // The QMF analysis/synthesis pair has a known gain of 2x (energy gain 4x).
    // This is inherent to the filter design and compensated in the codec.
    let input_energy: f32 = signal.iter().map(|&x| x * x).sum();
    let output_energy: f32 = output.iter().map(|&x| x * x).sum();

    let ratio = output_energy / input_energy;
    assert!(
        (ratio - 4.0).abs() < 0.1,
        "Energy ratio {ratio} should be ~4.0 (2x amplitude gain)"
    );
}

#[test]
fn test_analysis_synthesis_roundtrip_256() {
    let mut qmf_a = Qmf::new(256);
    let mut qmf_s = Qmf::new(256);

    let mut lower = vec![0.0f32; 128];
    let mut upper = vec![0.0f32; 128];
    let mut output = vec![0.0f32; 256];

    let signal: Vec<f32> = (0..256).map(|i| (i as f32 * 0.05).sin()).collect();

    for _ in 0..10 {
        qmf_a.analysis(&signal, &mut lower, &mut upper);
        qmf_s.synthesis(&mut output, &lower, &upper);
    }

    qmf_a.analysis(&signal, &mut lower, &mut upper);
    qmf_s.synthesis(&mut output, &lower, &upper);

    let input_energy: f32 = signal.iter().map(|&x| x * x).sum();
    let output_energy: f32 = output.iter().map(|&x| x * x).sum();

    let ratio = output_energy / input_energy;
    assert!(
        (ratio - 4.0).abs() < 0.1,
        "Energy ratio {ratio} should be ~4.0 (2x amplitude gain)"
    );
}

#[test]
fn test_analysis_state_carries_between_calls() {
    let mut qmf = Qmf::new(512);
    let input1 = vec![1.0f32; 512];
    let input2 = vec![0.0f32; 512];
    let mut lower = vec![0.0f32; 256];
    let mut upper = vec![0.0f32; 256];

    qmf.analysis(&input1, &mut lower, &mut upper);

    // Second call with zeros should still have non-zero output
    // due to the 46-sample history buffer
    qmf.analysis(&input2, &mut lower, &mut upper);

    let has_nonzero = lower.iter().chain(upper.iter()).any(|&x| x != 0.0);
    assert!(has_nonzero, "History buffer should cause non-zero output from zeros after non-zero input");
}
