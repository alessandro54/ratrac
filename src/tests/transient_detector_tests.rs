use super::*;

// --- Translated from transient_detector_ut.cpp ---

#[test]
fn test_analyze_gain_simple() {
    let mut input = [0.0f32; 256];
    for i in 0..256 {
        if i <= 24 {
            input[i] = 1.0;
        } else if i > 24 && i <= 32 {
            input[i] = 8.0;
        } else if i > 32 && i <= 66 {
            input[i] = 128.0;
        } else {
            input[i] = 0.5;
        }
    }

    let res = analyze_gain(&input, 256, 32, false);
    assert_eq!(32, res.len());

    for i in 0..3 {
        assert_eq!(1.0, res[i], "res[{i}] should be 1.0");
    }
    for i in 3..4 {
        assert_eq!(8.0, res[i], "res[{i}] should be 8.0");
    }
    for i in 4..9 {
        assert_eq!(128.0, res[i], "res[{i}] should be 128.0");
    }
    for i in 9..32 {
        assert_eq!(0.5, res[i], "res[{i}] should be 0.5");
    }
}

// --- Additional tests ---

#[test]
fn test_calculate_rms_known() {
    // RMS of [3, 4] = sqrt((9+16)/2) = sqrt(12.5)
    let rms = calculate_rms(&[3.0, 4.0]);
    assert!((rms - 12.5_f32.sqrt()).abs() < 1e-6);
}

#[test]
fn test_calculate_rms_zeros() {
    assert_eq!(0.0, calculate_rms(&[0.0, 0.0, 0.0]));
}

#[test]
fn test_calculate_rms_constant() {
    let rms = calculate_rms(&[5.0, 5.0, 5.0, 5.0]);
    assert!((rms - 5.0).abs() < 1e-6);
}

#[test]
fn test_calculate_peak() {
    assert_eq!(5.0, calculate_peak(&[1.0, -5.0, 3.0, -2.0]));
    assert_eq!(0.0, calculate_peak(&[0.0, 0.0]));
    assert_eq!(1.0, calculate_peak(&[-1.0, 0.5, 0.3]));
}

#[test]
fn test_analyze_gain_rms_mode() {
    let input = vec![1.0f32; 64];
    let res = analyze_gain(&input, 64, 8, true);
    assert_eq!(8, res.len());
    for &v in &res {
        assert!((v - 1.0).abs() < 1e-6, "RMS of constant 1.0 should be 1.0");
    }
}

#[test]
fn test_hp_filter_dc_rejection() {
    // HP filter should reject DC (constant) signal
    let mut detector = TransientDetector::new(16, 128);
    let input = vec![1.0f32; 128];
    let mut output = vec![0.0f32; 128];
    detector.hp_filter(&input, &mut output);

    // After filtering, output should be near zero (DC removed)
    // Allow some transient at the start due to filter settling
    let tail_energy: f32 = output[FIR_LEN..].iter().map(|&x| x * x).sum();
    let tail_len = (128 - FIR_LEN) as f32;
    let rms = (tail_energy / tail_len).sqrt();
    assert!(
        rms < 0.01,
        "HP filter should reject DC, got RMS = {rms}"
    );
}

#[test]
fn test_detect_no_transient_steady() {
    let mut detector = TransientDetector::new(16, 128);
    let input = vec![0.1f32; 128];

    // Run several frames with steady signal
    for _ in 0..5 {
        let result = detector.detect(&input);
        // After settling, should not detect transients
        let _ = result; // first frames may detect due to initial zero energy
    }
    // After 5 frames of steady signal, should be stable
    let result = detector.detect(&input);
    assert!(!result, "Steady signal should not trigger transient");
}

#[test]
fn test_detect_transient_sudden_jump() {
    let mut detector = TransientDetector::new(16, 128);

    // Feed several frames of quiet signal
    let quiet = vec![0.001f32; 128];
    for _ in 0..5 {
        detector.detect(&quiet);
    }

    // Now feed a loud signal — should detect transient
    let mut loud = vec![0.001f32; 128];
    // Make the second half much louder
    for i in 64..128 {
        loud[i] = 10.0;
    }
    let result = detector.detect(&loud);
    assert!(result, "Sudden energy jump should trigger transient");
}

#[test]
fn test_detect_transient_sudden_drop() {
    let mut detector = TransientDetector::new(16, 128);

    // Feed several frames of loud signal
    let loud = vec![10.0f32; 128];
    for _ in 0..5 {
        detector.detect(&loud);
    }

    // Now feed quiet — should detect transient (falling threshold 20 dB)
    let mut dropping = vec![10.0f32; 128];
    for i in 64..128 {
        dropping[i] = 0.001;
    }
    let result = detector.detect(&dropping);
    assert!(result, "Sudden energy drop should trigger transient");
}

#[test]
fn test_last_transient_pos() {
    let mut detector = TransientDetector::new(16, 128);

    // Initial position should be 0
    assert_eq!(0, detector.last_transient_pos());

    // After detection, position should be updated
    let quiet = vec![0.001f32; 128];
    for _ in 0..5 {
        detector.detect(&quiet);
    }

    let mut loud = vec![0.001f32; 128];
    for i in 64..128 {
        loud[i] = 10.0;
    }
    detector.detect(&loud);
    // Position should be > 0 if transient detected
    // (exact position depends on which short block triggered)
    let pos = detector.last_transient_pos();
    assert!(pos > 0, "Transient position should be set after detection");
}

#[test]
fn test_detector_stability() {
    let mut detector = TransientDetector::new(32, 256);
    let signal: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();

    // Run many frames, check no panics or NaN
    for _ in 0..100 {
        let _ = detector.detect(&signal);
    }
    // Just verifying no crashes
}
