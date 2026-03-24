use super::*;

#[test]
fn test_analysis_zeros() {
    let mut fb = Atrac1AnalysisFilterBank::new();
    let pcm = vec![0.0f32; 512];
    let mut low = vec![0.0f32; 128];
    let mut mid = vec![0.0f32; 128];
    let mut hi = vec![0.0f32; 256];

    fb.analysis(&pcm, &mut low, &mut mid, &mut hi);

    for i in 0..128 {
        assert_eq!(0.0, low[i], "low[{i}] should be zero");
        assert_eq!(0.0, mid[i], "mid[{i}] should be zero");
    }
    for i in 0..256 {
        assert_eq!(0.0, hi[i], "hi[{i}] should be zero");
    }
}

#[test]
fn test_synthesis_zeros() {
    let mut fb = Atrac1SynthesisFilterBank::new();
    let low = vec![0.0f32; 128];
    let mid = vec![0.0f32; 128];
    let hi = vec![0.0f32; 256];
    let mut pcm = vec![0.0f32; 512];

    fb.synthesis(&mut pcm, &low, &mid, &hi);

    for i in 0..512 {
        assert_eq!(0.0, pcm[i], "pcm[{i}] should be zero");
    }
}

#[test]
fn test_analysis_output_sizes() {
    let mut fb = Atrac1AnalysisFilterBank::new();
    let pcm: Vec<f32> = (0..512).map(|i| (i as f32 * 0.02).sin()).collect();
    let mut low = vec![0.0f32; 128];
    let mut mid = vec![0.0f32; 128];
    let mut hi = vec![0.0f32; 256];

    fb.analysis(&pcm, &mut low, &mut mid, &mut hi);

    // All outputs should be finite
    for i in 0..128 {
        assert!(low[i].is_finite(), "low[{i}] not finite");
        assert!(mid[i].is_finite(), "mid[{i}] not finite");
    }
    for i in 0..256 {
        assert!(hi[i].is_finite(), "hi[{i}] not finite");
    }
}

#[test]
fn test_analysis_produces_nonzero() {
    let mut fb = Atrac1AnalysisFilterBank::new();
    let pcm: Vec<f32> = (0..512).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut low = vec![0.0f32; 128];
    let mut mid = vec![0.0f32; 128];
    let mut hi = vec![0.0f32; 256];

    // Run a couple of frames so filter settles
    fb.analysis(&pcm, &mut low, &mut mid, &mut hi);
    fb.analysis(&pcm, &mut low, &mut mid, &mut hi);

    let has_low = low.iter().any(|&x| x != 0.0);
    let has_mid = mid.iter().any(|&x| x != 0.0);
    let has_hi = hi.iter().any(|&x| x != 0.0);

    assert!(has_low, "Low band should have non-zero output");
    assert!(has_mid, "Mid band should have non-zero output");
    assert!(has_hi, "High band should have non-zero output");
}

#[test]
fn test_analysis_synthesis_roundtrip() {
    let mut analysis = Atrac1AnalysisFilterBank::new();
    let mut synthesis = Atrac1SynthesisFilterBank::new();

    let signal: Vec<f32> = (0..512).map(|i| (i as f32 * 0.05).sin()).collect();
    let mut low = vec![0.0f32; 128];
    let mut mid = vec![0.0f32; 128];
    let mut hi = vec![0.0f32; 256];
    let mut pcm_out = vec![0.0f32; 512];

    // Warm up filters with several frames
    for _ in 0..20 {
        analysis.analysis(&signal, &mut low, &mut mid, &mut hi);
        synthesis.synthesis(&mut pcm_out, &low, &mid, &hi);
    }

    // After warmup, check energy preservation
    // The 3-band QMF has gain from two stages (analysis * synthesis at each level)
    let input_energy: f32 = signal.iter().map(|&x| x * x).sum();
    let output_energy: f32 = pcm_out.iter().map(|&x| x * x).sum();

    // Two QMF stages: first stage 4x, second stage 4x on low/mid portion
    // Total gain depends on signal distribution across bands
    // Just check output is reasonable (non-zero, finite, same order of magnitude)
    assert!(output_energy > 0.0, "Output should have energy");
    assert!(output_energy.is_finite(), "Output energy should be finite");

    let ratio = output_energy / input_energy;
    assert!(
        ratio > 0.1 && ratio < 100.0,
        "Energy ratio {ratio} seems unreasonable"
    );
}

#[test]
fn test_delay_compensation() {
    // The delay buffer is 39 samples. Verify it shifts correctly.
    let mut fb = Atrac1AnalysisFilterBank::new();

    let pcm1: Vec<f32> = (0..512).map(|i| (i + 1) as f32).collect();
    let pcm2: Vec<f32> = (0..512).map(|i| (i + 513) as f32).collect();

    let mut low = vec![0.0f32; 128];
    let mut mid = vec![0.0f32; 128];
    let mut hi1 = vec![0.0f32; 256];
    let mut hi2 = vec![0.0f32; 256];

    fb.analysis(&pcm1, &mut low, &mut mid, &mut hi1);
    fb.analysis(&pcm2, &mut low, &mut mid, &mut hi2);

    // hi2 should contain some delayed samples from frame1
    // (the first 39 samples of hi2 come from the delay buffer which
    // contains the tail of the previous frame's QMF output)
    // Just verify they're different from hi1
    let differ = hi1.iter().zip(hi2.iter()).any(|(&a, &b)| a != b);
    assert!(differ, "Consecutive frames should produce different high band output");
}

#[test]
fn test_multiple_frames_stability() {
    let mut analysis = Atrac1AnalysisFilterBank::new();
    let mut synthesis = Atrac1SynthesisFilterBank::new();

    let signal: Vec<f32> = (0..512).map(|i| (i as f32 * 0.03).sin()).collect();
    let mut low = vec![0.0f32; 128];
    let mut mid = vec![0.0f32; 128];
    let mut hi = vec![0.0f32; 256];
    let mut pcm_out = vec![0.0f32; 512];

    // Run many frames to check for numerical stability
    for frame in 0..100 {
        analysis.analysis(&signal, &mut low, &mut mid, &mut hi);
        synthesis.synthesis(&mut pcm_out, &low, &mid, &hi);

        for i in 0..512 {
            assert!(
                pcm_out[i].is_finite(),
                "Non-finite at frame {frame}, sample {i}"
            );
        }
    }
}
