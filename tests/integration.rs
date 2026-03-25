//! End-to-end integration tests for ratrac.
//! Generates synthetic audio, encodes to AEA, decodes back to WAV,
//! and verifies roundtrip quality.

use std::path::PathBuf;

use ratrac::aea::{AEA_FRAME_SIZE, AeaReader, AeaWriter};
use ratrac::atrac1::decoder::Atrac1Decoder;
use ratrac::atrac1::encoder::Atrac1Encoder;
use ratrac::atrac1::{Atrac1EncodeSettings, NUM_SAMPLES, WindowMode};
use ratrac::wav_output::WavWriter;

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("ratrac_integ_{name}"))
}

/// Generate a mono 440 Hz sine wave at 44100 Hz.
fn generate_sine(num_samples: usize, freq: f32, amplitude: f32) -> Vec<f32> {
    (0..num_samples)
        .map(|i| amplitude * (2.0 * std::f32::consts::PI * freq * i as f32 / 44100.0).sin())
        .collect()
}

/// Generate stereo: left = 440 Hz, right = 880 Hz.
fn generate_stereo_sine(num_frames: usize, amp: f32) -> Vec<f32> {
    let mut samples = vec![0.0f32; num_frames * 2];
    for i in 0..num_frames {
        samples[i * 2] = amp * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin();
        samples[i * 2 + 1] = amp * (2.0 * std::f32::consts::PI * 880.0 * i as f32 / 44100.0).sin();
    }
    samples
}

/// Generate a signal with a transient (silence then sudden burst).
fn generate_transient(num_samples: usize) -> Vec<f32> {
    let mut samples = vec![0.0f32; num_samples];
    // First half: quiet
    for i in 0..num_samples / 2 {
        samples[i] = 0.001 * (i as f32 * 0.1).sin();
    }
    // Second half: loud
    for i in num_samples / 2..num_samples {
        samples[i] = 0.8 * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 44100.0).sin();
    }
    samples
}

/// Generate white noise (deterministic PRNG).
fn generate_noise(num_samples: usize, amplitude: f32) -> Vec<f32> {
    let mut state: u32 = 12345;
    (0..num_samples)
        .map(|_| {
            // Simple xorshift32
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let normalized = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            normalized * amplitude
        })
        .collect()
}

/// Encode PCM to AEA file, return the path.
fn encode_to_aea(
    pcm: &[f32],
    num_channels: usize,
    aea_path: &PathBuf,
    settings: Atrac1EncodeSettings,
) {
    let total_frames_per_ch = (pcm.len() / num_channels).div_ceil(NUM_SAMPLES);
    let mut writer = AeaWriter::create(
        aea_path,
        "test",
        num_channels,
        (total_frames_per_ch * num_channels) as u32,
    )
    .unwrap();

    let mut encoder = Atrac1Encoder::new(settings);

    // First write skipped (dummy)
    writer.write_frame(&[0; AEA_FRAME_SIZE]).unwrap();

    let frame_size = NUM_SAMPLES * num_channels;
    let mut pos = 0;
    while pos < pcm.len() {
        let mut frame = vec![0.0f32; frame_size];
        let copy_len = (pcm.len() - pos).min(frame_size);
        frame[..copy_len].copy_from_slice(&pcm[pos..pos + copy_len]);

        let frames = encoder.encode_frame_interleaved(&frame, num_channels);
        for f in &frames {
            writer.write_frame(f).unwrap();
        }
        pos += frame_size;
    }
    writer.flush().unwrap();
}

/// Decode AEA file to interleaved PCM samples.
fn decode_from_aea(aea_path: &PathBuf) -> (Vec<f32>, usize) {
    let mut reader = AeaReader::open(aea_path).unwrap();
    let num_channels = reader.channel_num();
    let mut decoder = Atrac1Decoder::new();
    let mut all_samples = Vec::new();

    // Skip dummy frame
    reader.read_frame().unwrap();

    loop {
        match decoder.decode_frame_interleaved(&mut reader) {
            Some(samples) => all_samples.extend_from_slice(&samples),
            None => break,
        }
    }
    (all_samples, num_channels)
}

/// Find the delay between two signals by cross-correlation.
fn find_delay(reference: &[f32], test: &[f32], max_lag: usize) -> usize {
    let chunk = 10000.min(reference.len() / 2);
    let start = 2000.min(reference.len() / 4);
    let r = &reference[start..start + chunk];

    let mut best_corr = f64::MIN;
    let mut best_lag = 0;

    for lag in 0..max_lag {
        if start + lag + chunk > test.len() {
            break;
        }
        let t = &test[start + lag..start + lag + chunk];
        let num: f64 = r.iter().zip(t).map(|(&a, &b)| a as f64 * b as f64).sum();
        let e1: f64 = r.iter().map(|&a| (a as f64).powi(2)).sum();
        let e2: f64 = t.iter().map(|&b| (b as f64).powi(2)).sum();
        let denom = (e1 * e2).sqrt();
        let c = if denom > 0.0 { num / denom } else { 0.0 };
        if c > best_corr {
            best_corr = c;
            best_lag = lag;
        }
    }
    best_lag
}

/// Compute SNR in dB between reference and test (aligned).
fn compute_snr(reference: &[f32], test: &[f32], lag: usize) -> f64 {
    let skip = 3000;
    let n = (reference.len() - skip).min(test.len() - lag - skip) - 1000;
    let r = &reference[skip..skip + n];
    let t = &test[skip + lag..skip + lag + n];

    let e_ref: f64 = r.iter().map(|&s| (s as f64).powi(2)).sum();
    let e_diff: f64 = r
        .iter()
        .zip(t)
        .map(|(&a, &b)| ((a - b) as f64).powi(2))
        .sum();

    if e_diff > 0.0 {
        10.0 * (e_ref / e_diff).log10()
    } else {
        f64::INFINITY
    }
}

/// Compute normalized correlation between reference and test (aligned).
fn compute_correlation(reference: &[f32], test: &[f32], lag: usize) -> f64 {
    let skip = 3000;
    let n = (reference.len() - skip).min(test.len() - lag - skip) - 1000;
    let r = &reference[skip..skip + n];
    let t = &test[skip + lag..skip + lag + n];

    let num: f64 = r.iter().zip(t).map(|(&a, &b)| a as f64 * b as f64).sum();
    let e1: f64 = r.iter().map(|&a| (a as f64).powi(2)).sum();
    let e2: f64 = t.iter().map(|&b| (b as f64).powi(2)).sum();
    let denom = (e1 * e2).sqrt();
    if denom > 0.0 { num / denom } else { 0.0 }
}

// ============================================================
// Integration tests
// ============================================================

#[test]
fn test_mono_440hz_sine_roundtrip() {
    let aea_path = temp_path("mono_sine.aea");
    let pcm = generate_sine(44100 * 2, 440.0, 0.5); // 2 seconds

    encode_to_aea(&pcm, 1, &aea_path, Atrac1EncodeSettings::default());
    let (decoded, ch) = decode_from_aea(&aea_path);
    assert_eq!(1, ch);

    let lag = find_delay(&pcm, &decoded, 2048);
    let snr = compute_snr(&pcm, &decoded, lag);
    let corr = compute_correlation(&pcm, &decoded, lag);

    eprintln!("Mono 440Hz: delay={lag}, SNR={snr:.1} dB, correlation={corr:.6}");

    assert!(snr > 20.0, "SNR {snr:.1} dB too low (expected > 20)");
    assert!(
        corr > 0.99,
        "Correlation {corr:.6} too low (expected > 0.99)"
    );

    std::fs::remove_file(&aea_path).ok();
}

#[test]
fn test_stereo_roundtrip() {
    let aea_path = temp_path("stereo_sine.aea");
    let pcm = generate_stereo_sine(44100 * 2, 0.5); // 2 seconds stereo

    encode_to_aea(&pcm, 2, &aea_path, Atrac1EncodeSettings::default());
    let (decoded, ch) = decode_from_aea(&aea_path);
    assert_eq!(2, ch);

    // Check left channel (even samples)
    let orig_l: Vec<f32> = pcm.iter().step_by(2).copied().collect();
    let dec_l: Vec<f32> = decoded.iter().step_by(2).copied().collect();
    let lag = find_delay(&orig_l, &dec_l, 2048);
    let snr = compute_snr(&orig_l, &dec_l, lag);

    eprintln!("Stereo L: delay={lag}, SNR={snr:.1} dB");
    assert!(snr > 20.0, "Left channel SNR {snr:.1} too low");

    // Check right channel (odd samples)
    let orig_r: Vec<f32> = pcm.iter().skip(1).step_by(2).copied().collect();
    let dec_r: Vec<f32> = decoded.iter().skip(1).step_by(2).copied().collect();
    let lag_r = find_delay(&orig_r, &dec_r, 2048);
    let snr_r = compute_snr(&orig_r, &dec_r, lag_r);

    eprintln!("Stereo R: delay={lag_r}, SNR={snr_r:.1} dB");
    assert!(snr_r > 20.0, "Right channel SNR {snr_r:.1} too low");

    std::fs::remove_file(&aea_path).ok();
}

#[test]
fn test_silence_roundtrip() {
    let aea_path = temp_path("silence.aea");
    let pcm = vec![0.0f32; 44100]; // 1 second silence

    encode_to_aea(&pcm, 1, &aea_path, Atrac1EncodeSettings::default());
    let (decoded, ch) = decode_from_aea(&aea_path);
    assert_eq!(1, ch);

    // Decoded silence should be near-zero
    let rms: f32 = (decoded.iter().map(|&s| s * s).sum::<f32>() / decoded.len() as f32).sqrt();
    eprintln!("Silence RMS: {rms:.8}");
    assert!(rms < 0.01, "Silence RMS {rms} too high");

    std::fs::remove_file(&aea_path).ok();
}

#[test]
fn test_transient_roundtrip() {
    let aea_path = temp_path("transient.aea");
    let pcm = generate_transient(44100 * 2); // 2 seconds with transient

    encode_to_aea(&pcm, 1, &aea_path, Atrac1EncodeSettings::default());
    let (decoded, ch) = decode_from_aea(&aea_path);
    assert_eq!(1, ch);

    let lag = find_delay(&pcm, &decoded, 2048);
    let snr = compute_snr(&pcm, &decoded, lag);
    let corr = compute_correlation(&pcm, &decoded, lag);

    eprintln!("Transient: delay={lag}, SNR={snr:.1} dB, correlation={corr:.6}");

    assert!(snr > 10.0, "Transient SNR {snr:.1} too low");
    assert!(corr > 0.95, "Transient correlation {corr:.6} too low");

    std::fs::remove_file(&aea_path).ok();
}

#[test]
fn test_noise_roundtrip() {
    let aea_path = temp_path("noise.aea");
    let pcm = generate_noise(44100 * 2, 0.3); // 2 seconds noise

    encode_to_aea(&pcm, 1, &aea_path, Atrac1EncodeSettings::default());
    let (decoded, ch) = decode_from_aea(&aea_path);
    assert_eq!(1, ch);

    let lag = find_delay(&pcm, &decoded, 2048);
    let snr = compute_snr(&pcm, &decoded, lag);
    let corr = compute_correlation(&pcm, &decoded, lag);

    eprintln!("Noise: delay={lag}, SNR={snr:.1} dB, correlation={corr:.6}");

    // Noise is inherently hard for ATRAC1 (no entropy coding)
    assert!(snr > 3.0, "Noise SNR {snr:.1} too low");
    assert!(corr > 0.7, "Noise correlation {corr:.6} too low");

    std::fs::remove_file(&aea_path).ok();
}

#[test]
fn test_multitone_roundtrip() {
    let aea_path = temp_path("multitone.aea");
    let n = 44100 * 2;
    let pcm: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / 44100.0;
            0.15 * (2.0 * std::f32::consts::PI * 261.63 * t).sin()  // C4
                + 0.15 * (2.0 * std::f32::consts::PI * 329.63 * t).sin()  // E4
                + 0.15 * (2.0 * std::f32::consts::PI * 392.00 * t).sin() // G4
        })
        .collect();

    encode_to_aea(&pcm, 1, &aea_path, Atrac1EncodeSettings::default());
    let (decoded, _) = decode_from_aea(&aea_path);

    let lag = find_delay(&pcm, &decoded, 2048);
    let snr = compute_snr(&pcm, &decoded, lag);

    eprintln!("Multitone (C major): delay={lag}, SNR={snr:.1} dB");
    assert!(snr > 15.0, "Multitone SNR {snr:.1} too low");

    std::fs::remove_file(&aea_path).ok();
}

#[test]
fn test_notransient_mode() {
    let aea_path = temp_path("notransient.aea");
    let pcm = generate_transient(44100 * 2);

    let settings = Atrac1EncodeSettings {
        window_mode: WindowMode::NoTransient,
        window_mask: 0,
        ..Default::default()
    };
    encode_to_aea(&pcm, 1, &aea_path, settings);
    let (decoded, _) = decode_from_aea(&aea_path);

    let lag = find_delay(&pcm, &decoded, 2048);
    let snr = compute_snr(&pcm, &decoded, lag);

    eprintln!("No-transient mode: delay={lag}, SNR={snr:.1} dB");
    // Quality may be worse without transient detection but should still work
    assert!(snr > 5.0, "No-transient SNR {snr:.1} too low");

    std::fs::remove_file(&aea_path).ok();
}

#[test]
fn test_wav_output_valid() {
    let wav_path = temp_path("output_check.wav");
    let aea_path = temp_path("output_check.aea");
    let pcm = generate_sine(44100, 1000.0, 0.5);

    encode_to_aea(&pcm, 1, &aea_path, Atrac1EncodeSettings::default());
    let (decoded, _) = decode_from_aea(&aea_path);

    // Write WAV
    let mut writer = WavWriter::create(&wav_path, 1, 44100).unwrap();
    writer.write_samples(&decoded).unwrap();
    writer.finalize().unwrap();

    // Verify WAV is valid by reading header
    let wav_bytes = std::fs::read(&wav_path).unwrap();
    assert_eq!(&wav_bytes[0..4], b"RIFF");
    assert_eq!(&wav_bytes[8..12], b"WAVE");
    assert_eq!(&wav_bytes[12..16], b"fmt ");
    assert_eq!(&wav_bytes[36..40], b"data");

    // Check sample rate (bytes 24-27 LE)
    let sr = u32::from_le_bytes([wav_bytes[24], wav_bytes[25], wav_bytes[26], wav_bytes[27]]);
    assert_eq!(44100, sr);

    // Check channels (bytes 22-23 LE)
    let ch = u16::from_le_bytes([wav_bytes[22], wav_bytes[23]]);
    assert_eq!(1, ch);

    // Check bits per sample (bytes 34-35 LE)
    let bps = u16::from_le_bytes([wav_bytes[34], wav_bytes[35]]);
    assert_eq!(16, bps);

    // File should have reasonable size: header(44) + samples * 2 bytes
    let expected_data_size = decoded.len() * 2;
    assert_eq!(44 + expected_data_size, wav_bytes.len());

    std::fs::remove_file(&aea_path).ok();
    std::fs::remove_file(&wav_path).ok();
}

#[test]
fn test_fixed_bfu_roundtrip() {
    let aea_path = temp_path("fixed_bfu.aea");
    let pcm = generate_sine(44100 * 2, 440.0, 0.5);

    let settings = Atrac1EncodeSettings {
        bfu_idx_const: 4, // BfuAmountTab[3] = 36
        ..Default::default()
    };
    encode_to_aea(&pcm, 1, &aea_path, settings);
    let (decoded, _) = decode_from_aea(&aea_path);

    let lag = find_delay(&pcm, &decoded, 2048);
    let snr = compute_snr(&pcm, &decoded, lag);

    eprintln!("Fixed BFU (idx=4): delay={lag}, SNR={snr:.1} dB");
    assert!(snr > 15.0, "Fixed BFU SNR {snr:.1} too low");

    std::fs::remove_file(&aea_path).ok();
}

#[test]
fn test_aea_frame_count_matches() {
    let aea_path = temp_path("frame_count.aea");
    // Exactly 10 frames worth = 5120 samples mono
    let pcm = generate_sine(5120, 440.0, 0.5);

    encode_to_aea(&pcm, 1, &aea_path, Atrac1EncodeSettings::default());

    let reader = AeaReader::open(&aea_path).unwrap();
    assert_eq!(1, reader.channel_num());
    assert_eq!(10, reader.num_frames());

    std::fs::remove_file(&aea_path).ok();
}
