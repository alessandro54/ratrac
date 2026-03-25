//! Dual-mono test: encode fake stereo using the mono encode_frame path
//! vs the stereo encode_frame_interleaved path, compare quality.

use ratrac::aea::{AEA_FRAME_SIZE, AeaReader, AeaWriter};
use ratrac::atrac1::decoder::Atrac1Decoder;
use ratrac::atrac1::encoder::Atrac1Encoder;
use ratrac::atrac1::{Atrac1EncodeSettings, NUM_SAMPLES};

fn generate_signal(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32 / 44100.0;
            0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 1000.0 * t).sin()
                + 0.1 * (2.0 * std::f32::consts::PI * 3000.0 * t).sin()
        })
        .collect()
}

fn encode_decode_mono(signal: &[f32]) -> Vec<f32> {
    let path = std::env::temp_dir().join("dualmono_test_mono.aea");
    let num_frames = signal.len().div_ceil(NUM_SAMPLES);

    let mut encoder = Atrac1Encoder::new(Atrac1EncodeSettings::default());
    let mut writer = AeaWriter::create(&path, "test", 1, num_frames as u32).unwrap();
    writer.write_frame(&[0; AEA_FRAME_SIZE]).unwrap();

    let mut pos = 0;
    while pos < signal.len() {
        let mut frame = vec![0.0f32; NUM_SAMPLES];
        let len = (signal.len() - pos).min(NUM_SAMPLES);
        frame[..len].copy_from_slice(&signal[pos..pos + len]);
        let encoded = encoder.encode_frame(&frame, 0);
        writer.write_frame(&encoded).unwrap();
        pos += NUM_SAMPLES;
    }
    writer.flush().unwrap();

    // Decode
    let mut reader = AeaReader::open(&path).unwrap();
    let mut decoder = Atrac1Decoder::new();
    reader.read_frame().unwrap();
    let mut decoded = Vec::new();
    loop {
        match decoder.decode_frame_interleaved(&mut reader) {
            Some(s) => decoded.extend_from_slice(&s),
            None => break,
        }
    }
    std::fs::remove_file(&path).ok();
    decoded
}

fn encode_decode_stereo_interleaved(signal: &[f32]) -> Vec<f32> {
    let path = std::env::temp_dir().join("dualmono_test_stereo_interleaved.aea");
    let num_frames = signal.len().div_ceil(NUM_SAMPLES);

    let mut encoder = Atrac1Encoder::new(Atrac1EncodeSettings::default());
    let mut writer = AeaWriter::create(&path, "test", 2, (num_frames * 2) as u32).unwrap();
    writer.write_frame(&[0; AEA_FRAME_SIZE]).unwrap();

    // Create interleaved stereo (identical L/R)
    let mut pos = 0;
    while pos < signal.len() {
        let mut frame = vec![0.0f32; NUM_SAMPLES * 2];
        let len = (signal.len() - pos).min(NUM_SAMPLES);
        for i in 0..len {
            frame[i * 2] = signal[pos + i];
            frame[i * 2 + 1] = signal[pos + i];
        }
        let frames = encoder.encode_frame_interleaved(&frame, 2);
        for f in &frames {
            writer.write_frame(f).unwrap();
        }
        pos += NUM_SAMPLES;
    }
    writer.flush().unwrap();

    let mut reader = AeaReader::open(&path).unwrap();
    let mut decoder = Atrac1Decoder::new();
    reader.read_frame().unwrap();
    let mut decoded = Vec::new();
    loop {
        match decoder.decode_frame_interleaved(&mut reader) {
            Some(s) => decoded.extend_from_slice(&s),
            None => break,
        }
    }
    std::fs::remove_file(&path).ok();
    // Return left channel only
    decoded.iter().step_by(2).copied().collect()
}

fn encode_decode_stereo_via_mono(signal: &[f32]) -> Vec<f32> {
    // KEY TEST: encode stereo file using encode_frame (mono path) per channel
    let path = std::env::temp_dir().join("dualmono_test_stereo_mono_path.aea");
    let num_frames = signal.len().div_ceil(NUM_SAMPLES);

    let mut encoder = Atrac1Encoder::new(Atrac1EncodeSettings::default());
    let mut writer = AeaWriter::create(&path, "test", 2, (num_frames * 2) as u32).unwrap();
    writer.write_frame(&[0; AEA_FRAME_SIZE]).unwrap();

    let mut pos = 0;
    while pos < signal.len() {
        let mut channel_pcm = vec![0.0f32; NUM_SAMPLES];
        let len = (signal.len() - pos).min(NUM_SAMPLES);
        channel_pcm[..len].copy_from_slice(&signal[pos..pos + len]);

        // Call encode_frame independently for each channel (mono path)
        let frame_l = encoder.encode_frame(&channel_pcm, 0);
        let frame_r = encoder.encode_frame(&channel_pcm, 1);
        writer.write_frame(&frame_l).unwrap();
        writer.write_frame(&frame_r).unwrap();
        pos += NUM_SAMPLES;
    }
    writer.flush().unwrap();

    let mut reader = AeaReader::open(&path).unwrap();
    let mut decoder = Atrac1Decoder::new();
    reader.read_frame().unwrap();
    let mut decoded = Vec::new();
    loop {
        match decoder.decode_frame_interleaved(&mut reader) {
            Some(s) => decoded.extend_from_slice(&s),
            None => break,
        }
    }
    std::fs::remove_file(&path).ok();
    decoded.iter().step_by(2).copied().collect()
}

fn snr(original: &[f32], decoded: &[f32], skip: usize) -> f64 {
    let o = &original[skip..];
    // Find delay
    let chunk = 10000.min(o.len() / 2);
    let start = 2000.min(o.len() / 4);
    let mut best_lag = 0;
    let mut best_corr = f64::MIN;
    for lag in 0..2048 {
        if start + lag + chunk > decoded.len() {
            break;
        }
        let r = &o[start..start + chunk];
        let t = &decoded[start + lag..start + lag + chunk];
        let num: f64 = r.iter().zip(t).map(|(&a, &b)| a as f64 * b as f64).sum();
        let e1: f64 = r.iter().map(|&a| (a as f64).powi(2)).sum();
        let e2: f64 = t.iter().map(|&b| (b as f64).powi(2)).sum();
        let d = (e1 * e2).sqrt();
        let c = if d > 0.0 { num / d } else { 0.0 };
        if c > best_corr {
            best_corr = c;
            best_lag = lag;
        }
    }

    let n = o.len().min(decoded.len() - best_lag) - 1000;
    let e_ref: f64 = o[..n].iter().map(|&s| (s as f64).powi(2)).sum();
    let e_diff: f64 = o[..n]
        .iter()
        .zip(&decoded[best_lag..best_lag + n])
        .map(|(&a, &b)| ((a - b) as f64).powi(2))
        .sum();
    if e_diff > 0.0 {
        10.0 * (e_ref / e_diff).log10()
    } else {
        f64::INFINITY
    }
}

#[test]
fn test_dual_mono_isolation() {
    let signal = generate_signal(44100 * 3);

    let mono_decoded = encode_decode_mono(&signal);
    let stereo_interleaved_decoded = encode_decode_stereo_interleaved(&signal);
    let stereo_mono_path_decoded = encode_decode_stereo_via_mono(&signal);

    let snr_mono = snr(&signal, &mono_decoded, 3000);
    let snr_stereo_interleaved = snr(&signal, &stereo_interleaved_decoded, 3000);
    let snr_stereo_mono_path = snr(&signal, &stereo_mono_path_decoded, 3000);

    eprintln!("=== Dual-Mono Isolation Test ===");
    eprintln!("Pure mono (encode_frame):                    {snr_mono:.2} dB");
    eprintln!("Stereo via encode_frame_interleaved (L ch):  {snr_stereo_interleaved:.2} dB");
    eprintln!("Stereo via encode_frame per channel (L ch):  {snr_stereo_mono_path:.2} dB");
    eprintln!();
    eprintln!(
        "Interleaved vs Mono:    {:+.2} dB",
        snr_stereo_interleaved - snr_mono
    );
    eprintln!(
        "Mono-path vs Mono:      {:+.2} dB",
        snr_stereo_mono_path - snr_mono
    );
}
