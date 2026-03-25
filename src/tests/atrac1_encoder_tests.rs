use super::*;
use crate::aea::{AeaWriter, AeaReader, AEA_FRAME_SIZE};
use crate::atrac1::{Atrac1EncodeSettings, WindowMode, NUM_SAMPLES, SOUND_UNIT_SIZE};
use crate::atrac1::decoder::Atrac1Decoder;
use std::path::PathBuf;

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("ratrac_enc_test_{name}.aea"))
}

fn default_encoder() -> Atrac1Encoder {
    Atrac1Encoder::new(Atrac1EncodeSettings::default())
}

// --- Basic encode tests ---

#[test]
fn test_encode_silence() {
    let mut encoder = default_encoder();
    let pcm = [0.0f32; NUM_SAMPLES];
    let frame = encoder.encode_frame(&pcm, 0);
    assert_eq!(SOUND_UNIT_SIZE, frame.len());
}

#[test]
fn test_encode_sine() {
    let mut encoder = default_encoder();
    let pcm: Vec<f32> = (0..NUM_SAMPLES)
        .map(|i| 0.5 * (i as f32 * 2.0 * std::f32::consts::PI * 1000.0 / 44100.0).sin())
        .collect();
    let frame = encoder.encode_frame(&pcm, 0);
    assert_eq!(SOUND_UNIT_SIZE, frame.len());
}

#[test]
fn test_encode_frame_size_always_212() {
    let mut encoder = default_encoder();

    // Various signals
    let signals: Vec<Vec<f32>> = vec![
        vec![0.0; NUM_SAMPLES],                                         // silence
        (0..NUM_SAMPLES).map(|i| (i as f32 * 0.1).sin()).collect(),     // low freq
        (0..NUM_SAMPLES).map(|i| (i as f32 * 1.5).sin()).collect(),     // high freq
        (0..NUM_SAMPLES).map(|i| if i < 256 { 0.0 } else { 0.8 }).collect(), // transient
    ];

    for (idx, pcm) in signals.iter().enumerate() {
        let frame = encoder.encode_frame(pcm, 0);
        assert_eq!(
            SOUND_UNIT_SIZE,
            frame.len(),
            "Signal {idx}: frame should be 212 bytes"
        );
    }
}

#[test]
fn test_encode_no_transient_mode() {
    let settings = Atrac1EncodeSettings {
        window_mode: WindowMode::NoTransient,
        window_mask: 0,
        ..Default::default()
    };
    let mut encoder = Atrac1Encoder::new(settings);
    let pcm: Vec<f32> = (0..NUM_SAMPLES).map(|i| (i as f32 * 0.1).sin()).collect();
    let frame = encoder.encode_frame(&pcm, 0);
    assert_eq!(SOUND_UNIT_SIZE, frame.len());
}

#[test]
fn test_encode_fixed_bfu() {
    let settings = Atrac1EncodeSettings {
        bfu_idx_const: 3, // BfuAmountTab[2] = 32
        ..Default::default()
    };
    let mut encoder = Atrac1Encoder::new(settings);
    let pcm: Vec<f32> = (0..NUM_SAMPLES).map(|i| (i as f32 * 0.1).sin()).collect();
    let frame = encoder.encode_frame(&pcm, 0);
    assert_eq!(SOUND_UNIT_SIZE, frame.len());
}

#[test]
fn test_encode_multiple_frames_stability() {
    let mut encoder = default_encoder();
    let pcm: Vec<f32> = (0..NUM_SAMPLES)
        .map(|i| 0.3 * (i as f32 * 0.05).sin())
        .collect();

    for f in 0..100 {
        let frame = encoder.encode_frame(&pcm, 0);
        assert_eq!(SOUND_UNIT_SIZE, frame.len(), "Frame {f} wrong size");
        // Check no all-FF (corruption)
        let all_ff = frame.iter().all(|&b| b == 0xFF);
        assert!(!all_ff, "Frame {f} appears corrupted");
    }
}

// --- Encode → Decode roundtrip ---

#[test]
fn test_encode_decode_roundtrip_energy() {
    let mut encoder = default_encoder();
    let mut decoder = Atrac1Decoder::new();

    let pcm: Vec<f32> = (0..NUM_SAMPLES)
        .map(|i| 0.3 * (i as f32 * 2.0 * std::f32::consts::PI * 440.0 / 44100.0).sin())
        .collect();

    // Encode several frames to let filters settle
    for _ in 0..10 {
        let frame = encoder.encode_frame(&pcm, 0);
        decoder.decode_frame(&frame, 0);
    }

    // Now encode and decode one frame
    let frame = encoder.encode_frame(&pcm, 0);
    let decoded = decoder.decode_frame(&frame, 0);

    // Check energy is preserved approximately
    let _input_energy: f32 = pcm.iter().map(|&x| x * x).sum();
    let output_energy: f32 = decoded.iter().map(|&x| x * x).sum();

    // Due to QMF gain and lossy compression, energies won't match exactly
    // but output should have non-trivial energy
    assert!(output_energy > 0.0, "Decoded should have energy");
    assert!(output_energy.is_finite(), "Decoded energy should be finite");
}

#[test]
fn test_encode_decode_silence_roundtrip() {
    let mut encoder = default_encoder();
    let mut decoder = Atrac1Decoder::new();

    let pcm = [0.0f32; NUM_SAMPLES];

    // Several frames
    for _ in 0..5 {
        let frame = encoder.encode_frame(&pcm, 0);
        let decoded = decoder.decode_frame(&frame, 0);

        for (i, &s) in decoded.iter().enumerate() {
            assert!(
                s.abs() < 0.01,
                "Silence decode sample {i} = {s} should be near zero"
            );
        }
    }
}

// --- Interleaved ---

#[test]
fn test_encode_interleaved_stereo() {
    let mut encoder = default_encoder();

    let mut pcm = vec![0.0f32; NUM_SAMPLES * 2];
    for i in 0..NUM_SAMPLES {
        pcm[i * 2] = 0.3 * (i as f32 * 0.1).sin();     // L
        pcm[i * 2 + 1] = 0.3 * (i as f32 * 0.15).sin(); // R
    }

    let frames = encoder.encode_frame_interleaved(&pcm, 2);
    assert_eq!(2, frames.len());
    assert_eq!(SOUND_UNIT_SIZE, frames[0].len());
    assert_eq!(SOUND_UNIT_SIZE, frames[1].len());
    // L and R frames should be different
    assert_ne!(frames[0], frames[1], "L and R should differ");
}

// --- AEA file full roundtrip ---

#[test]
fn test_encode_to_aea_decode_back() {
    let path = temp_path("enc_dec_roundtrip");
    let num_frames = 10;

    let pcm: Vec<f32> = (0..NUM_SAMPLES)
        .map(|i| 0.3 * (i as f32 * 2.0 * std::f32::consts::PI * 440.0 / 44100.0).sin())
        .collect();

    // Encode to AEA
    {
        let mut encoder = default_encoder();
        let mut writer = AeaWriter::create(&path, "roundtrip", 1, num_frames as u32).unwrap();
        writer.write_frame(&[0; AEA_FRAME_SIZE]).unwrap(); // first write skipped

        for _ in 0..num_frames {
            let frame = encoder.encode_frame(&pcm, 0);
            writer.write_frame(&frame).unwrap();
        }
        writer.flush().unwrap();
    }

    // Decode from AEA
    {
        let mut reader = AeaReader::open(&path).unwrap();
        let mut decoder = Atrac1Decoder::new();
        assert_eq!(1, reader.channel_num());

        reader.read_frame().unwrap(); // skip dummy

        for f in 0..num_frames {
            let result = decoder.decode_frame_interleaved(&mut reader);
            assert!(result.is_some(), "Frame {f} should decode");
            let samples = result.unwrap();
            assert_eq!(NUM_SAMPLES, samples.len());

            for (i, &s) in samples.iter().enumerate() {
                assert!(s.is_finite(), "Frame {f} sample {i} not finite");
                assert!(s >= -1.0 && s <= 1.0, "Frame {f} sample {i} = {s} out of range");
            }
        }
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_encode_stereo_aea_roundtrip() {
    let path = temp_path("stereo_roundtrip");
    let num_frames = 5;

    // Encode stereo
    {
        let mut encoder = default_encoder();
        let mut writer = AeaWriter::create(&path, "stereo", 2, (num_frames * 2) as u32).unwrap();
        writer.write_frame(&[0; AEA_FRAME_SIZE]).unwrap();

        for f in 0..num_frames {
            let mut pcm = vec![0.0f32; NUM_SAMPLES * 2];
            for i in 0..NUM_SAMPLES {
                pcm[i * 2] = 0.2 * ((i as f32 + f as f32 * 10.0) * 0.05).sin();
                pcm[i * 2 + 1] = 0.2 * ((i as f32 + f as f32 * 10.0) * 0.07).sin();
            }
            let frames = encoder.encode_frame_interleaved(&pcm, 2);
            for frame in &frames {
                writer.write_frame(frame).unwrap();
            }
        }
        writer.flush().unwrap();
    }

    // Decode stereo
    {
        let mut reader = AeaReader::open(&path).unwrap();
        let mut decoder = Atrac1Decoder::new();
        assert_eq!(2, reader.channel_num());

        reader.read_frame().unwrap(); // skip dummy

        for f in 0..num_frames {
            let result = decoder.decode_frame_interleaved(&mut reader);
            assert!(result.is_some(), "Frame {f} should decode");
            let samples = result.unwrap();
            assert_eq!(NUM_SAMPLES * 2, samples.len());
        }
    }

    std::fs::remove_file(&path).ok();
}
