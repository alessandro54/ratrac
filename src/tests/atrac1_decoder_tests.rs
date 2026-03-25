use super::*;
use crate::aea::{AeaWriter, AeaReader, AEA_FRAME_SIZE};
use crate::atrac1::{BlockSizeMod, NUM_SAMPLES};
use crate::atrac1::bitalloc::{write_bitstream, write_frame};
use crate::scaler::{Scaler, ScaledBlock};
use crate::atrac1::SPECS_PER_BLOCK;
use std::path::PathBuf;

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("ratrac_dec_test_{name}.aea"))
}

// --- Basic decode tests ---

#[test]
fn test_decode_silence_frame() {
    let mut decoder = Atrac1Decoder::new();

    // Build a silent frame: all zero word lengths
    let block_size = BlockSizeMod::new();
    let scaled_blocks: Vec<ScaledBlock> = (0..52)
        .map(|i| ScaledBlock {
            scale_factor_index: 0,
            values: vec![0.0; SPECS_PER_BLOCK[i] as usize],
            max_energy: 0.0,
        })
        .collect();
    let bits_per_block = vec![0u32; 52];
    let frame = write_bitstream(&bits_per_block, &scaled_blocks, 7, &block_size);

    let samples = decoder.decode_frame(&frame, 0);

    // All samples should be zero or very near zero
    for (i, &s) in samples.iter().enumerate() {
        assert!(
            s.abs() < 1e-6,
            "Sample {i} should be ~0 for silence, got {s}"
        );
    }
}

#[test]
fn test_decode_frame_produces_512_samples() {
    let mut decoder = Atrac1Decoder::new();

    let block_size = BlockSizeMod::new();
    let scaled_blocks: Vec<ScaledBlock> = (0..20)
        .map(|i| ScaledBlock {
            scale_factor_index: 30,
            values: vec![0.1; SPECS_PER_BLOCK[i] as usize],
            max_energy: 0.01,
        })
        .collect();
    let bits_per_block = vec![4u32; 20];
    let frame = write_bitstream(&bits_per_block, &scaled_blocks, 0, &block_size);

    let samples = decoder.decode_frame(&frame, 0);
    assert_eq!(NUM_SAMPLES, samples.len());
}

#[test]
fn test_decode_frame_clipping() {
    let mut decoder = Atrac1Decoder::new();
    let scaler = Scaler::new();
    let block_size = BlockSizeMod::new();

    // Create a loud spectrum to test clipping
    let mut specs = vec![0.0f32; 512];
    for i in 0..512 {
        specs[i] = 0.9 * (i as f32 * 0.05).sin();
    }
    let scaled = scaler.scale_frame(&specs, &block_size);
    let (frame, _) = write_frame(&scaled, &block_size, 0.5, 0, false);

    let samples = decoder.decode_frame(&frame, 0);

    for (i, &s) in samples.iter().enumerate() {
        assert!(
            s >= -1.0 && s <= 1.0,
            "Sample {i} = {s} should be clipped to [-1, 1]"
        );
    }
}

#[test]
fn test_decode_frame_all_finite() {
    let mut decoder = Atrac1Decoder::new();
    let scaler = Scaler::new();
    let block_size = BlockSizeMod::new();

    let mut specs = vec![0.0f32; 512];
    for i in 0..512 {
        specs[i] = 0.3 * (i as f32 * 0.02).sin();
    }
    let scaled = scaler.scale_frame(&specs, &block_size);
    let (frame, _) = write_frame(&scaled, &block_size, 0.5, 0, false);

    let samples = decoder.decode_frame(&frame, 0);

    for (i, &s) in samples.iter().enumerate() {
        assert!(s.is_finite(), "Sample {i} is not finite: {s}");
    }
}

#[test]
fn test_decode_multiple_frames_stability() {
    let mut decoder = Atrac1Decoder::new();
    let scaler = Scaler::new();
    let block_size = BlockSizeMod::new();

    let mut specs = vec![0.0f32; 512];
    for i in 0..512 {
        specs[i] = 0.2 * (i as f32 * 0.03).sin();
    }
    let scaled = scaler.scale_frame(&specs, &block_size);
    let (frame, _) = write_frame(&scaled, &block_size, 0.5, 0, false);

    // Decode 50 frames to check stability
    for f in 0..50 {
        let samples = decoder.decode_frame(&frame, 0);
        for (i, &s) in samples.iter().enumerate() {
            assert!(
                s.is_finite(),
                "Frame {f}, sample {i} is not finite: {s}"
            );
        }
    }
}

// --- Encode → Decode roundtrip ---

#[test]
fn test_encode_decode_roundtrip_spectrum() {
    let mut decoder = Atrac1Decoder::new();
    let scaler = Scaler::new();
    let block_size = BlockSizeMod::new();

    // Create a known spectrum
    let mut specs_in = vec![0.0f32; 512];
    for i in 0..512 {
        specs_in[i] = 0.2 * (i as f32 * 0.03).sin();
    }

    let scaled_blocks = scaler.scale_frame(&specs_in, &block_size);
    let (frame, _) = write_frame(&scaled_blocks, &block_size, 0.5, 0, false);

    let samples = decoder.decode_frame(&frame, 0);

    // Verify we get non-trivial output
    let has_nonzero = samples.iter().any(|&s| s.abs() > 1e-6);
    assert!(has_nonzero, "Decoded samples should be non-zero for non-zero spectrum");

    // All finite
    for (i, &s) in samples.iter().enumerate() {
        assert!(s.is_finite(), "Sample {i} not finite");
    }
}

// --- AEA file roundtrip ---

#[test]
fn test_decode_from_aea_file() {
    let path = temp_path("decode_aea");
    let block_size = BlockSizeMod::new();
    let scaler = Scaler::new();

    // Encode 5 frames to AEA
    let num_frames = 5;
    {
        let mut writer = AeaWriter::create(&path, "test", 1, num_frames).unwrap();
        writer.write_frame(&[0; AEA_FRAME_SIZE]).unwrap(); // first write skipped

        for f in 0..num_frames {
            let mut specs = vec![0.0f32; 512];
            for i in 0..512 {
                specs[i] = 0.1 * ((i as f32 + f as f32 * 10.0) * 0.02).sin();
            }
            let scaled = scaler.scale_frame(&specs, &block_size);
            let (frame, _) = write_frame(&scaled, &block_size, 0.5, 0, false);
            writer.write_frame(&frame).unwrap();
        }
        writer.flush().unwrap();
    }

    // Decode
    {
        let mut reader = AeaReader::open(&path).unwrap();
        let mut decoder = Atrac1Decoder::new();

        // Skip dummy frame
        reader.read_frame().unwrap();

        for f in 0..num_frames {
            let result = decoder.decode_frame_interleaved(&mut reader);
            assert!(result.is_some(), "Frame {f} should decode successfully");
            let samples = result.unwrap();
            assert_eq!(NUM_SAMPLES, samples.len()); // mono

            for (i, &s) in samples.iter().enumerate() {
                assert!(s.is_finite(), "Frame {f}, sample {i} not finite");
            }
        }
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_decode_stereo_aea() {
    let path = temp_path("decode_stereo");
    let block_size = BlockSizeMod::new();
    let scaler = Scaler::new();

    let num_frames = 3;
    {
        let mut writer = AeaWriter::create(&path, "stereo", 2, num_frames * 2).unwrap();
        writer.write_frame(&[0; AEA_FRAME_SIZE]).unwrap(); // skipped

        for _f in 0..num_frames {
            // Write L and R frames
            for ch in 0..2 {
                let mut specs = vec![0.0f32; 512];
                for i in 0..512 {
                    specs[i] = 0.1 * ((i as f32 + ch as f32 * 100.0) * 0.02).sin();
                }
                let scaled = scaler.scale_frame(&specs, &block_size);
                let (frame, _) = write_frame(&scaled, &block_size, 0.5, 0, false);
                writer.write_frame(&frame).unwrap();
            }
        }
        writer.flush().unwrap();
    }

    {
        let mut reader = AeaReader::open(&path).unwrap();
        let mut decoder = Atrac1Decoder::new();
        assert_eq!(2, reader.channel_num());

        reader.read_frame().unwrap(); // skip dummy

        for f in 0..num_frames {
            let result = decoder.decode_frame_interleaved(&mut reader);
            assert!(result.is_some(), "Frame {f} should decode");
            let samples = result.unwrap();
            assert_eq!(NUM_SAMPLES * 2, samples.len()); // stereo interleaved

            for (i, &s) in samples.iter().enumerate() {
                assert!(s.is_finite(), "Frame {f}, sample {i} not finite");
            }
        }
    }

    std::fs::remove_file(&path).ok();
}
