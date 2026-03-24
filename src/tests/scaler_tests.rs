use super::*;
use crate::atrac1::{BlockSizeMod, SCALE_TABLE};

// --- Scaler basic tests ---

#[test]
fn test_scaler_new() {
    let scaler = Scaler::new();
    assert_eq!(64, scaler.scale_index.len());
}

#[test]
fn test_scale_single_block_zeros() {
    let scaler = Scaler::new();
    let input = vec![0.0f32; 8];
    let block = scaler.scale(&input);

    // All zeros → scale factor index 0 (smallest), all values 0
    assert_eq!(0, block.scale_factor_index);
    for &v in &block.values {
        assert_eq!(0.0, v);
    }
    assert_eq!(0.0, block.max_energy);
}

#[test]
fn test_scale_single_block_known() {
    let scaler = Scaler::new();
    let table = &*SCALE_TABLE;

    // Use a value that exactly matches a scale factor
    let sf_val = table[40]; // some known scale factor
    let input = vec![sf_val * 0.5; 4]; // half the scale factor
    let block = scaler.scale(&input);

    // Should pick scale factor >= sf_val*0.5
    // The scaled values should be in [-1, 1)
    for &v in &block.values {
        assert!(v.abs() < 1.0, "Scaled value {v} should be < 1.0");
    }
}

#[test]
fn test_scale_clipping() {
    let scaler = Scaler::new();
    let table = &*SCALE_TABLE;

    // Use a value at the exact scale factor boundary
    let sf_val = table[50];
    let input = vec![sf_val; 4]; // exactly at the scale factor
    let block = scaler.scale(&input);

    // Values at boundary should be clamped to ±0.99999
    for &v in &block.values {
        assert!(
            v.abs() <= 0.99999,
            "Scaled value {v} should be clamped to ±0.99999"
        );
    }
}

#[test]
fn test_scale_max_energy() {
    let scaler = Scaler::new();
    let input = vec![0.1, 0.5, 0.3, 0.2];
    let block = scaler.scale(&input);

    // Max energy should be max(v^2) = 0.5^2 = 0.25
    assert!((block.max_energy - 0.25).abs() < 1e-6);
}

#[test]
fn test_scale_factor_index_increases_with_amplitude() {
    let scaler = Scaler::new();

    let block_small = scaler.scale(&[0.001]);
    let block_large = scaler.scale(&[0.5]);

    assert!(
        block_large.scale_factor_index > block_small.scale_factor_index,
        "Larger amplitude should need larger scale factor index"
    );
}

#[test]
fn test_scale_preserves_sign() {
    let scaler = Scaler::new();
    let input = vec![-0.1, 0.2, -0.3, 0.4];
    let block = scaler.scale(&input);

    assert!(block.values[0] < 0.0);
    assert!(block.values[1] > 0.0);
    assert!(block.values[2] < 0.0);
    assert!(block.values[3] > 0.0);
}

#[test]
fn test_scale_above_max_clamps() {
    let scaler = Scaler::new();
    // Value above MAX_SCALE (1.0) should be clamped
    let input = vec![1.5, -1.5];
    let block = scaler.scale(&input);

    for &v in &block.values {
        assert!(v.abs() <= 0.99999, "Should clamp to ±0.99999, got {v}");
    }
}

// --- ScaleFrame ---

#[test]
fn test_scale_frame_long_window() {
    let scaler = Scaler::new();
    let specs = vec![0.01f32; 512];
    let block_size = BlockSizeMod::new(); // all long windows

    let blocks = scaler.scale_frame(&specs, &block_size);
    assert_eq!(52, blocks.len(), "Should produce 52 BFU blocks");

    // Check total values equals 512
    let total: usize = blocks.iter().map(|b| b.values.len()).sum();
    assert_eq!(512, total, "Total scaled values should be 512");
}

#[test]
fn test_scale_frame_short_window() {
    let scaler = Scaler::new();
    let specs = vec![0.01f32; 512];
    let block_size = BlockSizeMod::from_flags(true, true, true); // all short

    let blocks = scaler.scale_frame(&specs, &block_size);
    assert_eq!(52, blocks.len());

    let total: usize = blocks.iter().map(|b| b.values.len()).sum();
    assert_eq!(512, total);
}

#[test]
fn test_scale_frame_mixed_windows() {
    let scaler = Scaler::new();
    let specs = vec![0.01f32; 512];
    let block_size = BlockSizeMod::from_flags(true, false, true);

    let blocks = scaler.scale_frame(&specs, &block_size);
    assert_eq!(52, blocks.len());

    let total: usize = blocks.iter().map(|b| b.values.len()).sum();
    assert_eq!(512, total);
}

// --- QuantMantisas (translated from atrac_scale_ut.cpp) ---

#[test]
fn test_quant_mantisas_energy_preservation() {
    struct TestData {
        input: Vec<f32>,
        scale: f32,
        q: f32,
        max_diff: f32,
    }

    let test_data = vec![
        TestData {
            input: vec![-2.35, -0.84, 0.65, -1.39, 1.25, -0.41, -0.85, 0.89],
            scale: 2.35001,
            q: 2.5,
            max_diff: 0.5,
        },
        TestData {
            input: vec![-1.26, 1.26, -1.26, 1.26, -1.26, 1.26, -1.26, 1.26],
            scale: 2.35001,
            q: 2.5,
            max_diff: 0.4,
        },
        TestData {
            input: vec![-0.32, 0.13, 0.28, 0.35, 0.63, 0.86, 0.63, 0.04],
            scale: 1.0,
            q: 15.5,
            max_diff: 0.03,
        },
    ];

    for (idx, td) in test_data.iter().enumerate() {
        let scaled: Vec<f32> = td.input.iter().map(|&x| x / td.scale).collect();
        let e1: f32 = td.input.iter().map(|&x| x * x).sum();

        let mut mantissas = vec![0i32; td.input.len()];
        quant_mantissas(&scaled, 0, mantissas.len(), td.q, true, &mut mantissas);

        let e2: f32 = mantissas
            .iter()
            .map(|&m| {
                let t = m as f32 * (td.scale / td.q);
                t * t
            })
            .sum();

        assert!(
            (e2 - e1).abs() < td.max_diff,
            "Test {idx}: |e2-e1| = {} >= max_diff {}, e1={e1}, e2={e2}",
            (e2 - e1).abs(),
            td.max_diff
        );
    }
}

#[test]
fn test_quant_mantisas_no_energy_adjustment() {
    let input = vec![0.5, -0.3, 0.7, -0.1];
    let mut mantissas = vec![0i32; 4];

    let ratio = quant_mantissas(&input, 0, 4, 7.0, false, &mut mantissas);

    // With mul=7: 0.5*7=3.5→4, -0.3*7=-2.1→-2, 0.7*7=4.9→5, -0.1*7=-0.7→-1 (banker's)
    assert!(ratio.is_finite());
    assert!(ratio > 0.0);
}

#[test]
fn test_quant_mantisas_zeros() {
    let input = vec![0.0f32; 8];
    let mut mantissas = vec![0i32; 8];

    let ratio = quant_mantissas(&input, 0, 8, 7.0, false, &mut mantissas);

    for &m in &mantissas {
        assert_eq!(0, m);
    }
    assert_eq!(1.0, ratio); // 0/0 case returns 1.0
}
