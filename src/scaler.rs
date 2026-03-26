use crate::atrac1::{
    BLOCKS_PER_BAND, BlockSizeMod, MAX_BFUS, NUM_QMF, SCALE_TABLE, SPECS_PER_BLOCK,
    SPECS_START_LONG, SPECS_START_SHORT,
};
use crate::util::to_int;

const MAX_SCALE: f32 = 1.0;

/// A scaled block of spectral coefficients with its scale factor.
#[derive(Debug, Clone)]
pub struct ScaledBlock {
    pub scale_factor_index: u8,
    pub values: Vec<f32>,
    pub max_energy: f32,
}

/// Scaler: finds optimal scale factors and normalizes spectral coefficients.
/// Uses a sorted lookup table matching C++ `std::map<float, uint8_t>::lower_bound`.
pub struct Scaler {
    /// Sorted (scale_value, index) pairs for lower_bound lookup.
    scale_index: Vec<(f32, u8)>,
}

impl Default for Scaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Scaler {
    pub fn new() -> Self {
        let table = &*SCALE_TABLE;
        let mut scale_index: Vec<(f32, u8)> = table
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, i as u8))
            .collect();
        // ScaleTable is already monotonically increasing, so this is sorted
        scale_index.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        Self { scale_index }
    }

    /// Find the smallest scale factor >= max_abs_spec (lower_bound).
    fn find_scale(&self, max_abs_spec: f32) -> (f32, u8) {
        let pos = self.scale_index.partition_point(|&(v, _)| v < max_abs_spec);
        if pos < self.scale_index.len() {
            self.scale_index[pos]
        } else {
            // Should not happen if max_abs_spec <= MAX_SCALE, but fallback to last
            *self.scale_index.last().unwrap()
        }
    }

    /// Scale a single block of spectral coefficients.
    /// Finds the optimal scale factor and normalizes values to [-0.99999, 0.99999].
    pub fn scale(&self, input: &[f32]) -> ScaledBlock {
        let mut max_abs_spec = 0.0f32;
        for &v in input {
            max_abs_spec = max_abs_spec.max(v.abs());
        }
        if max_abs_spec > MAX_SCALE {
            max_abs_spec = MAX_SCALE;
        }

        let (scale_factor, scale_factor_index) = self.find_scale(max_abs_spec);

        let mut values = Vec::with_capacity(input.len());
        let mut max_energy = 0.0f32;

        for &v in input {
            let scaled = v / scale_factor;
            max_energy = max_energy.max(v * v);
            let clamped = if scaled.abs() >= 1.0 {
                if scaled > 0.0 { 0.99999 } else { -0.99999 }
            } else {
                scaled
            };
            values.push(clamped);
        }

        ScaledBlock {
            scale_factor_index,
            values,
            max_energy,
        }
    }

    /// Scale an entire ATRAC1 frame (512 spectral coefficients) into BFU blocks.
    pub fn scale_frame(&self, specs: &[f32], block_size: &BlockSizeMod) -> Vec<ScaledBlock> {
        let mut scaled_blocks = Vec::with_capacity(MAX_BFUS);

        for band in 0..NUM_QMF {
            let short_win = block_size.short_win(band);
            let start_bfu = BLOCKS_PER_BAND[band] as usize;
            let end_bfu = BLOCKS_PER_BAND[band + 1] as usize;

            for bfu in start_bfu..end_bfu {
                let spec_start = if short_win {
                    SPECS_START_SHORT[bfu] as usize
                } else {
                    SPECS_START_LONG[bfu] as usize
                };
                let len = SPECS_PER_BLOCK[bfu] as usize;
                scaled_blocks.push(self.scale(&specs[spec_start..spec_start + len]));
            }
        }

        scaled_blocks
    }
}

/// Quantize scaled spectral coefficients to integer mantissas.
///
/// Each input value (in [-1, 1]) is multiplied by `mul` and rounded to the
/// nearest integer using banker's rounding (`round_ties_even`).
///
/// When `ea` (energy adjustment) is enabled, the function tries to minimize
/// the energy difference between original and quantized signals by nudging
/// borderline mantissas (values near ±0.5) up or down. This improves
/// overall energy preservation at the cost of individual coefficient accuracy.
///
/// Returns the energy ratio `original_energy / quantized_energy`.
pub fn quant_mantissas(
    input: &[f32],
    first: usize,
    last: usize,
    mul: f32,
    ea: bool,
    mantissas: &mut [i32],
) -> f32 {
    let mut e1 = 0.0f32;
    let mut e2 = 0.0f32;
    let inv2 = 1.0 / (mul * mul);

    for (j, f) in (first..last).enumerate() {
        let t = input[j] * mul;
        e1 += input[j] * input[j];
        mantissas[f] = to_int(t);
        e2 += mantissas[f] as f32 * mantissas[f] as f32 * inv2;
    }

    if !ea || e1 == 0.0 {
        return if e2 != 0.0 { e1 / e2 } else { 1.0 };
    }

    // Energy adjustment: try to minimize |e2 - e1| by nudging mantissas
    let mut candidates: Vec<(f32, usize)> = Vec::new();
    for (j, f) in (first..last).enumerate() {
        let t = input[j] * mul;
        let delta = t - (t.trunc() + 0.5);
        if delta.abs() < 0.25 {
            candidates.push((delta, f));
        }
    }

    if candidates.is_empty() {
        return if e2 != 0.0 { e1 / e2 } else { 1.0 };
    }

    candidates.sort_by(|a, b| a.0.abs().partial_cmp(&b.0.abs()).unwrap());

    if e2 < e1 {
        for &(_, f) in &candidates {
            let j = f - first;
            let t = input[j] * mul;
            if (mantissas[f].abs() as f32) < t.abs() && (mantissas[f].abs() as f32) < (mul - 1.0) {
                let mut m = mantissas[f];
                if m > 0 {
                    m += 1;
                }
                if m < 0 {
                    m -= 1;
                }
                if m == 0 {
                    m = if t > 0.0 { 1 } else { -1 };
                }

                let ex = e2 - mantissas[f] as f32 * mantissas[f] as f32 * inv2
                    + m as f32 * m as f32 * inv2;

                if (ex - e1).abs() < (e2 - e1).abs() {
                    mantissas[f] = m;
                    e2 = ex;
                }
            }
        }
    } else if e2 > e1 {
        for &(_, f) in &candidates {
            let j = f - first;
            let t = input[j] * mul;
            if (mantissas[f].abs() as f32) > t.abs() {
                let mut m = mantissas[f];
                if m > 0 {
                    m -= 1;
                }
                if m < 0 {
                    m += 1;
                }

                let ex = e2 - mantissas[f] as f32 * mantissas[f] as f32 * inv2
                    + m as f32 * m as f32 * inv2;

                if (ex - e1).abs() < (e2 - e1).abs() {
                    mantissas[f] = m;
                    e2 = ex;
                }
            }
        }
    }

    if e2 != 0.0 { e1 / e2 } else { 1.0 }
}

#[cfg(test)]
#[path = "tests/scaler_tests.rs"]
mod tests;
