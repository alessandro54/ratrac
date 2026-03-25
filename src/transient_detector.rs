const PREV_BUF_SZ: usize = 20;
const FIR_LEN: usize = 21;

/// FIR high-pass filter coefficients (10 values, each pre-multiplied by 2.0).
/// Symmetric filter: applied to pairs (i, FIR_LEN-i).
const FIRCOEF: [f32; 10] = [
    -8.65163e-18 * 2.0,
    -0.00851586 * 2.0,
    -6.74764e-18 * 2.0,
    0.0209036 * 2.0,
    -3.36639e-17 * 2.0,
    -0.0438162 * 2.0,
    -1.54175e-17 * 2.0,
    0.0931738 * 2.0,
    -5.52212e-17 * 2.0,
    -0.313819 * 2.0,
];

fn calculate_rms(data: &[f32]) -> f32 {
    let n = data.len() as f32;
    let s: f32 = data.iter().map(|&x| x * x).sum();
    (s / n).sqrt()
}

fn calculate_peak(data: &[f32]) -> f32 {
    data.iter().map(|&x| x.abs()).fold(0.0f32, f32::max)
}

/// Transient detector for adaptive window switching.
/// Detects sudden energy changes in HP-filtered signal.
/// Rising threshold: 16 dB, falling threshold: 20 dB.
pub struct TransientDetector {
    short_sz: usize,
    block_sz: usize,
    n_short_blocks: usize,
    hpf_buffer: Vec<f32>, // block_sz + FIR_LEN
    last_energy: f32,
    last_transient_pos: u16,
}

impl TransientDetector {
    pub fn new(short_sz: usize, block_sz: usize) -> Self {
        Self {
            short_sz,
            block_sz,
            n_short_blocks: block_sz / short_sz,
            hpf_buffer: vec![0.0; block_sz + FIR_LEN],
            last_energy: 0.0,
            last_transient_pos: 0,
        }
    }

    /// Apply 21-tap symmetric HP FIR filter.
    fn hp_filter(&mut self, input: &[f32], output: &mut [f32]) {
        // Copy previous tail into buffer front
        // Then copy new input after it
        self.hpf_buffer[PREV_BUF_SZ..PREV_BUF_SZ + self.block_sz]
            .copy_from_slice(&input[..self.block_sz]);

        let in_buf = &self.hpf_buffer;
        for i in 0..self.block_sz {
            let mut s = in_buf[i + 10]; // center tap (coefficient = 1.0)
            let mut s2 = 0.0f32;
            // (FIR_LEN - 1) / 2 - 1 = 9, so j = 0, 2, 4, 6, 8
            let mut j = 0;
            while j < 9 {
                s += FIRCOEF[j] * (in_buf[i + j] + in_buf[i + FIR_LEN - j]);
                s2 += FIRCOEF[j + 1] * (in_buf[i + j + 1] + in_buf[i + FIR_LEN - j - 1]);
                j += 2;
            }
            output[i] = (s + s2) / 2.0;
        }

        // Save tail for next call
        self.hpf_buffer[..PREV_BUF_SZ]
            .copy_from_slice(&input[self.block_sz - PREV_BUF_SZ..self.block_sz]);
    }

    /// Detect transients in the input buffer.
    /// Returns true if a transient was detected.
    pub fn detect(&mut self, buf: &[f32]) -> bool {
        let n_blocks_to_analyze = self.n_short_blocks + 1;
        let mut filtered = vec![0.0f32; self.block_sz];
        self.hp_filter(buf, &mut filtered);

        let mut rms_per_block = vec![0.0f32; n_blocks_to_analyze];
        rms_per_block[0] = self.last_energy;

        let mut trans = false;
        for i in 1..n_blocks_to_analyze {
            let start = (i - 1) * self.short_sz;
            rms_per_block[i] =
                19.0 * calculate_rms(&filtered[start..start + self.short_sz]).log10();

            if rms_per_block[i] - rms_per_block[i - 1] > 16.0 {
                trans = true;
                self.last_transient_pos = i as u16;
            }
            if rms_per_block[i - 1] - rms_per_block[i] > 20.0 {
                trans = true;
                self.last_transient_pos = i as u16;
            }
        }

        self.last_energy = rms_per_block[self.n_short_blocks];
        trans
    }

    pub fn last_transient_pos(&self) -> u32 {
        self.last_transient_pos as u32
    }
}

/// Analyze gain profile: divide input into segments, return RMS or peak per segment.
pub fn analyze_gain(input: &[f32], len: usize, max_points: usize, use_rms: bool) -> Vec<f32> {
    let step = len / max_points;
    let mut res = Vec::with_capacity(max_points);
    let mut pos = 0;
    while pos < len {
        let segment = &input[pos..pos + step];
        let val = if use_rms {
            calculate_rms(segment)
        } else {
            calculate_peak(segment)
        };
        res.push(val);
        pos += step;
    }
    res
}

#[cfg(test)]
#[path = "tests/transient_detector_tests.rs"]
mod tests;
