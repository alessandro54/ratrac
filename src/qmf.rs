/// 24-tap half-coefficients for the QMF filter bank.
/// Copied verbatim from C++ TQmf::TapHalf[24].
const TAP_HALF: [f32; 24] = [
    -0.00001461907,  -0.00009205479, -0.000056157569,  0.00030117269,
     0.0002422519,   -0.00085293897, -0.0005205574,    0.0020340169,
     0.00078333891,  -0.0042153862,  -0.00075614988,   0.0078402944,
    -0.000061169922, -0.01344162,     0.0024626821,    0.021736089,
    -0.007801671,    -0.034090221,    0.01880949,      0.054326009,
    -0.043596379,    -0.099384367,    0.13207909,      0.46424159,
];

/// Quadrature Mirror Filter bank.
/// Analysis splits `n_in` samples into two `n_in/2` subbands (lower, upper).
/// Synthesis reconstructs `n_in` samples from two `n_in/2` subbands.
pub struct Qmf {
    n_in: usize,
    qmf_window: [f32; 48],
    pcm_buffer: Vec<f32>,       // size: n_in + 46
    pcm_buffer_merge: Vec<f32>, // size: n_in + 46
}

impl Qmf {
    pub fn new(n_in: usize) -> Self {
        // Build symmetric 48-tap window from 24 half-taps
        let mut qmf_window = [0.0f32; 48];
        for i in 0..24 {
            qmf_window[i] = TAP_HALF[i] * 2.0;
            qmf_window[47 - i] = TAP_HALF[i] * 2.0;
        }

        Self {
            n_in,
            qmf_window,
            pcm_buffer: vec![0.0; n_in + 46],
            pcm_buffer_merge: vec![0.0; n_in + 46],
        }
    }

    /// Split `input` (n_in samples) into `lower` and `upper` subbands (n_in/2 each).
    pub fn analysis(&mut self, input: &[f32], lower: &mut [f32], upper: &mut [f32]) {
        let n_in = self.n_in;

        // Shift history: copy last 46 samples to front
        self.pcm_buffer.copy_within(n_in..n_in + 46, 0);

        // Copy new input after history
        self.pcm_buffer[46..46 + n_in].copy_from_slice(&input[..n_in]);

        for j in (0..n_in).step_by(2) {
            let mut lo = 0.0f32;
            let mut up = 0.0f32;
            for i in 0..24 {
                lo += self.qmf_window[2 * i] * self.pcm_buffer[48 - 1 + j - (2 * i)];
                up += self.qmf_window[2 * i + 1] * self.pcm_buffer[48 - 1 + j - (2 * i) - 1];
            }
            lower[j / 2] = lo + up;
            upper[j / 2] = lo - up;
        }
    }

    /// Reconstruct `output` (n_in samples) from `lower` and `upper` subbands (n_in/2 each).
    pub fn synthesis(&mut self, output: &mut [f32], lower: &[f32], upper: &[f32]) {
        let n_in = self.n_in;

        // Interleave lower and upper into new part of merge buffer
        let new_part_start = 46;
        for i in (0..n_in).step_by(4) {
            self.pcm_buffer_merge[new_part_start + i] = lower[i / 2] + upper[i / 2];
            self.pcm_buffer_merge[new_part_start + i + 1] = lower[i / 2] - upper[i / 2];
            self.pcm_buffer_merge[new_part_start + i + 2] = lower[i / 2 + 1] + upper[i / 2 + 1];
            self.pcm_buffer_merge[new_part_start + i + 3] = lower[i / 2 + 1] - upper[i / 2 + 1];
        }

        // Windowed filtering
        let mut out_idx = 0;
        for _j in (0..n_in / 2).rev() {
            let win_start = out_idx; // winP advances by 2 each iteration
            let mut s1 = 0.0f32;
            let mut s2 = 0.0f32;
            for i in (0..48).step_by(2) {
                s1 += self.pcm_buffer_merge[win_start + i] * self.qmf_window[i];
                s2 += self.pcm_buffer_merge[win_start + i + 1] * self.qmf_window[i + 1];
            }
            output[out_idx] = s2;
            output[out_idx + 1] = s1;
            out_idx += 2;
        }

        // Shift history
        self.pcm_buffer_merge.copy_within(n_in..n_in + 46, 0);
    }
}

#[cfg(test)]
#[path = "tests/qmf_tests.rs"]
mod tests;
