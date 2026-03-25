/// 24-tap half-coefficients for the QMF filter bank.
/// Copied verbatim from C++ TQmf::TapHalf[24].
#[allow(clippy::excessive_precision)]
const TAP_HALF: [f64; 24] = [
    -0.00001461907,
    -0.00009205479,
    -0.000056157569,
    0.00030117269,
    0.0002422519,
    -0.00085293897,
    -0.0005205574,
    0.0020340169,
    0.00078333891,
    -0.0042153862,
    -0.00075614988,
    0.0078402944,
    -0.000061169922,
    -0.01344162,
    0.0024626821,
    0.021736089,
    -0.007801671,
    -0.034090221,
    0.01880949,
    0.054326009,
    -0.043596379,
    -0.099384367,
    0.13207909,
    0.46424159,
];

/// Quadrature Mirror Filter bank with f64 internal precision.
/// All internal state and computation uses f64 to minimize rounding error.
/// Input/output interface remains f32 for compatibility.
pub struct Qmf {
    n_in: usize,
    win_even: [f64; 24],
    win_odd: [f64; 24],
    qmf_window: [f64; 48],
    pcm_buffer: Vec<f64>,
    pcm_buffer_merge: Vec<f64>,
}

impl Qmf {
    pub fn new(n_in: usize) -> Self {
        let mut qmf_window = [0.0f64; 48];
        for i in 0..24 {
            qmf_window[i] = TAP_HALF[i] * 2.0;
            qmf_window[47 - i] = TAP_HALF[i] * 2.0;
        }

        let mut win_even = [0.0f64; 24];
        let mut win_odd = [0.0f64; 24];
        for i in 0..24 {
            win_even[i] = qmf_window[2 * i];
            win_odd[i] = qmf_window[2 * i + 1];
        }

        Self {
            n_in,
            win_even,
            win_odd,
            qmf_window,
            pcm_buffer: vec![0.0; n_in + 46],
            pcm_buffer_merge: vec![0.0; n_in + 46],
        }
    }

    #[inline]
    pub fn analysis(&mut self, input: &[f32], lower: &mut [f32], upper: &mut [f32]) {
        let n_in = self.n_in;

        self.pcm_buffer.copy_within(n_in..n_in + 46, 0);
        for i in 0..n_in {
            self.pcm_buffer[46 + i] = input[i] as f64;
        }

        let buf = &self.pcm_buffer[..n_in + 46];
        let we = &self.win_even;
        let wo = &self.win_odd;

        for j in (0..n_in).step_by(2) {
            let base = 47 + j;
            let window = &buf[j..base + 1];

            let mut lo = 0.0f64;
            let mut up = 0.0f64;

            for i in 0..24 {
                lo = we[i].mul_add(window[47 - 2 * i], lo);
                up = wo[i].mul_add(window[46 - 2 * i], up);
            }

            lower[j / 2] = (lo + up) as f32;
            upper[j / 2] = (lo - up) as f32;
        }
    }

    #[inline]
    pub fn synthesis(&mut self, output: &mut [f32], lower: &[f32], upper: &[f32]) {
        let n_in = self.n_in;

        let new_part = &mut self.pcm_buffer_merge[46..46 + n_in];
        for i in (0..n_in).step_by(4) {
            let l0 = lower[i / 2] as f64;
            let u0 = upper[i / 2] as f64;
            let l1 = lower[i / 2 + 1] as f64;
            let u1 = upper[i / 2 + 1] as f64;
            new_part[i] = l0 + u0;
            new_part[i + 1] = l0 - u0;
            new_part[i + 2] = l1 + u1;
            new_part[i + 3] = l1 - u1;
        }

        let win = &self.qmf_window;

        let mut out_idx = 0;
        for _j in 0..n_in / 2 {
            let mbuf = &self.pcm_buffer_merge[out_idx..out_idx + 48];

            let mut s1 = 0.0f64;
            let mut s2 = 0.0f64;

            for i in (0..48).step_by(4) {
                s1 = win[i].mul_add(mbuf[i], s1);
                s2 = win[i + 1].mul_add(mbuf[i + 1], s2);
                s1 = win[i + 2].mul_add(mbuf[i + 2], s1);
                s2 = win[i + 3].mul_add(mbuf[i + 3], s2);
            }

            output[out_idx] = s2 as f32;
            output[out_idx + 1] = s1 as f32;
            out_idx += 2;
        }

        self.pcm_buffer_merge.copy_within(n_in..n_in + 46, 0);
    }
}

#[cfg(test)]
#[path = "tests/qmf_tests.rs"]
mod tests;
