/// 24-tap half-coefficients for the QMF filter bank.
/// Copied verbatim from C++ TQmf::TapHalf[24].
#[allow(clippy::excessive_precision)]
const TAP_HALF: [f32; 24] = [
    -0.00001461907,
    -0.00009205479,
    -0.000_056_157_57,
    0.000_301_172_7,
    0.0002422519,
    -0.000_852_939,
    -0.0005205574,
    0.002_034_017,
    0.000_783_338_9,
    -0.004_215_386,
    -0.000_756_149_9,
    0.007_840_294,
    -0.000_061_169_92,
    -0.01344162,
    0.002_462_682,
    0.021_736_09,
    -0.007801671,
    -0.034_090_22,
    0.01880949,
    0.054_326_01,
    -0.043_596_38,
    -0.099_384_37,
    0.132_079_1,
    0.464_241_6,
];

/// Quadrature Mirror Filter bank.
/// Analysis splits `n_in` samples into two `n_in/2` subbands (lower, upper).
/// Synthesis reconstructs `n_in` samples from two `n_in/2` subbands.
pub struct Qmf {
    n_in: usize,
    /// Even-indexed window coefficients: qmf_window[0], qmf_window[2], ... qmf_window[46]
    win_even: [f32; 24],
    /// Odd-indexed window coefficients: qmf_window[1], qmf_window[3], ... qmf_window[47]
    win_odd: [f32; 24],
    /// Full 48-tap window (kept for synthesis which needs sequential access)
    qmf_window: [f32; 48],
    pcm_buffer: Vec<f32>,       // size: n_in + 46
    pcm_buffer_merge: Vec<f32>, // size: n_in + 46
}

impl Qmf {
    pub fn new(n_in: usize) -> Self {
        let mut qmf_window = [0.0f32; 48];
        for i in 0..24 {
            qmf_window[i] = TAP_HALF[i] * 2.0;
            qmf_window[47 - i] = TAP_HALF[i] * 2.0;
        }

        let mut win_even = [0.0f32; 24];
        let mut win_odd = [0.0f32; 24];
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

    /// Split `input` (n_in samples) into `lower` and `upper` subbands (n_in/2 each).
    #[inline]
    pub fn analysis(&mut self, input: &[f32], lower: &mut [f32], upper: &mut [f32]) {
        let n_in = self.n_in;

        self.pcm_buffer.copy_within(n_in..n_in + 46, 0);
        self.pcm_buffer[46..46 + n_in].copy_from_slice(&input[..n_in]);

        let buf = &self.pcm_buffer[..n_in + 46];
        let we = &self.win_even;
        let wo = &self.win_odd;

        for j in (0..n_in).step_by(2) {
            let base = 47 + j;

            // Take a slice covering all indices we'll access:
            // lo reads buf[base], buf[base-2], ..., buf[base-46] = buf[j+1]
            // up reads buf[base-1], buf[base-3], ..., buf[base-47] = buf[j]
            // So range is buf[j..base+1], length 48
            let window = &buf[j..base + 1];
            // window[47] = buf[base], window[47 - 2*i] = buf[base - 2*i]
            // window[46] = buf[base-1], window[46 - 2*i] = buf[base - 1 - 2*i]

            let mut lo = 0.0f32;
            let mut up = 0.0f32;

            for i in 0..24 {
                lo = we[i].mul_add(window[47 - 2 * i], lo);
                up = wo[i].mul_add(window[46 - 2 * i], up);
            }

            lower[j / 2] = lo + up;
            upper[j / 2] = lo - up;
        }
    }

    /// Reconstruct `output` (n_in samples) from `lower` and `upper` subbands (n_in/2 each).
    #[inline]
    pub fn synthesis(&mut self, output: &mut [f32], lower: &[f32], upper: &[f32]) {
        let n_in = self.n_in;

        let new_part = &mut self.pcm_buffer_merge[46..46 + n_in];
        for i in (0..n_in).step_by(4) {
            let l0 = lower[i / 2];
            let u0 = upper[i / 2];
            let l1 = lower[i / 2 + 1];
            let u1 = upper[i / 2 + 1];
            new_part[i] = l0 + u0;
            new_part[i + 1] = l0 - u0;
            new_part[i + 2] = l1 + u1;
            new_part[i + 3] = l1 - u1;
        }

        let win = &self.qmf_window;

        let mut out_idx = 0;
        for _j in 0..n_in / 2 {
            // Take a 48-element slice for this iteration
            let mbuf = &self.pcm_buffer_merge[out_idx..out_idx + 48];

            let mut s1 = 0.0f32;
            let mut s2 = 0.0f32;

            for i in (0..48).step_by(4) {
                s1 = win[i].mul_add(mbuf[i], s1);
                s2 = win[i + 1].mul_add(mbuf[i + 1], s2);
                s1 = win[i + 2].mul_add(mbuf[i + 2], s1);
                s2 = win[i + 3].mul_add(mbuf[i + 3], s2);
            }

            output[out_idx] = s2;
            output[out_idx + 1] = s1;
            out_idx += 2;
        }

        self.pcm_buffer_merge.copy_within(n_in..n_in + 46, 0);
    }
}

#[cfg(test)]
#[path = "tests/qmf_tests.rs"]
mod tests;
