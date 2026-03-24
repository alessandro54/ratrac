use rustfft::{num_complex::Complex, FftPlanner};

/// Compute twiddle factors (interleaved cos, sin) matching C++ CalcSinCos.
/// Uses f64 for computation, stores as f32.
fn calc_sin_cos(n: usize, scale: f32) -> Vec<f32> {
    let mut tmp = vec![0.0f32; n / 2];
    let alpha = 2.0 * std::f64::consts::PI / (8.0 * n as f64);
    let omega = 2.0 * std::f64::consts::PI / n as f64;
    let scale = (scale as f64 / n as f64).sqrt();
    for i in 0..(n / 4) {
        tmp[2 * i] = (scale * (omega * i as f64 + alpha).cos()) as f32;
        tmp[2 * i + 1] = (scale * (omega * i as f64 + alpha).sin()) as f32;
    }
    tmp
}

/// Forward MDCT: N input samples -> N/2 output coefficients.
pub struct Mdct {
    n: usize,
    sin_cos: Vec<f32>,
    fft_in: Vec<Complex<f32>>,
    fft_out: Vec<Complex<f32>>,
    buf: Vec<f32>,
    planner: std::sync::Arc<dyn rustfft::Fft<f32>>,
}

impl Mdct {
    pub fn new(n: usize, scale: f32) -> Self {
        let n4 = n / 4;
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n4);
        Self {
            n,
            sin_cos: calc_sin_cos(n, scale),
            fft_in: vec![Complex::new(0.0, 0.0); n4],
            fft_out: vec![Complex::new(0.0, 0.0); n4],
            buf: vec![0.0; n / 2],
            planner: fft,
        }
    }

    /// Forward MDCT. Input must be exactly N samples. Returns N/2 coefficients.
    pub fn process(&mut self, input: &[f32]) -> &[f32] {
        let n = self.n;
        let n2 = n / 2;
        let n4 = n / 4;
        let n34 = 3 * n4;

        // Pre-rotation: build complex FFT input
        // First loop: i in 0..n4 step 2 (but we index sin_cos by pairs)
        for i in (0..n4).step_by(2) {
            let r0 = input[n34 - 1 - i] + input[n34 + i];
            let i0 = input[n4 + i] - input[n4 - 1 - i];

            let c = self.sin_cos[i];
            let s = self.sin_cos[i + 1];

            // sin_cos is interleaved [cos, sin, cos, sin, ...]
            // index i corresponds to pair i/2
            self.fft_in[i / 2] = Complex::new(r0 * c + i0 * s, i0 * c - r0 * s);
        }

        // Second loop: i in n4..n2 step 2
        for i in (n4..n2).step_by(2) {
            let r0 = input[n34 - 1 - i] - input[i - n4];
            let i0 = input[n4 + i] + input[5 * n4 - 1 - i];

            let c = self.sin_cos[i];
            let s = self.sin_cos[i + 1];

            self.fft_in[i / 2] = Complex::new(r0 * c + i0 * s, i0 * c - r0 * s);
        }

        // FFT
        self.fft_out.copy_from_slice(&self.fft_in);
        self.planner.process(&mut self.fft_out);

        // Post-rotation
        for i in (0..n2).step_by(2) {
            let r0 = self.fft_out[i / 2].re;
            let i0 = self.fft_out[i / 2].im;

            let c = self.sin_cos[i];
            let s = self.sin_cos[i + 1];

            self.buf[i] = -r0 * c - i0 * s;
            self.buf[n2 - 1 - i] = -r0 * s + i0 * c;
        }

        &self.buf
    }
}

/// Inverse MDCT: N/2 input coefficients -> N output samples.
pub struct Midct {
    n: usize,
    sin_cos: Vec<f32>,
    fft_in: Vec<Complex<f32>>,
    fft_out: Vec<Complex<f32>>,
    buf: Vec<f32>,
    planner: std::sync::Arc<dyn rustfft::Fft<f32>>,
}

impl Midct {
    /// Create inverse MDCT. Default scale = N (matching C++ `TMIDCT<N>` default).
    /// C++ constructor does TMDCTBase(N, scale/2), so we pass scale/2 to calc_sin_cos.
    pub fn new(n: usize, scale: f32) -> Self {
        let n4 = n / 4;
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n4);
        Self {
            n,
            sin_cos: calc_sin_cos(n, scale / 2.0),
            fft_in: vec![Complex::new(0.0, 0.0); n4],
            fft_out: vec![Complex::new(0.0, 0.0); n4],
            buf: vec![0.0; n],
            planner: fft,
        }
    }

    /// Inverse MDCT. Input must be N/2 coefficients. Returns N samples.
    pub fn process(&mut self, input: &[f32]) -> &[f32] {
        let n = self.n;
        let n2 = n / 2;
        let n4 = n / 4;
        let n34 = 3 * n4;
        let n54 = 5 * n4;

        // Pre-rotation with -2.0 factor
        for i in (0..n2).step_by(2) {
            let r0 = input[i];
            let i0 = input[n2 - 1 - i];

            let c = self.sin_cos[i];
            let s = self.sin_cos[i + 1];

            self.fft_in[i / 2] = Complex::new(
                -2.0 * (i0 * s + r0 * c),
                -2.0 * (i0 * c - r0 * s),
            );
        }

        // FFT
        self.fft_out.copy_from_slice(&self.fft_in);
        self.planner.process(&mut self.fft_out);

        // Post-rotation: two loops matching C++
        // First loop: n in 0..n4 step 2
        for i in (0..n4).step_by(2) {
            let r0 = self.fft_out[i / 2].re;
            let i0 = self.fft_out[i / 2].im;

            let c = self.sin_cos[i];
            let s = self.sin_cos[i + 1];

            let r1 = r0 * c + i0 * s;
            let i1 = r0 * s - i0 * c;

            self.buf[n34 - 1 - i] = r1;
            self.buf[n34 + i] = r1;
            self.buf[n4 + i] = i1;
            self.buf[n4 - 1 - i] = -i1;
        }

        // Second loop: n in n4..n2 step 2
        for i in (n4..n2).step_by(2) {
            let r0 = self.fft_out[i / 2].re;
            let i0 = self.fft_out[i / 2].im;

            let c = self.sin_cos[i];
            let s = self.sin_cos[i + 1];

            let r1 = r0 * c + i0 * s;
            let i1 = r0 * s - i0 * c;

            self.buf[n34 - 1 - i] = r1;
            self.buf[i - n4] = -r1;
            self.buf[n4 + i] = i1;
            self.buf[n54 - 1 - i] = i1;
        }

        &self.buf
    }
}

#[cfg(test)]
#[path = "tests/mdct_tests.rs"]
mod tests;
