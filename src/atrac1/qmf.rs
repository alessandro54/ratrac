use crate::qmf::Qmf;

const N_IN_SAMPLES: usize = 512;
const DELAY_COMP: usize = 39;

/// ATRAC1 analysis filter bank: splits 512 PCM samples into 3 bands.
/// - low: 128 samples (0 - 5.5 kHz)
/// - mid: 128 samples (5.5 - 11 kHz)
/// - hi: 256 samples (11 - 22 kHz)
pub struct Atrac1AnalysisFilterBank {
    qmf1: Qmf,          // 512 -> 256 + 256
    qmf2: Qmf,          // 256 -> 128 + 128
    mid_low_tmp: Vec<f32>,  // 512
    delay_buf: Vec<f32>,    // DELAY_COMP + 512
}

impl Atrac1AnalysisFilterBank {
    pub fn new() -> Self {
        Self {
            qmf1: Qmf::new(N_IN_SAMPLES),
            qmf2: Qmf::new(N_IN_SAMPLES / 2),
            mid_low_tmp: vec![0.0; 512],
            delay_buf: vec![0.0; DELAY_COMP + 512],
        }
    }

    /// Split 512 PCM samples into 3 frequency bands.
    /// - `pcm`: 512 input samples
    /// - `low`: 128 output samples (low band)
    /// - `mid`: 128 output samples (mid band)
    /// - `hi`: 256 output samples (high band)
    pub fn analysis(&mut self, pcm: &[f32], low: &mut [f32], mid: &mut [f32], hi: &mut [f32]) {
        // Shift delay buffer history
        self.delay_buf.copy_within(256..256 + DELAY_COMP, 0);

        // First QMF: 512 -> mid_low_tmp(256) + delay_buf[39..](256)
        self.qmf1.analysis(pcm, &mut self.mid_low_tmp[..256], &mut self.delay_buf[DELAY_COMP..DELAY_COMP + 256]);

        // Second QMF: 256 -> low(128) + mid(128)
        self.qmf2.analysis(&self.mid_low_tmp[..256], low, mid);

        // Copy delayed high band
        hi.copy_from_slice(&self.delay_buf[..256]);
    }
}

/// ATRAC1 synthesis filter bank: reconstructs 512 PCM samples from 3 bands.
pub struct Atrac1SynthesisFilterBank {
    qmf1: Qmf,
    qmf2: Qmf,
    mid_low_tmp: Vec<f32>,
    delay_buf: Vec<f32>,
}

impl Atrac1SynthesisFilterBank {
    pub fn new() -> Self {
        Self {
            qmf1: Qmf::new(N_IN_SAMPLES),
            qmf2: Qmf::new(N_IN_SAMPLES / 2),
            mid_low_tmp: vec![0.0; 512],
            delay_buf: vec![0.0; DELAY_COMP + 512],
        }
    }

    /// Reconstruct 512 PCM samples from 3 frequency bands.
    /// - `pcm`: 512 output samples
    /// - `low`: 128 input samples (low band)
    /// - `mid`: 128 input samples (mid band)
    /// - `hi`: 256 input samples (high band)
    pub fn synthesis(&mut self, pcm: &mut [f32], low: &[f32], mid: &[f32], hi: &[f32]) {
        // Shift delay buffer history
        self.delay_buf.copy_within(256..256 + DELAY_COMP, 0);

        // Copy high band into delay buffer
        self.delay_buf[DELAY_COMP..DELAY_COMP + 256].copy_from_slice(&hi[..256]);

        // Second QMF synthesis: low(128) + mid(128) -> mid_low_tmp(256)
        self.qmf2.synthesis(&mut self.mid_low_tmp[..256], low, mid);

        // First QMF synthesis: mid_low_tmp(256) + delay_buf(256) -> pcm(512)
        self.qmf1.synthesis(pcm, &self.mid_low_tmp[..256], &self.delay_buf[..256]);
    }
}

#[cfg(test)]
#[path = "../tests/atrac1_qmf_tests.rs"]
mod tests;
