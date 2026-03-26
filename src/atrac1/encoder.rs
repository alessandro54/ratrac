//! ATRAC1 encoder with two quality presets:
//! - `Fast`: zero-latency, reference-equivalent encoding
//! - `Best`: look-ahead transient detection + NMR-driven bit reallocation

use std::collections::VecDeque;

use crate::atrac1::bitalloc::write_frame;
use crate::atrac1::mdct_impl::Atrac1Mdct;
use crate::atrac1::qmf::Atrac1AnalysisFilterBank;
use crate::atrac1::{Atrac1EncodeSettings, BlockSizeMod, NUM_SAMPLES, Quality, WindowMode};
use crate::psychoacoustic::create_loudness_curve;
use crate::scaler::Scaler;
use crate::transient_detector::TransientDetector;
use crate::util::invert_spectr;

const LOUD_FACTOR: f32 = 0.006;
const LOOKAHEAD_FRAMES: usize = 1;

// ─── Transient Detection ─────────────────────────────────────────────────────

/// QMF-domain transient detectors for one channel (low, mid, high bands).
struct BandTransientDetectors {
    low: TransientDetector,
    mid: TransientDetector,
    hi: TransientDetector,
}

impl BandTransientDetectors {
    fn new() -> Self {
        Self {
            low: TransientDetector::new(16, 128),
            mid: TransientDetector::new(16, 128),
            hi: TransientDetector::new(16, 256),
        }
    }
}

/// Fast time-domain transient detector for look-ahead.
/// Scans raw PCM (before QMF) for sudden energy spikes.
///
/// Tuned to trigger only on violent transients (snare hits, hard consonants).
/// False positives waste bits on short-window overhead. The ratio of 12.0
/// means energy must jump 12x (~11 dB) between consecutive 16-sample blocks.
fn has_transient_in_pcm(pcm: &[f32]) -> bool {
    // Higher ratio = fewer triggers = saves bits for audio data.
    // 8.0 was too sensitive (triggered on minor volume bumps).
    // 12.0 catches real attacks without starving the encoder.
    let attack_ratio = 12.0f32;
    let mut prev_energy = 0.0001f32;

    for chunk in pcm.chunks(16) {
        let energy: f32 = chunk.iter().map(|&s| s * s).sum();
        if energy > prev_energy * attack_ratio {
            return true;
        }
        prev_energy = prev_energy.max(energy) * 0.9 + energy * 0.1;
    }

    false
}

// ─── Channel Extraction ──────────────────────────────────────────────────────

/// Extract one channel from interleaved PCM into a fixed-size buffer.
fn deinterleave(pcm: &[f32], channel: usize, num_channels: usize) -> [f32; NUM_SAMPLES] {
    let mut out = [0.0f32; NUM_SAMPLES];
    for i in 0..NUM_SAMPLES {
        let idx = i * num_channels + channel;
        if idx < pcm.len() {
            out[i] = pcm[idx];
        }
    }
    out
}

// ─── Encoder ─────────────────────────────────────────────────────────────────

/// ATRAC1 encoder with optional look-ahead and analysis-by-synthesis.
pub struct Atrac1Encoder {
    settings: Atrac1EncodeSettings,

    // DSP pipeline (shared across channels — stateless between calls)
    mdct: Atrac1Mdct,
    scaler: Scaler,

    // Per-channel state
    analysis_banks: [Atrac1AnalysisFilterBank; 2],
    transient_detectors: [BandTransientDetectors; 2],
    loudness: [f64; 2],

    // Per-channel overlap buffers for MDCT windowing
    pcm_buf_low: [[f32; 256 + 16]; 2],
    pcm_buf_mid: [[f32; 256 + 16]; 2],
    pcm_buf_hi: [[f32; 512 + 16]; 2],

    // Loudness weighting curve (precomputed)
    loudness_curve: Vec<f32>,

    // Temporal forward masking: previous frame's masking curve per channel.
    // After a loud transient, the masking threshold stays elevated for ~50ms,
    // allowing the encoder to save bits on the quiet frames that follow.
    prev_masking: [[f32; 52]; 2],

    // Look-ahead ring buffer (only active in Quality::Best)
    delay_line: VecDeque<Vec<f32>>,
    future_transients: VecDeque<Vec<bool>>,
}

impl Atrac1Encoder {
    pub fn new(settings: Atrac1EncodeSettings) -> Self {
        let capacity = if settings.quality == Quality::Best {
            LOOKAHEAD_FRAMES + 1
        } else {
            0
        };

        Self {
            settings,
            mdct: Atrac1Mdct::new(),
            scaler: Scaler::new(),
            analysis_banks: [
                Atrac1AnalysisFilterBank::new(),
                Atrac1AnalysisFilterBank::new(),
            ],
            transient_detectors: [BandTransientDetectors::new(), BandTransientDetectors::new()],
            loudness: [LOUD_FACTOR as f64; 2],
            prev_masking: [[0.0; 52]; 2],
            pcm_buf_low: [[0.0; 256 + 16]; 2],
            pcm_buf_mid: [[0.0; 256 + 16]; 2],
            pcm_buf_hi: [[0.0; 512 + 16]; 2],
            loudness_curve: create_loudness_curve(NUM_SAMPLES),
            delay_line: VecDeque::with_capacity(capacity),
            future_transients: VecDeque::with_capacity(capacity),
        }
    }

    // ── Single-Channel Pipeline ──────────────────────────────────────────────

    /// Detect transients in QMF-domain band outputs for one channel.
    /// Returns a 3-bit window mask (bit 0=low, bit 1=mid, bit 2=high).
    fn detect_transients(&mut self, ch: usize) -> u32 {
        match self.settings.window_mode {
            WindowMode::Auto => {
                let td = &mut self.transient_detectors[ch];
                let mut mask = 0u32;
                mask |= td.low.detect(&self.pcm_buf_low[ch][..128]) as u32;
                mask |= (td.mid.detect(&invert_spectr(&self.pcm_buf_mid[ch][..128])) as u32) << 1;
                mask |= (td.hi.detect(&invert_spectr(&self.pcm_buf_hi[ch][..256])) as u32) << 2;
                mask
            }
            WindowMode::NoTransient => self.settings.window_mask,
        }
    }

    /// Encode one channel: QMF → transient detect → MDCT → scale → allocate → write.
    /// Returns (encoded_frame, loudness_value, window_mask).
    fn encode_one_channel(
        &mut self,
        pcm: &[f32],
        ch: usize,
        force_short_windows: bool,
    ) -> (Vec<u8>, f64, u32) {
        // Step 1: QMF analysis — split into 3 frequency bands
        self.analysis_banks[ch].analysis(
            pcm,
            &mut self.pcm_buf_low[ch][..128],
            &mut self.pcm_buf_mid[ch][..128],
            &mut self.pcm_buf_hi[ch][..256],
        );

        // Step 2: Window mode — detect transients or use look-ahead hint
        let window_mask = if force_short_windows {
            7 // all bands short (pre-echo prevention from look-ahead)
        } else {
            self.detect_transients(ch)
        };
        let block_size = BlockSizeMod::from_flags(
            window_mask & 1 != 0,
            window_mask & 2 != 0,
            window_mask & 4 != 0,
        );

        // Step 3: MDCT — transform to frequency domain
        let mut specs = vec![0.0f32; 512];
        self.mdct.mdct(
            &mut specs,
            &mut self.pcm_buf_low[ch],
            &mut self.pcm_buf_mid[ch],
            &mut self.pcm_buf_hi[ch],
            &block_size,
        );

        // Step 4: Loudness tracking (f64 to prevent drift over thousands of frames)
        let frame_loudness: f64 = specs
            .iter()
            .zip(self.loudness_curve.iter())
            .map(|(&s, &c)| (s as f64) * (s as f64) * (c as f64))
            .sum();
        if window_mask == 0 {
            self.loudness[ch] = 0.98 * self.loudness[ch] + 0.02 * frame_loudness;
        }

        // Step 5: Scale factors + bit allocation + quantization + bitstream
        let mut scaled = self.scaler.scale_frame(&specs, &block_size);

        // Step 5b: Temporal forward masking (Quality::Best only)
        // After a loud transient, the ear is "deaf" for ~50ms. We artificially
        // raise max_energy for quiet BFUs that follow a loud frame, telling the
        // bit allocator they don't need as many bits.
        if self.settings.quality == Quality::Best {
            let decay = 0.7f32; // ~50ms decay at 11.6ms/frame ≈ 4-5 frames
            for i in 0..scaled.len().min(52) {
                // Combine current energy with decayed previous masking
                let temporal_mask = self.prev_masking[ch][i] * decay;
                if temporal_mask > scaled[i].max_energy && scaled[i].max_energy > 0.0 {
                    // This BFU is masked by the previous frame's energy
                    // Inflate its apparent energy so it gets fewer bits
                    scaled[i].max_energy = temporal_mask;
                }
                // Update temporal masking state for next frame
                self.prev_masking[ch][i] =
                    self.prev_masking[ch][i].max(scaled[i].max_energy) * decay;
            }
        }

        let loudness_param = (self.loudness[ch] / LOUD_FACTOR as f64) as f32;
        let (frame, _) = write_frame(
            &scaled,
            &block_size,
            loudness_param,
            self.settings.bfu_idx_const,
            self.settings.fast_bfu_num_search,
        );

        (frame, frame_loudness, window_mask)
    }

    // ── Public API ───────────────────────────────────────────────────────────

    /// Encode 512 mono PCM samples. Returns a 212-byte AEA frame.
    pub fn encode_frame(&mut self, pcm: &[f32], channel: usize) -> Vec<u8> {
        self.encode_one_channel(pcm, channel, false).0
    }

    /// Encode interleaved PCM (512 × num_channels samples).
    /// Returns one AEA frame per channel.
    ///
    /// In `Quality::Best` mode, returns an empty `Vec` during the first frame
    /// (priming the look-ahead buffer). Call [`flush`] at EOF to drain.
    pub fn encode_frame_interleaved(&mut self, pcm: &[f32], num_channels: usize) -> Vec<Vec<u8>> {
        let use_lookahead =
            self.settings.quality == Quality::Best && self.settings.window_mode == WindowMode::Auto;

        if use_lookahead {
            self.encode_lookahead(pcm, num_channels)
        } else {
            self.encode_immediate(pcm, num_channels)
        }
    }

    /// Flush the look-ahead buffer. Call once after all input is consumed.
    /// Returns the final delayed frame(s), or empty if no look-ahead was used.
    pub fn flush(&mut self, num_channels: usize) -> Vec<Vec<u8>> {
        let Some(pcm) = self.delay_line.pop_front() else {
            return Vec::new();
        };
        self.future_transients.pop_front();

        // Encode the final frame with no look-ahead information
        (0..num_channels)
            .map(|ch| {
                let channel_pcm = deinterleave(&pcm, ch, num_channels);
                self.encode_one_channel(&channel_pcm, ch, false).0
            })
            .collect()
    }

    // ── Encoding Strategies ──────────────────────────────────────────────────

    /// Immediate encoding: no look-ahead, no delay.
    fn encode_immediate(&mut self, pcm: &[f32], num_channels: usize) -> Vec<Vec<u8>> {
        (0..num_channels)
            .map(|ch| {
                let channel_pcm = deinterleave(pcm, ch, num_channels);
                self.encode_one_channel(&channel_pcm, ch, false).0
            })
            .collect()
    }

    /// Look-ahead encoding: buffer one frame, analyze the next for transients,
    /// encode the current frame with foreknowledge of upcoming attacks.
    fn encode_lookahead(&mut self, pcm: &[f32], num_channels: usize) -> Vec<Vec<u8>> {
        // Scan the incoming (future) frame for transients per channel
        let transients: Vec<bool> = (0..num_channels)
            .map(|ch| has_transient_in_pcm(&deinterleave(pcm, ch, num_channels)))
            .collect();

        self.delay_line.push_back(pcm.to_vec());
        self.future_transients.push_back(transients);

        // Still filling the buffer?
        if self.delay_line.len() <= LOOKAHEAD_FRAMES {
            return Vec::new();
        }

        // Pop the oldest frame and encode it
        let current_pcm = self.delay_line.pop_front().unwrap();
        self.future_transients.pop_front();

        // Check what's coming next (clone to release borrow on self)
        let next_transients = self.future_transients.front().cloned();

        (0..num_channels)
            .map(|ch| {
                let channel_pcm = deinterleave(&current_pcm, ch, num_channels);
                let force_short = next_transients
                    .as_ref()
                    .and_then(|t| t.get(ch).copied())
                    .unwrap_or(false);
                self.encode_one_channel(&channel_pcm, ch, force_short).0
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "../tests/atrac1_encoder_tests.rs"]
mod tests;
