use crate::atrac1::bitalloc::write_frame;
use crate::atrac1::mdct_impl::Atrac1Mdct;
use crate::atrac1::qmf::Atrac1AnalysisFilterBank;
use crate::atrac1::{Atrac1EncodeSettings, BlockSizeMod, NUM_SAMPLES, WindowMode};
use crate::psychoacoustic::create_loudness_curve;
use crate::scaler::Scaler;
use crate::transient_detector::TransientDetector;
use crate::util::invert_spectr;

const LOUD_FACTOR: f32 = 0.006;

struct ChannelTransientDetectors {
    low: TransientDetector,
    mid: TransientDetector,
    hi: TransientDetector,
}

impl ChannelTransientDetectors {
    fn new() -> Self {
        Self {
            low: TransientDetector::new(16, 128),
            mid: TransientDetector::new(16, 128),
            hi: TransientDetector::new(16, 256),
        }
    }
}

/// Full ATRAC1 encoder: PCM samples → AEA frames.
pub struct Atrac1Encoder {
    settings: Atrac1EncodeSettings,
    mdct: Atrac1Mdct,
    analysis_filter_bank: [Atrac1AnalysisFilterBank; 2],
    transient_detectors: [ChannelTransientDetectors; 2],
    scaler: Scaler,
    loudness_curve: Vec<f32>,
    loudness: f64, // f64 to prevent FP error accumulation over thousands of frames

    pcm_buf_low: [[f32; 256 + 16]; 2],
    pcm_buf_mid: [[f32; 256 + 16]; 2],
    pcm_buf_hi: [[f32; 512 + 16]; 2],
}

impl Atrac1Encoder {
    pub fn new(settings: Atrac1EncodeSettings) -> Self {
        Self {
            settings,
            mdct: Atrac1Mdct::new(),
            analysis_filter_bank: [
                Atrac1AnalysisFilterBank::new(),
                Atrac1AnalysisFilterBank::new(),
            ],
            transient_detectors: [
                ChannelTransientDetectors::new(),
                ChannelTransientDetectors::new(),
            ],
            scaler: Scaler::new(),
            loudness_curve: create_loudness_curve(NUM_SAMPLES),
            loudness: LOUD_FACTOR as f64,

            pcm_buf_low: [[0.0; 256 + 16]; 2],
            pcm_buf_mid: [[0.0; 256 + 16]; 2],
            pcm_buf_hi: [[0.0; 512 + 16]; 2],
        }
    }

    fn encode_channel(
        &mut self,
        pcm: &[f32],
        channel: usize,
    ) -> (Vec<f32>, f64, u32, BlockSizeMod) {
        self.analysis_filter_bank[channel].analysis(
            pcm,
            &mut self.pcm_buf_low[channel][..128],
            &mut self.pcm_buf_mid[channel][..128],
            &mut self.pcm_buf_hi[channel][..256],
        );

        let window_mask = match self.settings.window_mode {
            WindowMode::Auto => {
                let td = &mut self.transient_detectors[channel];
                let mut mask: u32 = 0;
                mask |= td.low.detect(&self.pcm_buf_low[channel][..128]) as u32;
                let inv_mid = invert_spectr(&self.pcm_buf_mid[channel][..128]);
                mask |= (td.mid.detect(&inv_mid) as u32) << 1;
                let inv_hi = invert_spectr(&self.pcm_buf_hi[channel][..256]);
                mask |= (td.hi.detect(&inv_hi) as u32) << 2;
                mask
            }
            WindowMode::NoTransient => self.settings.window_mask,
        };

        let block_size = BlockSizeMod::from_flags(
            window_mask & 1 != 0,
            window_mask & 2 != 0,
            window_mask & 4 != 0,
        );

        let mut specs = vec![0.0f32; 512];
        self.mdct.mdct(
            &mut specs,
            &mut self.pcm_buf_low[channel],
            &mut self.pcm_buf_mid[channel],
            &mut self.pcm_buf_hi[channel],
            &block_size,
        );

        let frame_loudness: f64 = specs
            .iter()
            .zip(self.loudness_curve.iter())
            .map(|(&s, &c)| (s as f64) * (s as f64) * (c as f64))
            .sum();

        (specs, frame_loudness, window_mask, block_size)
    }

    /// Encode 512 PCM samples for one channel (mono). Returns a 212-byte frame.
    pub fn encode_frame(&mut self, pcm: &[f32], channel: usize) -> Vec<u8> {
        let (specs, frame_loudness, window_mask, block_size) =
            self.encode_channel(pcm, channel);

        if window_mask == 0 {
            self.loudness = 0.98 * self.loudness + 0.02 * frame_loudness;
        }

        let scaled_blocks = self.scaler.scale_frame(&specs, &block_size);
        let (frame, _) = write_frame(
            &scaled_blocks,
            &block_size,
            (self.loudness / LOUD_FACTOR as f64) as f32,
            self.settings.bfu_idx_const,
            self.settings.fast_bfu_num_search,
        );

        frame
    }

    /// Encode interleaved PCM (512 * num_channels samples).
    /// Returns one 212-byte frame per channel.
    pub fn encode_frame_interleaved(
        &mut self,
        pcm: &[f32],
        num_channels: usize,
    ) -> Vec<Vec<u8>> {
        let mut channel_data: Vec<(Vec<f32>, f64, u32, BlockSizeMod)> =
            Vec::with_capacity(num_channels);

        for ch in 0..num_channels {
            let mut channel_pcm = [0.0f32; NUM_SAMPLES];
            for i in 0..NUM_SAMPLES {
                channel_pcm[i] = pcm[i * num_channels + ch];
            }
            channel_data.push(self.encode_channel(&channel_pcm, ch));
        }

        let window_masks: Vec<u32> = channel_data.iter().map(|d| d.2).collect();
        if num_channels == 2 && window_masks[0] == 0 && window_masks[1] == 0 {
            // Stereo loudness in f64
            self.loudness =
                0.98 * self.loudness + 0.01 * (channel_data[0].1 + channel_data[1].1);
        } else if window_masks[0] == 0 {
            self.loudness = 0.98 * self.loudness + 0.02 * channel_data[0].1;
        }

        let mut frames = Vec::with_capacity(num_channels);
        for ch in 0..num_channels {
            let (ref specs, _, _, ref block_size) = channel_data[ch];
            let scaled_blocks = self.scaler.scale_frame(specs, block_size);
            let (frame, _) = write_frame(
                &scaled_blocks,
                block_size,
                (self.loudness / LOUD_FACTOR as f64) as f32,
                self.settings.bfu_idx_const,
                self.settings.fast_bfu_num_search,
            );
            frames.push(frame);
        }

        frames
    }
}

#[cfg(test)]
#[path = "../tests/atrac1_encoder_tests.rs"]
mod tests;
