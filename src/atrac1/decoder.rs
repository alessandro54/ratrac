use crate::aea::AeaReader;
use crate::atrac1::qmf::Atrac1SynthesisFilterBank;
use crate::atrac1::dequantiser::dequant;
use crate::atrac1::mdct_impl::Atrac1Mdct;
use crate::atrac1::{BlockSizeMod, NUM_SAMPLES};
use crate::bitstream::BitStream;

/// Full ATRAC1 decoder: AEA frames → PCM samples.
pub struct Atrac1Decoder {
    mdct: Atrac1Mdct,
    synthesis_filter_bank: [Atrac1SynthesisFilterBank; 2],
    /// Per-channel band buffers with overlap tail space.
    /// low/mid: 256 + 16, hi: 512 + 16
    pcm_buf_low: [[f32; 256 + 16]; 2],
    pcm_buf_mid: [[f32; 256 + 16]; 2],
    pcm_buf_hi: [[f32; 512 + 16]; 2],
    pcm_value_max: f32,
    pcm_value_min: f32,
}

impl Atrac1Decoder {
    pub fn new() -> Self {
        Self {
            mdct: Atrac1Mdct::new(),
            synthesis_filter_bank: [
                Atrac1SynthesisFilterBank::new(),
                Atrac1SynthesisFilterBank::new(),
            ],
            pcm_buf_low: [[0.0; 256 + 16]; 2],
            pcm_buf_mid: [[0.0; 256 + 16]; 2],
            pcm_buf_hi: [[0.0; 512 + 16]; 2],
            pcm_value_max: 1.0,
            pcm_value_min: -1.0,
        }
    }

    /// Decode a single raw AEA frame (212 bytes) for one channel.
    /// Returns 512 PCM samples.
    pub fn decode_frame(&mut self, frame_data: &[u8], channel: usize) -> [f32; NUM_SAMPLES] {
        let mut bs = BitStream::from_bytes(frame_data);

        // Parse block size mode (8 bits)
        let mode = BlockSizeMod::from_bitstream(&mut bs);

        // Dequantise spectral coefficients
        let mut specs = [0.0f32; 512];
        dequant(&mut bs, &mode, &mut specs);

        // Inverse MDCT
        self.mdct.imdct(
            &mut specs,
            &mode,
            &mut self.pcm_buf_low[channel],
            &mut self.pcm_buf_mid[channel],
            &mut self.pcm_buf_hi[channel],
        );

        // QMF synthesis
        let mut sum = [0.0f32; 512];
        self.synthesis_filter_bank[channel].synthesis(
            &mut sum,
            &self.pcm_buf_low[channel][..128],
            &self.pcm_buf_mid[channel][..128],
            &self.pcm_buf_hi[channel][..256],
        );

        // Clip
        for sample in &mut sum {
            if *sample > self.pcm_value_max {
                *sample = self.pcm_value_max;
            }
            if *sample < self.pcm_value_min {
                *sample = self.pcm_value_min;
            }
        }

        sum
    }

    /// Decode one frame from a multi-channel AEA stream.
    /// Returns interleaved PCM: [ch0_s0, ch1_s0, ch0_s1, ch1_s1, ...]
    /// or mono: [s0, s1, s2, ...]
    pub fn decode_frame_interleaved(
        &mut self,
        reader: &mut AeaReader,
    ) -> Option<Vec<f32>> {
        let num_channels = reader.channel_num();
        let mut output = vec![0.0f32; NUM_SAMPLES * num_channels];

        for channel in 0..num_channels {
            let frame_data = reader.read_frame().ok()??;
            let samples = self.decode_frame(&frame_data, channel);

            for i in 0..NUM_SAMPLES {
                output[i * num_channels + channel] = samples[i];
            }
        }

        Some(output)
    }
}

#[cfg(test)]
#[path = "../tests/atrac1_decoder_tests.rs"]
mod tests;
