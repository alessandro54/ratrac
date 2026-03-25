use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

/// Simple WAV file writer (16-bit PCM only).
pub struct WavWriter {
    file: File,
    num_channels: u16,
    sample_rate: u32,
    data_bytes: u32,
}

impl WavWriter {
    pub fn create(path: &Path, num_channels: u16, sample_rate: u32) -> io::Result<Self> {
        let file = File::create(path)?;
        let mut w = Self {
            file,
            num_channels,
            sample_rate,
            data_bytes: 0,
        };
        // Write placeholder header (will be patched on finalize)
        w.write_header()?;
        Ok(w)
    }

    fn write_header(&mut self) -> io::Result<()> {
        let bits_per_sample: u16 = 16;
        let byte_rate = self.sample_rate * self.num_channels as u32 * 2;
        let block_align = self.num_channels * 2;
        let data_size = self.data_bytes;
        let file_size = 36 + data_size;

        self.file.write_all(b"RIFF")?;
        self.file.write_all(&file_size.to_le_bytes())?;
        self.file.write_all(b"WAVE")?;

        // fmt chunk
        self.file.write_all(b"fmt ")?;
        self.file.write_all(&16u32.to_le_bytes())?; // chunk size
        self.file.write_all(&1u16.to_le_bytes())?; // PCM format
        self.file.write_all(&self.num_channels.to_le_bytes())?;
        self.file.write_all(&self.sample_rate.to_le_bytes())?;
        self.file.write_all(&byte_rate.to_le_bytes())?;
        self.file.write_all(&block_align.to_le_bytes())?;
        self.file.write_all(&bits_per_sample.to_le_bytes())?;

        // data chunk
        self.file.write_all(b"data")?;
        self.file.write_all(&data_size.to_le_bytes())?;

        Ok(())
    }

    /// Write interleaved f32 samples, converting to i16.
    pub fn write_samples(&mut self, samples: &[f32]) -> io::Result<()> {
        for &s in samples {
            let clamped = s.clamp(-1.0, 1.0);
            let i16_val = (clamped * 32767.0) as i16;
            self.file.write_all(&i16_val.to_le_bytes())?;
        }
        self.data_bytes += (samples.len() * 2) as u32;
        Ok(())
    }

    /// Finalize: rewrite header with correct sizes.
    pub fn finalize(mut self) -> io::Result<()> {
        use std::io::Seek;
        self.file.seek(io::SeekFrom::Start(0))?;
        self.write_header()?;
        self.file.flush()?;
        Ok(())
    }
}
