use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

const BUF_SIZE: usize = 64 * 1024; // 64KB write buffer

/// Simple WAV file writer (16-bit PCM only) with buffered I/O.
pub struct WavWriter {
    writer: BufWriter<File>,
    num_channels: u16,
    sample_rate: u32,
    data_bytes: u32,
}

impl WavWriter {
    pub fn create(path: &Path, num_channels: u16, sample_rate: u32) -> io::Result<Self> {
        let file = File::create(path)?;
        let mut w = Self {
            writer: BufWriter::with_capacity(BUF_SIZE, file),
            num_channels,
            sample_rate,
            data_bytes: 0,
        };
        w.write_header()?;
        Ok(w)
    }

    fn write_header(&mut self) -> io::Result<()> {
        let bits_per_sample: u16 = 16;
        let byte_rate = self.sample_rate * self.num_channels as u32 * 2;
        let block_align = self.num_channels * 2;
        let data_size = self.data_bytes;
        let file_size = 36 + data_size;

        self.writer.write_all(b"RIFF")?;
        self.writer.write_all(&file_size.to_le_bytes())?;
        self.writer.write_all(b"WAVE")?;

        self.writer.write_all(b"fmt ")?;
        self.writer.write_all(&16u32.to_le_bytes())?;
        self.writer.write_all(&1u16.to_le_bytes())?;
        self.writer.write_all(&self.num_channels.to_le_bytes())?;
        self.writer.write_all(&self.sample_rate.to_le_bytes())?;
        self.writer.write_all(&byte_rate.to_le_bytes())?;
        self.writer.write_all(&block_align.to_le_bytes())?;
        self.writer.write_all(&bits_per_sample.to_le_bytes())?;

        self.writer.write_all(b"data")?;
        self.writer.write_all(&data_size.to_le_bytes())?;

        Ok(())
    }

    /// Write interleaved f32 samples, converting to i16.
    /// Batches the conversion into a byte buffer before writing.
    pub fn write_samples(&mut self, samples: &[f32]) -> io::Result<()> {
        // Convert all samples to bytes in one batch
        let mut byte_buf = Vec::with_capacity(samples.len() * 2);
        for &s in samples {
            let clamped = s.clamp(-1.0, 1.0);
            let i16_val = (clamped * 32767.0).round_ties_even() as i16;
            byte_buf.extend_from_slice(&i16_val.to_le_bytes());
        }
        self.writer.write_all(&byte_buf)?;
        self.data_bytes += byte_buf.len() as u32;
        Ok(())
    }

    /// Finalize: flush buffer, rewrite header with correct sizes.
    pub fn finalize(mut self) -> io::Result<()> {
        self.writer.flush()?;
        // Get inner file for seeking
        let mut file = self.writer.into_inner()?;
        use std::io::Seek;
        file.seek(io::SeekFrom::Start(0))?;

        // Rewrite header directly on file
        let bits_per_sample: u16 = 16;
        let byte_rate = self.sample_rate * self.num_channels as u32 * 2;
        let block_align = self.num_channels * 2;
        let file_size = 36 + self.data_bytes;

        file.write_all(b"RIFF")?;
        file.write_all(&file_size.to_le_bytes())?;
        file.write_all(b"WAVE")?;
        file.write_all(b"fmt ")?;
        file.write_all(&16u32.to_le_bytes())?;
        file.write_all(&1u16.to_le_bytes())?;
        file.write_all(&self.num_channels.to_le_bytes())?;
        file.write_all(&self.sample_rate.to_le_bytes())?;
        file.write_all(&byte_rate.to_le_bytes())?;
        file.write_all(&block_align.to_le_bytes())?;
        file.write_all(&bits_per_sample.to_le_bytes())?;
        file.write_all(b"data")?;
        file.write_all(&self.data_bytes.to_le_bytes())?;
        file.flush()?;

        Ok(())
    }
}
