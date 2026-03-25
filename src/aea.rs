use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

pub const AEA_META_SIZE: usize = 2048;
pub const AEA_FRAME_SIZE: usize = 212;
const AEA_MAGIC: [u8; 4] = [0x00, 0x08, 0x00, 0x00];

#[derive(Debug)]
pub enum AeaError {
    Io(io::Error),
    InvalidFormat,
    InvalidChannels,
}

impl From<io::Error> for AeaError {
    fn from(e: io::Error) -> Self {
        AeaError::Io(e)
    }
}

impl std::fmt::Display for AeaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AeaError::Io(e) => write!(f, "AEA I/O error: {e}"),
            AeaError::InvalidFormat => write!(f, "Invalid AEA format (bad magic bytes)"),
            AeaError::InvalidChannels => write!(f, "Invalid AEA channel count (must be 1 or 2)"),
        }
    }
}

impl std::error::Error for AeaError {}

/// AEA file reader for ATRAC1 compressed data.
pub struct AeaReader {
    file: File,
    header: [u8; AEA_META_SIZE],
    file_size: u64,
}

impl AeaReader {
    pub fn open(path: &Path) -> Result<Self, AeaError> {
        let mut file = File::open(path)?;
        let file_size = file.metadata()?.len();

        let mut header = [0u8; AEA_META_SIZE];
        file.read_exact(&mut header)?;

        // Validate magic bytes
        if header[0..4] != AEA_MAGIC {
            return Err(AeaError::InvalidFormat);
        }

        // Validate channel count
        if header[264] >= 3 {
            return Err(AeaError::InvalidChannels);
        }

        Ok(Self {
            file,
            header,
            file_size,
        })
    }

    /// Number of channels (1 or 2).
    pub fn channel_num(&self) -> usize {
        let ch = self.header[264];
        if ch == 0 { 1 } else { ch as usize }
    }

    /// Title string from header (bytes 4..20).
    pub fn name(&self) -> String {
        let end = self.header[4..20]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(16);
        String::from_utf8_lossy(&self.header[4..4 + end]).to_string()
    }

    /// Total number of samples.
    /// Formula: 512 * ((file_size - 2048) / 212 / channels - 5)
    pub fn length_in_samples(&self) -> u64 {
        let n_channels = self.channel_num() as u64;
        let data_size = self.file_size - AEA_META_SIZE as u64;
        let total_frames = data_size / AEA_FRAME_SIZE as u64 / n_channels;
        if total_frames < 5 {
            return 0;
        }
        512 * (total_frames - 5)
    }

    /// Read one 212-byte frame. Returns None at EOF.
    pub fn read_frame(&mut self) -> Result<Option<Vec<u8>>, AeaError> {
        let mut buf = vec![0u8; AEA_FRAME_SIZE];
        match self.file.read_exact(&mut buf) {
            Ok(()) => Ok(Some(buf)),
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(None),
            Err(e) => Err(AeaError::Io(e)),
        }
    }

    /// Number of frames stored in header at offset 260 (LE u32).
    pub fn num_frames(&self) -> u32 {
        u32::from_le_bytes([
            self.header[260],
            self.header[261],
            self.header[262],
            self.header[263],
        ])
    }
}

/// AEA file writer for ATRAC1 compressed data.
pub struct AeaWriter {
    file: File,
    first_write: bool,
}

impl AeaWriter {
    pub fn create(
        path: &Path,
        title: &str,
        num_channels: usize,
        num_frames: u32,
    ) -> Result<Self, AeaError> {
        let mut file = File::create(path)?;

        // Build header
        let mut header = [0u8; AEA_META_SIZE];
        header[0..4].copy_from_slice(&AEA_MAGIC);

        // Title (max 16 bytes, null-terminated at byte 19)
        let title_bytes = title.as_bytes();
        let copy_len = title_bytes.len().min(15);
        header[4..4 + copy_len].copy_from_slice(&title_bytes[..copy_len]);
        header[19] = 0;

        // Frame count at offset 260 (LE u32)
        header[260..264].copy_from_slice(&num_frames.to_le_bytes());

        // Channel count at offset 264
        header[264] = num_channels as u8;

        // Write header
        file.write_all(&header)?;

        // Write dummy frame (212 zero bytes)
        let dummy = [0u8; AEA_FRAME_SIZE];
        file.write_all(&dummy)?;

        Ok(Self {
            file,
            first_write: true,
        })
    }

    /// Write a frame. The first call is silently skipped (matching C++ behavior).
    pub fn write_frame(&mut self, data: &[u8]) -> Result<(), AeaError> {
        if self.first_write {
            self.first_write = false;
            return Ok(());
        }

        // Pad or truncate to 212 bytes
        let mut frame = [0u8; AEA_FRAME_SIZE];
        let copy_len = data.len().min(AEA_FRAME_SIZE);
        frame[..copy_len].copy_from_slice(&data[..copy_len]);

        self.file.write_all(&frame)?;
        Ok(())
    }

    /// Flush and finalize.
    pub fn flush(&mut self) -> Result<(), AeaError> {
        self.file.flush()?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "tests/aea_tests.rs"]
mod tests;
