use std::fs::File;
use std::path::Path;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Audio stream metadata (available before decoding starts).
pub struct AudioInfo {
    pub sample_rate: u32,
    pub channels: usize,
    pub total_frames: Option<u64>,
}

/// Streaming audio reader. Yields chunks of interleaved f32 PCM on demand.
pub struct AudioReader {
    format: Box<dyn FormatReader>,
    decoder: Box<dyn symphonia::core::codecs::Decoder>,
    track_id: u32,
    pub info: AudioInfo,
}

impl AudioReader {
    /// Open an audio file for streaming decode.
    pub fn open(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }

        let probed = symphonia::default::get_probe().format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )?;

        let format = probed.format;

        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or("No audio track found")?;

        let track_id = track.id;
        let codec_params = track.codec_params.clone();

        let sample_rate = codec_params.sample_rate.ok_or("Unknown sample rate")?;
        let channels = codec_params
            .channels
            .map(|c| c.count())
            .ok_or("Unknown channel count")?;
        let total_frames = codec_params.n_frames;

        let decoder =
            symphonia::default::get_codecs().make(&codec_params, &DecoderOptions::default())?;

        Ok(Self {
            format,
            decoder,
            track_id,
            info: AudioInfo {
                sample_rate,
                channels,
                total_frames,
            },
        })
    }

    /// Read the next chunk of interleaved f32 samples.
    /// Returns None at EOF.
    pub fn next_chunk(&mut self) -> Result<Option<Vec<f32>>, Box<dyn std::error::Error>> {
        loop {
            let packet = match self.format.next_packet() {
                Ok(p) => p,
                Err(symphonia::core::errors::Error::IoError(ref e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    return Ok(None);
                }
                Err(e) => return Err(e.into()),
            };

            if packet.track_id() != self.track_id {
                continue;
            }

            let decoded = self.decoder.decode(&packet)?;
            let spec = *decoded.spec();
            let num_frames = decoded.frames();

            let mut sample_buf = SampleBuffer::<f32>::new(num_frames as u64, spec);
            sample_buf.copy_interleaved_ref(decoded);

            return Ok(Some(sample_buf.samples().to_vec()));
        }
    }
}

/// Buffered frame reader: accumulates samples from the streaming reader
/// and yields exact-sized frames for the encoder.
pub struct FrameReader {
    reader: AudioReader,
    buffer: Vec<f32>,
    frame_size: usize, // NUM_SAMPLES * channels
    finished: bool,
}

impl FrameReader {
    pub fn new(reader: AudioReader, samples_per_frame: usize) -> Self {
        let frame_size = samples_per_frame * reader.info.channels;
        Self {
            reader,
            buffer: Vec::new(),
            frame_size,
            finished: false,
        }
    }

    pub fn info(&self) -> &AudioInfo {
        &self.reader.info
    }

    /// Get the next frame of exactly `frame_size` interleaved samples.
    /// Zero-pads the last frame if needed. Returns None when all data is consumed.
    pub fn next_frame(&mut self) -> Result<Option<Vec<f32>>, Box<dyn std::error::Error>> {
        // Fill buffer until we have enough for one frame
        while self.buffer.len() < self.frame_size && !self.finished {
            match self.reader.next_chunk()? {
                Some(chunk) => self.buffer.extend_from_slice(&chunk),
                None => self.finished = true,
            }
        }

        if self.buffer.is_empty() {
            return Ok(None);
        }

        // Take one frame from the buffer
        let mut frame = vec![0.0f32; self.frame_size];
        let copy_len = self.buffer.len().min(self.frame_size);
        frame[..copy_len].copy_from_slice(&self.buffer[..copy_len]);

        // Remove consumed samples
        if copy_len >= self.buffer.len() {
            self.buffer.clear();
        } else {
            self.buffer = self.buffer[copy_len..].to_vec();
        }

        Ok(Some(frame))
    }
}

// Keep the old API for tests and backward compatibility
pub struct AudioData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: usize,
}

impl AudioData {
    pub fn num_frames(&self) -> u64 {
        self.samples.len() as u64 / self.channels as u64
    }
}

/// Read entire audio file into memory (legacy API, used by tests).
pub fn read_audio(path: &Path) -> Result<AudioData, Box<dyn std::error::Error>> {
    let mut reader = AudioReader::open(path)?;
    let mut all_samples = Vec::new();

    while let Some(chunk) = reader.next_chunk()? {
        all_samples.extend_from_slice(&chunk);
    }

    Ok(AudioData {
        samples: all_samples,
        sample_rate: reader.info.sample_rate,
        channels: reader.info.channels,
    })
}
