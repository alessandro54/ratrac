use std::path::PathBuf;

use clap::{Parser, Subcommand};
use hound::{WavReader, WavSpec, WavWriter};

use ratrac::aea::{AeaReader, AeaWriter, AEA_FRAME_SIZE};
use ratrac::atrac1::decoder::Atrac1Decoder;
use ratrac::atrac1::encoder::Atrac1Encoder;
use ratrac::atrac1::{Atrac1EncodeSettings, WindowMode, NUM_SAMPLES};

#[derive(Parser)]
#[command(name = "ratrac", about = "ATRAC1 encoder/decoder in Rust")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Encode a WAV file to ATRAC1 (.aea)
    Encode {
        /// Input WAV file
        #[arg(short, long)]
        input: PathBuf,

        /// Output AEA file
        #[arg(short, long)]
        output: PathBuf,

        /// Fixed BFU index (1-8, 0 = auto)
        #[arg(long, default_value = "0")]
        bfuidxconst: u32,

        /// Use fast BFU number search
        #[arg(long, default_value = "false")]
        bfuidxfast: bool,

        /// Disable transient detection (optional window mask: 0-7)
        #[arg(long)]
        notransient: Option<Option<u32>>,
    },
    /// Decode an ATRAC1 (.aea) file to WAV
    Decode {
        /// Input AEA file
        #[arg(short, long)]
        input: PathBuf,

        /// Output WAV file
        #[arg(short, long)]
        output: PathBuf,
    },
}

fn encode(
    input: &PathBuf,
    output: &PathBuf,
    bfu_idx_const: u32,
    bfu_idx_fast: bool,
    notransient: Option<Option<u32>>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Open WAV
    let reader = WavReader::open(input)?;
    let spec = reader.spec();

    if spec.sample_rate != 44100 {
        return Err(format!("Unsupported sample rate: {} (must be 44100)", spec.sample_rate).into());
    }

    let num_channels = spec.channels as usize;
    let total_samples = reader.len() as u64 / num_channels as u64;

    eprintln!(
        "Input: {} ({} ch, {} Hz, {} samples)",
        input.display(),
        num_channels,
        spec.sample_rate,
        total_samples,
    );

    // Read all PCM samples as f32
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1u32 << (bits - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.unwrap() as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.unwrap())
            .collect(),
    };

    // Encoder settings
    let (window_mode, window_mask) = match notransient {
        Some(mask_opt) => {
            let mask = mask_opt.unwrap_or(0);
            eprintln!(
                "Transient detection disabled, bands: low={}, mid={}, hi={}",
                if mask & 1 != 0 { "short" } else { "long" },
                if mask & 2 != 0 { "short" } else { "long" },
                if mask & 4 != 0 { "short" } else { "long" },
            );
            (WindowMode::NoTransient, mask)
        }
        None => (WindowMode::Auto, 0),
    };

    let settings = Atrac1EncodeSettings {
        bfu_idx_const,
        fast_bfu_num_search: bfu_idx_fast,
        window_mode,
        window_mask,
    };

    let mut encoder = Atrac1Encoder::new(settings);

    // Calculate frames
    let num_frames = (total_samples + NUM_SAMPLES as u64 - 1) / NUM_SAMPLES as u64;

    // Create AEA output
    let mut writer = AeaWriter::create(
        output,
        "ratrac",
        num_channels,
        (num_frames * num_channels as u64) as u32,
    )?;

    // First write is skipped (dummy)
    writer.write_frame(&[0; AEA_FRAME_SIZE])?;

    eprintln!(
        "Output: {} (ATRAC1, {} frames)",
        output.display(),
        num_frames,
    );

    // Process frames
    let frame_size = NUM_SAMPLES * num_channels;
    let mut pos = 0;
    let mut frame_count = 0u64;

    while pos < samples.len() {
        // Get one frame of interleaved PCM, zero-pad if needed
        let mut pcm_frame = vec![0.0f32; frame_size];
        let remaining = samples.len() - pos;
        let copy_len = remaining.min(frame_size);
        pcm_frame[..copy_len].copy_from_slice(&samples[pos..pos + copy_len]);

        // Encode
        let frames = encoder.encode_frame_interleaved(&pcm_frame, num_channels);
        for frame in &frames {
            writer.write_frame(frame)?;
        }

        pos += frame_size;
        frame_count += 1;

        if frame_count % 100 == 0 {
            let pct = (frame_count * 100) / num_frames;
            eprint!("\r  {}% done", pct);
        }
    }

    writer.flush()?;
    eprintln!("\rDone ({} frames encoded)", frame_count);
    Ok(())
}

fn decode(input: &PathBuf, output: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    // Open AEA
    let mut reader = AeaReader::open(input)?;
    let num_channels = reader.channel_num();
    let total_samples = reader.length_in_samples();

    eprintln!(
        "Input: {} ({} ch, name=\"{}\", {} samples)",
        input.display(),
        num_channels,
        reader.name(),
        total_samples,
    );

    // Create WAV output
    let spec = WavSpec {
        channels: num_channels as u16,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut wav_writer = WavWriter::create(output, spec)?;

    let mut decoder = Atrac1Decoder::new();

    // Skip dummy frame
    reader.read_frame()?;

    eprintln!("Output: {} (WAV, 16-bit)", output.display());

    let mut frame_count = 0u64;
    let total_frames = if total_samples > 0 {
        total_samples / NUM_SAMPLES as u64
    } else {
        u64::MAX
    };

    loop {
        let result = decoder.decode_frame_interleaved(&mut reader);
        match result {
            Some(samples) => {
                for &s in &samples {
                    // Convert f32 [-1, 1] to i16
                    let clamped = s.clamp(-1.0, 1.0);
                    let sample_i16 = (clamped * 32767.0) as i16;
                    wav_writer.write_sample(sample_i16)?;
                }
                frame_count += 1;

                if frame_count % 100 == 0 {
                    let pct = (frame_count * 100).min(total_frames * 100) / total_frames.max(1);
                    eprint!("\r  {}% done", pct);
                }
            }
            None => break,
        }
    }

    wav_writer.finalize()?;
    eprintln!("\rDone ({} frames decoded)", frame_count);
    Ok(())
}

fn main() {
    let cli = Cli::parse();

    let result = match &cli.command {
        Commands::Encode {
            input,
            output,
            bfuidxconst,
            bfuidxfast,
            notransient,
        } => encode(input, output, *bfuidxconst, *bfuidxfast, *notransient),
        Commands::Decode { input, output } => decode(input, output),
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
