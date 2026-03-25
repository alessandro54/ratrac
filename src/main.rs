use std::path::PathBuf;

use clap::{Parser, Subcommand};

use ratrac::aea::{AEA_FRAME_SIZE, AeaReader, AeaWriter};
use ratrac::atrac1::decoder::Atrac1Decoder;
use ratrac::atrac1::encoder::Atrac1Encoder;
use ratrac::atrac1::{Atrac1EncodeSettings, NUM_SAMPLES, WindowMode};
use ratrac::audio_input::read_audio;
use ratrac::wav_output::WavWriter;

#[derive(Parser)]
#[command(name = "ratrac", about = "ATRAC1 encoder/decoder in Rust")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Encode an audio file to ATRAC1 (.aea)
    /// Supports: WAV, FLAC, MP3, AAC, OGG/Vorbis
    Encode {
        /// Input audio file
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
    // Read audio using symphonia (supports WAV, FLAC, MP3, AAC, OGG)
    let audio = read_audio(input)?;

    if audio.sample_rate != 44100 {
        return Err(format!(
            "Unsupported sample rate: {} (must be 44100)",
            audio.sample_rate
        )
        .into());
    }

    let num_channels = audio.channels;
    let total_samples = audio.num_frames();

    eprintln!(
        "Input: {} ({} ch, {} Hz, {} samples)",
        input.display(),
        num_channels,
        audio.sample_rate,
        total_samples,
    );

    let samples = audio.samples;

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
    let num_frames = total_samples.div_ceil(NUM_SAMPLES as u64);

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
        let mut pcm_frame = vec![0.0f32; frame_size];
        let remaining = samples.len() - pos;
        let copy_len = remaining.min(frame_size);
        pcm_frame[..copy_len].copy_from_slice(&samples[pos..pos + copy_len]);

        let frames = encoder.encode_frame_interleaved(&pcm_frame, num_channels);
        for frame in &frames {
            writer.write_frame(frame)?;
        }

        pos += frame_size;
        frame_count += 1;

        if frame_count % 100 == 0 {
            let pct = (frame_count * 100) / num_frames;
            eprint!("\r  {pct}% done");
        }
    }

    writer.flush()?;
    eprintln!("\rDone ({frame_count} frames encoded)");
    Ok(())
}

fn decode(input: &PathBuf, output: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
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

    let mut wav_writer = WavWriter::create(output, num_channels as u16, 44100)?;

    let mut decoder = Atrac1Decoder::new();

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
                wav_writer.write_samples(&samples)?;
                frame_count += 1;

                if frame_count % 100 == 0 {
                    let pct = (frame_count * 100).min(total_frames * 100) / total_frames.max(1);
                    eprint!("\r  {pct}% done");
                }
            }
            None => break,
        }
    }

    wav_writer.finalize()?;
    eprintln!("\rDone ({frame_count} frames decoded)");
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
