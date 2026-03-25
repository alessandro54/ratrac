use std::path::PathBuf;

use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};

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

fn make_progress_bar(total: u64, msg: &str) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template(
            "{msg} {bar:40.green/black} {percent}% ({pos}/{len}) [{elapsed_precise} / {eta_precise}]",
        )
        .unwrap()
        .progress_chars("█░"),
    );
    pb.set_message(msg.to_string());
    pb
}

fn format_duration(seconds: f64) -> String {
    let mins = (seconds / 60.0) as u64;
    let secs = seconds % 60.0;
    if mins > 0 {
        format!("{mins}m {secs:.0}s")
    } else {
        format!("{secs:.1}s")
    }
}

fn encode(
    input: &PathBuf,
    output: &PathBuf,
    bfu_idx_const: u32,
    bfu_idx_fast: bool,
    notransient: Option<Option<u32>>,
) -> Result<(), Box<dyn std::error::Error>> {
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
    let duration_secs = total_samples as f64 / 44100.0;

    eprintln!(
        "  Input:  {} ({} ch, {} Hz, {})",
        input.display(),
        num_channels,
        audio.sample_rate,
        format_duration(duration_secs),
    );

    let samples = audio.samples;

    let (window_mode, window_mask) = match notransient {
        Some(mask_opt) => {
            let mask = mask_opt.unwrap_or(0);
            eprintln!(
                "  Transient detection disabled (mask: low={}, mid={}, hi={})",
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
    let num_frames = total_samples.div_ceil(NUM_SAMPLES as u64);

    let mut writer = AeaWriter::create(
        output,
        "ratrac",
        num_channels,
        (num_frames * num_channels as u64) as u32,
    )?;
    writer.write_frame(&[0; AEA_FRAME_SIZE])?;

    let input_size = std::fs::metadata(input).map(|m| m.len()).unwrap_or(0);
    let output_size_est = 2048 + num_frames * 212 * num_channels as u64;
    let ratio = input_size as f64 / output_size_est as f64;

    eprintln!(
        "  Output: {} (ATRAC1, {} frames, ~{:.1}x compression)",
        output.display(),
        num_frames,
        ratio,
    );

    let pb = make_progress_bar(num_frames, "Encoding");

    let frame_size = NUM_SAMPLES * num_channels;
    let mut pos = 0;

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
        pb.inc(1);
    }

    writer.flush()?;
    pb.finish_and_clear();
    eprintln!("  Done.");
    Ok(())
}

fn decode(input: &PathBuf, output: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = AeaReader::open(input)?;
    let num_channels = reader.channel_num();
    let total_samples = reader.length_in_samples();
    let duration_secs = total_samples as f64 / 44100.0;

    eprintln!(
        "  Input:  {} ({} ch, name=\"{}\", {})",
        input.display(),
        num_channels,
        reader.name(),
        format_duration(duration_secs),
    );
    eprintln!("  Output: {} (WAV, 16-bit)", output.display());

    let mut wav_writer = WavWriter::create(output, num_channels as u16, 44100)?;
    let mut decoder = Atrac1Decoder::new();

    reader.read_frame()?;

    let total_frames = if total_samples > 0 {
        total_samples / NUM_SAMPLES as u64
    } else {
        0
    };

    let pb = make_progress_bar(total_frames, "Decoding");

    loop {
        let result = decoder.decode_frame_interleaved(&mut reader);
        match result {
            Some(samples) => {
                wav_writer.write_samples(&samples)?;
                pb.inc(1);
            }
            None => break,
        }
    }

    wav_writer.finalize()?;
    pb.finish_and_clear();
    eprintln!("  Done.");
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
