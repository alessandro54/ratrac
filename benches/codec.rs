use criterion::{Criterion, black_box, criterion_group, criterion_main};

use ratrac::atrac1::bitalloc::write_frame;
use ratrac::atrac1::decoder::Atrac1Decoder;
use ratrac::atrac1::dequantiser::dequant;
use ratrac::atrac1::encoder::Atrac1Encoder;
use ratrac::atrac1::mdct_impl::Atrac1Mdct;
use ratrac::atrac1::qmf::{Atrac1AnalysisFilterBank, Atrac1SynthesisFilterBank};
use ratrac::atrac1::{Atrac1EncodeSettings, BlockSizeMod, NUM_SAMPLES};
use ratrac::bitstream::BitStream;
use ratrac::mdct::{Mdct, Midct};
use ratrac::qmf::Qmf;
use ratrac::scaler::Scaler;

fn generate_sine(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
        .collect()
}

// --- Low-level DSP ---

fn bench_mdct_512(c: &mut Criterion) {
    let mut mdct = Mdct::new(512, 1.0);
    let input: Vec<f32> = generate_sine(512);
    c.bench_function("mdct_512", |b| {
        b.iter(|| {
            let result = mdct.process(black_box(&input));
            black_box(result[0]);
        })
    });
}

fn bench_midct_512(c: &mut Criterion) {
    let mut midct = Midct::new(512, 1024.0);
    let input: Vec<f32> = generate_sine(256);
    c.bench_function("midct_512", |b| {
        b.iter(|| {
            let result = midct.process(black_box(&input));
            black_box(result[0]);
        })
    });
}

fn bench_qmf_analysis_512(c: &mut Criterion) {
    let mut qmf = Qmf::new(512);
    let input = generate_sine(512);
    let mut lower = vec![0.0f32; 256];
    let mut upper = vec![0.0f32; 256];
    c.bench_function("qmf_analysis_512", |b| {
        b.iter(|| qmf.analysis(black_box(&input), &mut lower, &mut upper))
    });
}

fn bench_qmf_synthesis_512(c: &mut Criterion) {
    let mut qmf = Qmf::new(512);
    let lower = vec![0.1f32; 256];
    let upper = vec![0.1f32; 256];
    let mut output = vec![0.0f32; 512];
    c.bench_function("qmf_synthesis_512", |b| {
        b.iter(|| qmf.synthesis(&mut output, black_box(&lower), black_box(&upper)))
    });
}

// --- ATRAC1 components ---

fn bench_atrac1_analysis_filter_bank(c: &mut Criterion) {
    let mut fb = Atrac1AnalysisFilterBank::new();
    let pcm = generate_sine(512);
    let mut low = [0.0f32; 128];
    let mut mid = [0.0f32; 128];
    let mut hi = [0.0f32; 256];
    c.bench_function("atrac1_analysis_3band", |b| {
        b.iter(|| fb.analysis(black_box(&pcm), &mut low, &mut mid, &mut hi))
    });
}

fn bench_atrac1_synthesis_filter_bank(c: &mut Criterion) {
    let mut fb = Atrac1SynthesisFilterBank::new();
    let low = [0.1f32; 128];
    let mid = [0.1f32; 128];
    let hi = [0.1f32; 256];
    let mut pcm = [0.0f32; 512];
    c.bench_function("atrac1_synthesis_3band", |b| {
        b.iter(|| fb.synthesis(&mut pcm, black_box(&low), black_box(&mid), black_box(&hi)))
    });
}

fn bench_atrac1_mdct_forward(c: &mut Criterion) {
    let mut mdct = Atrac1Mdct::new();
    let mut low = vec![0.0f32; 256 + 16];
    let mut mid = vec![0.0f32; 256 + 16];
    let mut hi = vec![0.0f32; 512 + 16];
    let mut specs = vec![0.0f32; 512];
    for i in 0..128 {
        low[i] = (i as f32 * 0.1).sin();
        mid[i] = (i as f32 * 0.2).cos();
    }
    for i in 0..256 {
        hi[i] = (i as f32 * 0.05).sin();
    }
    let bs = BlockSizeMod::new();
    c.bench_function("atrac1_mdct_forward", |b| {
        b.iter(|| mdct.mdct(&mut specs, &mut low, &mut mid, &mut hi, black_box(&bs)))
    });
}

fn bench_atrac1_mdct_inverse(c: &mut Criterion) {
    let mut mdct = Atrac1Mdct::new();
    let mut specs: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
    let mut low = vec![0.0f32; 256 + 16];
    let mut mid = vec![0.0f32; 256 + 16];
    let mut hi = vec![0.0f32; 512 + 16];
    let bs = BlockSizeMod::new();
    c.bench_function("atrac1_mdct_inverse", |b| {
        b.iter(|| mdct.imdct(&mut specs, black_box(&bs), &mut low, &mut mid, &mut hi))
    });
}

fn bench_scaler(c: &mut Criterion) {
    let scaler = Scaler::new();
    let specs: Vec<f32> = (0..512).map(|i| 0.3 * (i as f32 * 0.03).sin()).collect();
    let bs = BlockSizeMod::new();
    c.bench_function("scaler_frame", |b| {
        b.iter(|| scaler.scale_frame(black_box(&specs), &bs))
    });
}

fn bench_bitalloc_write_frame(c: &mut Criterion) {
    let scaler = Scaler::new();
    let specs: Vec<f32> = (0..512).map(|i| 0.3 * (i as f32 * 0.03).sin()).collect();
    let bs = BlockSizeMod::new();
    let scaled = scaler.scale_frame(&specs, &bs);
    c.bench_function("bitalloc_write_frame", |b| {
        b.iter(|| write_frame(black_box(&scaled), &bs, 0.5, 0, false))
    });
}

fn bench_dequantise(c: &mut Criterion) {
    // Build a realistic frame
    let scaler = Scaler::new();
    let specs: Vec<f32> = (0..512).map(|i| 0.3 * (i as f32 * 0.03).sin()).collect();
    let bs = BlockSizeMod::new();
    let scaled = scaler.scale_frame(&specs, &bs);
    let (frame, _) = write_frame(&scaled, &bs, 0.5, 0, false);

    c.bench_function("dequantise_frame", |b| {
        b.iter(|| {
            let mut stream = BitStream::from_bytes(&frame);
            let mode = BlockSizeMod::from_bitstream(&mut stream);
            let mut out = [0.0f32; 512];
            dequant(&mut stream, &mode, &mut out);
            black_box(out);
        })
    });
}

// --- Full pipeline ---

fn bench_encode_frame(c: &mut Criterion) {
    let mut encoder = Atrac1Encoder::new(Atrac1EncodeSettings::default());
    let pcm = generate_sine(NUM_SAMPLES);
    // Warmup
    for _ in 0..5 {
        encoder.encode_frame(&pcm, 0);
    }
    c.bench_function("encode_frame_mono", |b| {
        b.iter(|| encoder.encode_frame(black_box(&pcm), 0))
    });
}

fn bench_decode_frame(c: &mut Criterion) {
    // Prepare a frame
    let mut encoder = Atrac1Encoder::new(Atrac1EncodeSettings::default());
    let pcm = generate_sine(NUM_SAMPLES);
    for _ in 0..5 {
        encoder.encode_frame(&pcm, 0);
    }
    let frame = encoder.encode_frame(&pcm, 0);

    let mut decoder = Atrac1Decoder::new();
    // Warmup
    for _ in 0..5 {
        decoder.decode_frame(&frame, 0);
    }
    c.bench_function("decode_frame_mono", |b| {
        b.iter(|| decoder.decode_frame(black_box(&frame), 0))
    });
}

criterion_group!(
    dsp,
    bench_mdct_512,
    bench_midct_512,
    bench_qmf_analysis_512,
    bench_qmf_synthesis_512,
);

criterion_group!(
    atrac1_components,
    bench_atrac1_analysis_filter_bank,
    bench_atrac1_synthesis_filter_bank,
    bench_atrac1_mdct_forward,
    bench_atrac1_mdct_inverse,
    bench_scaler,
    bench_bitalloc_write_frame,
    bench_dequantise,
);

criterion_group!(pipeline, bench_encode_frame, bench_decode_frame,);

criterion_main!(dsp, atrac1_components, pipeline);
