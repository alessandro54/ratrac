/// ATH lookup table from Musepack (Frank Klemm).
/// 128 entries in millibel relative to 20 µPa.
/// Covers 10 Hz to ~30 kHz, 4 steps per third of an octave.
const ATH_TAB: [i16; 140] = [
    /*    10.0 */  9669, 9669, 9626, 9512,
    /*    12.6 */  9353, 9113, 8882, 8676,
    /*    15.8 */  8469, 8243, 7997, 7748,
    /*    20.0 */  7492, 7239, 7000, 6762,
    /*    25.1 */  6529, 6302, 6084, 5900,
    /*    31.6 */  5717, 5534, 5351, 5167,
    /*    39.8 */  5004, 4812, 4638, 4466,
    /*    50.1 */  4310, 4173, 4050, 3922,
    /*    63.1 */  3723, 3577, 3451, 3281,
    /*    79.4 */  3132, 3036, 2902, 2760,
    /*   100.0 */  2658, 2591, 2441, 2301,
    /*   125.9 */  2212, 2125, 2018, 1900,
    /*   158.5 */  1770, 1682, 1594, 1512,
    /*   199.5 */  1430, 1341, 1260, 1198,
    /*   251.2 */  1136, 1057,  998,  943,
    /*   316.2 */   887,  846,  744,  712,
    /*   398.1 */   693,  668,  637,  606,
    /*   501.2 */   580,  555,  529,  502,
    /*   631.0 */   475,  448,  422,  398,
    /*   794.3 */   375,  351,  327,  322,
    /*  1000.0 */   312,  301,  291,  268,
    /*  1258.9 */   246,  215,  182,  146,
    /*  1584.9 */   107,   61,   13,  -35,
    /*  1995.3 */   -96, -156, -179, -235,
    /*  2511.9 */  -295, -350, -401, -421,
    /*  3162.3 */  -446, -499, -532, -535,
    /*  3981.1 */  -513, -476, -431, -313,
    /*  5011.9 */  -179,    8,  203,  403,
    /*  6309.6 */   580,  736,  881, 1022,
    /*  7943.3 */  1154, 1251, 1348, 1421,
    /* 10000.0 */  1479, 1399, 1285, 1193,
    /* 12589.3 */  1287, 1519, 1914, 2369,
    /* 15848.9 */  3352, 4352, 5352, 6352,
    /* 19952.6 */  7352, 8352, 9352, 9999,
    /* 25118.9 */  9999, 9999, 9999, 9999,
];

/// Absolute Threshold of Hearing formula (Frank Klemm / Musepack).
/// Input: frequency in Hz. Output: threshold in dB SPL.
fn ath_formula_frank(freq: f32) -> f32 {
    let freq = freq.clamp(10.0, 29853.0) as f64;
    let freq_log = 40.0 * (0.1 * freq).log10();
    let index = freq_log as usize;
    let frac = freq_log - index as f64;
    (0.01 * (ATH_TAB[index] as f64 * (1.0 - frac) + ATH_TAB[index + 1] as f64 * frac)) as f32
}

/// Calculate per-bin ATH threshold.
/// `len`: number of frequency bins, `sample_rate`: Hz.
pub fn calc_ath(len: usize, sample_rate: usize) -> Vec<f32> {
    let mf = sample_rate as f32 / 2000.0;
    (0..len)
        .map(|i| {
            let f = (i + 1) as f32 * mf / len as f32; // frequency in kHz
            let trh = ath_formula_frank(1.0e3 * f) - 100.0;
            trh - f * f * 0.015
        })
        .collect()
}

/// Create loudness weighting curve for a given number of bins.
/// Matches C++ CreateLoudnessCurve.
pub fn create_loudness_curve(sz: usize) -> Vec<f32> {
    (0..sz)
        .map(|i| {
            let f = (i + 3) as f32 * 0.5 * 44100.0 / sz as f32;
            let t = f.log10() - 3.5;
            let t = -10.0 * t * t + 3.0 - f / 3000.0;
            10.0_f32.powf(0.1 * t)
        })
        .collect()
}

/// Analyze scale factor spread: returns 0.0 (noise-like) to 1.0 (tone-like).
/// sigma = stddev of scale factor indices, clamped to [0, 14], divided by 14.
pub fn analyze_scale_factor_spread(scale_factor_indices: &[u8]) -> f32 {
    let n = scale_factor_indices.len() as f32;
    let mean: f32 = scale_factor_indices.iter().map(|&x| x as f32).sum::<f32>() / n;

    let sigma: f32 = scale_factor_indices
        .iter()
        .map(|&x| {
            let t = x as f32 - mean;
            t * t
        })
        .sum::<f32>()
        / n;
    let sigma = sigma.sqrt().min(14.0);
    sigma / 14.0
}

/// Track loudness (stereo): exponential moving average.
pub fn track_loudness_stereo(prev: f32, l0: f32, l1: f32) -> f32 {
    0.98 * prev + 0.01 * (l0 + l1)
}

/// Track loudness (mono): exponential moving average.
pub fn track_loudness_mono(prev: f32, l: f32) -> f32 {
    0.98 * prev + 0.02 * l
}

#[cfg(test)]
#[path = "tests/psychoacoustic_tests.rs"]
mod tests;
