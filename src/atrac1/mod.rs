use std::sync::LazyLock;

use crate::bitstream::BitStream;

pub const MAX_BFUS: usize = 52;
pub const NUM_QMF: usize = 3;
pub const NUM_SAMPLES: usize = 512;
pub const SOUND_UNIT_SIZE: usize = 212;
pub const BITS_PER_BFU_AMOUNT_TAB_IDX: usize = 3;
pub const BITS_PER_IDWL: usize = 4;
pub const BITS_PER_IDSF: usize = 6;

pub const SPECS_PER_BLOCK: [u32; MAX_BFUS] = [
    // low band (20 BFUs)
    8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6,
    // mid band (16 BFUs)
    6, 6, 6, 6, 7, 7, 7, 7, 9, 9, 9, 9, 10, 10, 10, 10,
    // high band (16 BFUs)
    12, 12, 12, 12, 12, 12, 12, 12, 20, 20, 20, 20, 20, 20, 20, 20,
];

pub const BLOCKS_PER_BAND: [u32; NUM_QMF + 1] = [0, 20, 36, 52];

pub const SPECS_START_LONG: [u32; MAX_BFUS] = [
    0, 8, 16, 24, 32, 36, 40, 44, 48, 56, 64, 72, 80, 86, 92, 98, 104, 110, 116, 122,
    128, 134, 140, 146, 152, 159, 166, 173, 180, 189, 198, 207, 216, 226, 236, 246,
    256, 268, 280, 292, 304, 316, 328, 340, 352, 372, 392, 412, 432, 452, 472, 492,
];

pub const SPECS_START_SHORT: [u32; MAX_BFUS] = [
    0, 32, 64, 96, 8, 40, 72, 104, 12, 44, 76, 108, 20, 52, 84, 116, 26, 58, 90, 122,
    128, 160, 192, 224, 134, 166, 198, 230, 141, 173, 205, 237, 150, 182, 214, 246,
    256, 288, 320, 352, 384, 416, 448, 480, 268, 300, 332, 364, 396, 428, 460, 492,
];

pub const BFU_AMOUNT_TAB: [u32; 8] = [20, 28, 32, 36, 40, 44, 48, 52];

/// ScaleTable[i] = 2^(i/3 - 21), computed in f64 then stored as f32.
pub static SCALE_TABLE: LazyLock<[f32; 64]> = LazyLock::new(|| {
    let mut table = [0.0f32; 64];
    for i in 0..64 {
        table[i] = 2.0_f64.powf(i as f64 / 3.0 - 21.0) as f32;
    }
    table
});

/// SineWindow[i] = sin((i + 0.5) * PI / 64), computed in f64 then stored as f32.
pub static SINE_WINDOW: LazyLock<[f32; 32]> = LazyLock::new(|| {
    let mut window = [0.0f32; 32];
    for i in 0..32 {
        window[i] = ((i as f64 + 0.5) * (std::f64::consts::PI / 64.0)).sin() as f32;
    }
    window
});

/// Map BFU index to band number (0=low, 1=mid, 2=high).
pub fn bfu_to_band(i: u32) -> u32 {
    if i < 20 {
        0
    } else if i < 36 {
        1
    } else {
        2
    }
}

/// Block size mode: controls long vs short windows per band.
/// LogCount: [low, mid, high] — 0 = long block, 2 = short (low/mid), 3 = short (high).
#[derive(Debug, Clone, Default)]
pub struct BlockSizeMod {
    pub log_count: [i32; 3],
}

impl BlockSizeMod {
    pub fn new() -> Self {
        Self { log_count: [0, 0, 0] }
    }

    /// Parse from bitstream: reads 8 bits (2 per band + 2 unused).
    /// Matches C++: tmp[0] = 2 - Read(2), tmp[1] = 2 - Read(2), tmp[2] = 3 - Read(2), skip 2.
    pub fn from_bitstream(stream: &mut BitStream) -> Self {
        let low = 2 - stream.read(2) as i32;
        let mid = 2 - stream.read(2) as i32;
        let hi = 3 - stream.read(2) as i32;
        stream.read(2); // skip unused 2 bits
        Self { log_count: [low, mid, hi] }
    }

    /// Create from boolean flags (true = short window).
    /// Matches C++: low/mid short -> 2, hi short -> 3.
    pub fn from_flags(low_short: bool, mid_short: bool, hi_short: bool) -> Self {
        Self {
            log_count: [
                if low_short { 2 } else { 0 },
                if mid_short { 2 } else { 0 },
                if hi_short { 3 } else { 0 },
            ],
        }
    }

    /// Returns true if the given band uses short windows.
    pub fn short_win(&self, band: usize) -> bool {
        self.log_count[band] != 0
    }
}

/// Encoder settings for ATRAC1.
#[derive(Debug, Clone)]
pub struct Atrac1EncodeSettings {
    pub bfu_idx_const: u32,
    pub fast_bfu_num_search: bool,
    pub window_mode: WindowMode,
    pub window_mask: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowMode {
    NoTransient,
    Auto,
}

impl Default for Atrac1EncodeSettings {
    fn default() -> Self {
        Self {
            bfu_idx_const: 0,
            fast_bfu_num_search: false,
            window_mode: WindowMode::Auto,
            window_mask: 0,
        }
    }
}

pub mod qmf;

#[cfg(test)]
#[path = "../tests/atrac1_tests.rs"]
mod tests;
