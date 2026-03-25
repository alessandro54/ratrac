use crate::atrac1::{BlockSizeMod, NUM_QMF, SINE_WINDOW};
use crate::mdct::{Mdct, Midct};
use crate::util::swap_array;

/// Windowed overlap function matching C++ vector_fmul_window.
///
/// C++ does: dst+=len, win+=len, src0+=len, then loops i=-len..-1, j=len-1..0
/// After pointer offsets with len=16 and SineWindow[32]:
///   win[i] (i=-16..-1) → SineWindow[0..15]
///   win[j] (j=15..0)   → SineWindow[16..31] (reversed)
///   src0[i] (i=-16..-1) → prevBuf[0..15]
///   src1[j] (j=15..0)   → invBuf[start+15..start] (reversed)
///   dst[i] → dstBuf[start..start+15]
///   dst[j] → dstBuf[start+16..start+31]
fn vector_fmul_window(
    dst: &mut [f32],
    dst_offset: usize,
    src0: &[f32],
    src0_offset: usize,
    src1: &[f32],
    src1_offset: usize,
    win: &[f32],
    len: usize,
) {
    for k in 0..len {
        let i = k;
        let j = len - 1 - k;
        let s0 = src0[src0_offset + i];
        let s1 = src1[src1_offset + j];
        // After win+=len: win[i-len] = win_orig[i], win[j] = win_orig[j+len]
        let wi = win[i];           // SineWindow[0..15]
        let wj = win[j + len];     // SineWindow[16..31]
        dst[dst_offset + i] = s0 * wj - s1 * wi;
        dst[dst_offset + j + len] = s0 * wi + s1 * wj;
    }
}

/// ATRAC1 MDCT wrapper: manages 6 transform instances and handles
/// windowing, band reordering, and short-block compensation.
pub struct Atrac1Mdct {
    mdct512: Mdct,
    mdct256: Mdct,
    mdct64: Mdct,
    midct512: Midct,
    midct256: Midct,
    midct64: Midct,
}

impl Atrac1Mdct {
    pub fn new() -> Self {
        Self {
            mdct512: Mdct::new(512, 1.0),
            mdct256: Mdct::new(256, 0.5),
            mdct64: Mdct::new(64, 0.5),
            midct512: Midct::new(512, 512.0 * 2.0),
            midct256: Midct::new(256, 256.0 * 2.0),
            midct64: Midct::new(64, 64.0 * 2.0),
        }
    }

    /// Forward MDCT for all 3 bands.
    /// `specs`: output 512 spectral coefficients
    /// `low`, `mid`, `hi`: input time-domain band buffers (with overlap tail space)
    ///   low/mid: [0..256+16), hi: [0..512+16)
    ///   First 128/256 = current samples, rest = overlap from previous frame
    pub fn mdct(
        &mut self,
        specs: &mut [f32],
        low: &mut [f32],
        mid: &mut [f32],
        hi: &mut [f32],
        block_size: &BlockSizeMod,
    ) {
        let sine_window = &*SINE_WINDOW;
        let mut pos: usize = 0;

        for band in 0..NUM_QMF {
            let num_mdct_blocks = 1usize << block_size.log_count[band];
            let src_buf: &mut [f32] = match band {
                0 => low,
                1 => mid,
                _ => hi,
            };
            let buf_sz: usize = if band == 2 { 256 } else { 128 };
            let block_sz = if num_mdct_blocks == 1 { buf_sz } else { 32 };
            let win_start = if num_mdct_blocks == 1 {
                if band == 2 { 112 } else { 48 }
            } else {
                0
            };
            let multiple: f32 = if num_mdct_blocks != 1 && band == 2 { 2.0 } else { 1.0 };

            let mut tmp = vec![0.0f32; 512];
            let mut block_pos: usize = 0;

            for _k in 0..num_mdct_blocks {
                // Copy overlap tail into tmp
                tmp[win_start..win_start + 32]
                    .copy_from_slice(&src_buf[buf_sz..buf_sz + 32]);

                // Window and save new overlap
                for i in 0..32 {
                    src_buf[buf_sz + i] =
                        sine_window[i] * src_buf[block_pos + block_sz - 32 + i];
                    src_buf[block_pos + block_sz - 32 + i] =
                        sine_window[31 - i] * src_buf[block_pos + block_sz - 32 + i];
                }

                // Copy current block into tmp
                tmp[win_start + 32..win_start + 32 + block_sz]
                    .copy_from_slice(&src_buf[block_pos..block_pos + block_sz]);

                // Transform
                let sp = if num_mdct_blocks == 1 {
                    if band == 2 {
                        self.mdct512.process(&tmp)
                    } else {
                        self.mdct256.process(&tmp)
                    }
                } else {
                    self.mdct64.process(&tmp)
                };

                let sp_len = sp.len();
                specs[block_pos + pos..block_pos + pos + sp_len].copy_from_slice(sp);

                // Apply multiple
                if multiple != 1.0 {
                    for i in 0..sp_len {
                        specs[block_pos + pos + i] *= multiple;
                    }
                }

                // Swap for bands 1 and 2
                if band > 0 {
                    swap_array(&mut specs[block_pos + pos..block_pos + pos + sp_len]);
                }

                block_pos += 32;
            }
            pos += buf_sz;
        }
    }

    /// Inverse MDCT for all 3 bands.
    /// `specs`: input 512 spectral coefficients (modified in-place for band swap)
    /// `low`, `mid`, `hi`: output time-domain band buffers (with overlap tail space)
    pub fn imdct(
        &mut self,
        specs: &mut [f32],
        mode: &BlockSizeMod,
        low: &mut [f32],
        mid: &mut [f32],
        hi: &mut [f32],
    ) {
        let sine_window = &*SINE_WINDOW;
        let mut pos: usize = 0;

        for band in 0..NUM_QMF {
            let num_mdct_blocks = 1usize << mode.log_count[band];
            let buf_sz: usize = if band == 2 { 256 } else { 128 };
            let block_sz = if num_mdct_blocks == 1 { buf_sz } else { 32 };

            let dst_buf: &mut [f32] = match band {
                0 => low,
                1 => mid,
                _ => hi,
            };

            let mut inv_buf = vec![0.0f32; 512];
            let mut start: usize = 0;

            // prev_buf points into dst_buf at the overlap tail from previous frame
            // C++: float* prevBuf = &dstBuf[bufSz * 2 - 16]
            // We'll track the prev_buf as a range into dst_buf or inv_buf
            let mut prev_is_dst = true;
            let mut prev_offset = buf_sz * 2 - 16;

            for _block in 0..num_mdct_blocks {
                // Swap for bands 1 and 2
                if band > 0 {
                    swap_array(&mut specs[pos..pos + block_sz]);
                }

                // Inverse transform
                let inv = if num_mdct_blocks != 1 {
                    self.midct64.process(&specs[pos..pos + block_sz])
                } else if buf_sz == 128 {
                    self.midct256.process(&specs[pos..pos + block_sz])
                } else {
                    self.midct512.process(&specs[pos..pos + block_sz])
                };
                let inv_len = inv.len();

                // Copy middle portion of inverse into inv_buf
                for i in 0..inv_len / 2 {
                    inv_buf[start + i] = inv[i + inv_len / 4];
                }

                // vector_fmul_window: overlap-add with sine window
                if prev_is_dst {
                    // prev is in dst_buf at prev_offset
                    // We need to create a temporary copy since dst_buf is being written
                    let prev_copy: Vec<f32> =
                        dst_buf[prev_offset..prev_offset + 16].to_vec();
                    vector_fmul_window(
                        dst_buf, start, &prev_copy, 0, &inv_buf, start, sine_window, 16,
                    );
                } else {
                    // prev is in inv_buf at prev_offset
                    let prev_copy: Vec<f32> =
                        inv_buf[prev_offset..prev_offset + 16].to_vec();
                    vector_fmul_window(
                        dst_buf, start, &prev_copy, 0, &inv_buf, start, sine_window, 16,
                    );
                }

                // Update prev_buf to point into inv_buf
                prev_is_dst = false;
                prev_offset = start + 16;
                start += block_sz;
                pos += block_sz;
            }

            // Copy tail for long blocks
            if num_mdct_blocks == 1 {
                let tail_len = if band == 2 { 240 } else { 112 };
                dst_buf[32..32 + tail_len].copy_from_slice(&inv_buf[16..16 + tail_len]);
            }

            // Save overlap tail for next frame
            for j in 0..16 {
                dst_buf[buf_sz * 2 - 16 + j] = inv_buf[buf_sz - 16 + j];
            }
        }
    }
}

#[cfg(test)]
#[path = "../tests/atrac1_mdct_impl_tests.rs"]
mod tests;
