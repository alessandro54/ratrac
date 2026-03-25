//! Bit-level I/O for ATRAC1 bitstreams.
//!
//! ATRAC1 frames pack fields at arbitrary bit widths (3-bit BFU count,
//! 4-bit word lengths, 6-bit scale factors, etc.). This module handles
//! reading and writing individual bit fields into a byte buffer.
//!
//! ## Bit packing model
//!
//! Bits are packed MSB-first (big-endian) within each byte:
//! ```text
//! Byte:    [7 6 5 4 3 2 1 0] [7 6 5 4 3 2 1 0] ...
//! Write:    ←────────────────  ←────────────────
//! First write goes into bit 7 of byte 0, then bit 6, etc.
//! ```
//!
//! Maximum 23 bits per read/write operation.

/// Maximum bits per single read/write call.
const MAX_BITS_PER_OP: usize = 23;

/// Sign-extend a value from `bits` width to a full i32.
///
/// Example: `make_sign(0b110, 3)` → `-2` (the 3-bit two's complement of 6).
///
/// This matches the C++ `MakeSign()` function used in the dequantiser
/// to reconstruct signed mantissa values from unsigned bitstream reads.
pub fn make_sign(val: i32, bits: u32) -> i32 {
    let shift = 32 - bits;
    ((val as u32) << shift) as i32 >> shift
}

/// Bit-level reader/writer for ATRAC1 bitstreams.
///
/// Supports both writing (encoding) and reading (decoding) of variable-width
/// bit fields. The internal buffer grows automatically during writes.
pub struct BitStream {
    buf: Vec<u8>,
    /// Total bits written (write cursor).
    bits_used: usize,
    /// Current read position in bits (read cursor).
    read_pos: usize,
}

impl BitStream {
    /// Create an empty bitstream for writing.
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            bits_used: 0,
            read_pos: 0,
        }
    }

    /// Create a bitstream from existing bytes for reading.
    pub fn from_bytes(buf: &[u8]) -> Self {
        Self {
            buf: buf.to_vec(),
            bits_used: buf.len() * 8,
            read_pos: 0,
        }
    }

    /// Write `n` bits of `val` into the bitstream (MSB-first packing).
    ///
    /// # Panics
    /// Panics if `n > 23`.
    pub fn write(&mut self, val: u32, n: usize) {
        assert!(
            n <= MAX_BITS_PER_OP,
            "BitStream::write: n must be <= {MAX_BITS_PER_OP}, got {n}"
        );

        let bits_left = (self.buf.len() * 8).saturating_sub(self.bits_used) as i32;
        let bits_req = n as i32 - bits_left;
        let bytes_pos = self.bits_used / 8;
        let overlap = self.bits_used % 8;

        // Grow buffer if needed
        if overlap > 0 || bits_req >= 0 {
            let extra = bits_req / 8 + if overlap > 0 { 2 } else { 1 };
            self.buf.resize(self.buf.len() + extra as usize, 0);
        }

        // Pack val into big-endian bytes, shifted to align with current bit position
        let packed: u32 = (val << (32 - n)) >> overlap;
        let packed_bytes = packed.to_be_bytes();

        let count = n / 8 + if overlap > 0 { 2 } else { 1 };
        for i in 0..count {
            self.buf[bytes_pos + i] |= packed_bytes[i];
        }

        self.bits_used += n;
    }

    /// Read `n` bits from the bitstream, advancing the read cursor.
    ///
    /// # Panics
    /// Panics if `n > 23`.
    pub fn read(&mut self, n: usize) -> u32 {
        assert!(
            n <= MAX_BITS_PER_OP,
            "BitStream::read: n must be <= {MAX_BITS_PER_OP}, got {n}"
        );

        let bytes_pos = self.read_pos / 8;
        let overlap = self.read_pos % 8;

        // Load bytes in big-endian order into a u32
        let count = n / 8 + if overlap > 0 { 2 } else { 1 };
        let mut be_bytes = [0u8; 4];
        for i in 0..count {
            be_bytes[3 - i] = self.buf[bytes_pos + i];
        }
        let mut val = u32::from_le_bytes(be_bytes);

        // Shift out the overlap bits, then right-justify the result
        val = (val << overlap) >> (32 - n);

        self.read_pos += n;
        val
    }

    /// Total bits written so far.
    pub fn size_in_bits(&self) -> usize {
        self.bits_used
    }

    /// Size of the underlying buffer in bytes.
    pub fn buf_size(&self) -> usize {
        self.buf.len()
    }

    /// Read-only access to the raw byte buffer.
    pub fn get_bytes(&self) -> &[u8] {
        &self.buf
    }

    /// Reset the read cursor to the beginning.
    pub fn reset_read_pos(&mut self) {
        self.read_pos = 0;
    }
}

impl Default for BitStream {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "tests/bitstream_tests.rs"]
mod tests;
