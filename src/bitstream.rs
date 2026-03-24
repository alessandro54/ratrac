/// Sign-extend a value from `bits` width to i32.
/// Matches C++ MakeSign: uses arithmetic shift to propagate sign bit.
pub fn make_sign(val: i32, bits: u32) -> i32 {
    let shift = 32 - bits;
    ((val as u32) << shift) as i32 >> shift
}

/// Bit-level reader/writer matching the C++ TBitStream implementation.
/// Packs bits big-endian within a little-endian byte buffer.
pub struct BitStream {
    buf: Vec<u8>,
    bits_used: usize,
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

    /// Write `n` bits (max 23) of `val` into the bitstream.
    pub fn write(&mut self, val: u32, n: usize) {
        assert!(n <= 23, "BitStream::write: n must be <= 23, got {n}");

        let bits_left = self.buf.len() * 8 - self.bits_used;
        let bits_left = bits_left as i32;
        let bits_req = n as i32 - bits_left;
        let bytes_pos = self.bits_used / 8;
        let overlap = self.bits_used % 8;

        if overlap > 0 || bits_req >= 0 {
            let extra = bits_req / 8 + if overlap > 0 { 2 } else { 1 };
            self.buf.resize(self.buf.len() + extra as usize, 0);
        }

        // Shift val to MSB, then right by overlap to align
        let t: u32 = (val << (32 - n)) >> overlap;
        let t_bytes = t.to_be_bytes(); // big-endian = [3-i] indexing on LE

        let count = n / 8 + if overlap > 0 { 2 } else { 1 };
        for i in 0..count {
            self.buf[bytes_pos + i] |= t_bytes[i];
        }

        self.bits_used += n;
    }

    /// Read `n` bits (max 23) from the bitstream.
    pub fn read(&mut self, n: usize) -> u32 {
        assert!(n <= 23, "BitStream::read: n must be <= 23, got {n}");

        let bytes_pos = self.read_pos / 8;
        let overlap = self.read_pos % 8;

        let mut t: u32 = 0;
        let count = n / 8 + if overlap > 0 { 2 } else { 1 };
        // Read bytes into u32 big-endian (matching C++ `t.bytes[3-i] = Buf[pos+i]`)
        let t_bytes: &mut [u8; 4] = unsafe { &mut *(&mut t as *mut u32 as *mut [u8; 4]) };
        for i in 0..count {
            // On LE CPU, C++ does t.bytes[3-i] = Buf[pos+i]
            // to_be_bytes()[i] == bytes[3-i] on LE, so we use be indexing
            t_bytes[3 - i] = self.buf[bytes_pos + i];
        }

        // Now t has the bytes loaded in big-endian layout on LE
        // C++ does: t.ui = (t.ui << overlap >> (32 - n))
        t = (t << overlap) >> (32 - n);

        self.read_pos += n;
        t
    }

    /// Number of bits written so far.
    pub fn size_in_bits(&self) -> usize {
        self.bits_used
    }

    /// Size of the underlying buffer in bytes.
    pub fn buf_size(&self) -> usize {
        self.buf.len()
    }

    /// Access the underlying byte buffer.
    pub fn get_bytes(&self) -> &[u8] {
        &self.buf
    }

    /// Reset read position to the beginning.
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
