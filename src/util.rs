/// Reverse array in-place. Matches C++ SwapArray.
pub fn swap_array<T>(p: &mut [T]) {
    p.reverse();
}

/// Negate every even-indexed element in-place. Matches C++ InvertSpectrInPlase.
pub fn invert_spectr_in_place(data: &mut [f32]) {
    for i in (0..data.len()).step_by(2) {
        data[i] = -data[i];
    }
}

/// Return a copy with every even-indexed element negated. Matches C++ InvertSpectr.
pub fn invert_spectr(data: &[f32]) -> Vec<f32> {
    let mut buf = data.to_vec();
    invert_spectr_in_place(&mut buf);
    buf
}

/// Find position of the highest set bit using De Bruijn sequence.
/// Matches C++ GetFirstSetBit exactly.
pub fn get_first_set_bit(mut x: u32) -> u16 {
    const MULTIPLY_DE_BRUIJN_BIT_POSITION: [u16; 32] = [
        0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30, 8, 12, 20, 28, 15, 17, 24, 7,
        19, 27, 23, 6, 26, 5, 4, 31,
    ];
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    MULTIPLY_DE_BRUIJN_BIT_POSITION[(x.wrapping_mul(0x07C4ACDD) >> 27) as usize]
}

/// Ceiling division by 8. Matches C++ Div8Ceil.
pub fn div8_ceil(input: u32) -> u32 {
    1 + (input - 1) / 8
}

/// Calculate median of a slice. Matches C++ CalcMedian.
pub fn calc_median<T: Clone + Ord>(data: &[T]) -> T {
    let mut tmp = data.to_vec();
    tmp.sort();
    let pos = (tmp.len() - 1) / 2;
    tmp[pos].clone()
}

/// Sum of squares. Matches C++ CalcEnergy.
pub fn calc_energy(data: &[f32]) -> f32 {
    data.iter().map(|&x| x * x).sum()
}

/// Float to int with banker's rounding (round-half-to-even).
/// Matches C++ `lrint(x)` under FE_TONEAREST.
/// CRITICAL: Do NOT use f32::round() which is round-half-away-from-zero.
pub fn to_int(x: f32) -> i32 {
    x.round_ties_even() as i32
}

#[cfg(test)]
#[path = "tests/util_tests.rs"]
mod tests;
