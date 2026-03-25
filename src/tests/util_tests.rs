use super::*;

// --- Translated from util_ut.cpp ---

#[test]
fn test_swap_array() {
    let mut arr: [f32; 8] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    swap_array(&mut arr);
    for i in 0..8 {
        assert!(
            (i as f32 - arr[7 - i]).abs() < 1e-12,
            "swap_array mismatch at index {i}"
        );
    }
}

#[test]
fn test_get_first_set_bit() {
    assert_eq!(1, get_first_set_bit(2));
    assert_eq!(1, get_first_set_bit(3));
    assert_eq!(2, get_first_set_bit(4));
    assert_eq!(2, get_first_set_bit(5));
    assert_eq!(2, get_first_set_bit(6));
    assert_eq!(2, get_first_set_bit(7));
    assert_eq!(3, get_first_set_bit(8));
    assert_eq!(3, get_first_set_bit(9));
    assert_eq!(3, get_first_set_bit(10));
}

#[test]
fn test_calc_energy() {
    assert!((0.0 - calc_energy(&[0.0])).abs() < 1e-12);
    assert!((1.0 - calc_energy(&[1.0])).abs() < 1e-12);
    assert!((2.0 - calc_energy(&[1.0, 1.0])).abs() < 1e-12);
    assert!((5.0 - calc_energy(&[2.0, 1.0])).abs() < 1e-12);
    assert!((5.0 - calc_energy(&[1.0, 2.0])).abs() < 1e-12);
    assert!((8.0 - calc_energy(&[2.0, 2.0])).abs() < 1e-12);
}

// --- Additional tests ---

#[test]
fn test_swap_array_odd_length() {
    let mut arr = [1, 2, 3, 4, 5];
    swap_array(&mut arr);
    assert_eq!([5, 4, 3, 2, 1], arr);
}

#[test]
fn test_swap_array_single() {
    let mut arr = [42];
    swap_array(&mut arr);
    assert_eq!([42], arr);
}

#[test]
fn test_swap_array_empty() {
    let mut arr: [i32; 0] = [];
    swap_array(&mut arr);
    assert_eq!(0, arr.len());
}

#[test]
fn test_invert_spectr_in_place() {
    let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    invert_spectr_in_place(&mut data);
    assert_eq!([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0], data);
}

#[test]
fn test_invert_spectr() {
    let data = [1.0, 2.0, 3.0, 4.0];
    let result = invert_spectr(&data);
    assert_eq!(vec![-1.0, 2.0, -3.0, 4.0], result);
    // Original unchanged
    assert_eq!([1.0, 2.0, 3.0, 4.0], data);
}

#[test]
fn test_invert_spectr_single() {
    let mut data = [5.0];
    invert_spectr_in_place(&mut data);
    assert_eq!([-5.0], data);
}

#[test]
fn test_get_first_set_bit_powers_of_two() {
    assert_eq!(0, get_first_set_bit(1));
    assert_eq!(1, get_first_set_bit(2));
    assert_eq!(2, get_first_set_bit(4));
    assert_eq!(3, get_first_set_bit(8));
    assert_eq!(4, get_first_set_bit(16));
    assert_eq!(8, get_first_set_bit(256));
    assert_eq!(16, get_first_set_bit(65536));
}

#[test]
fn test_get_first_set_bit_large() {
    assert_eq!(31, get_first_set_bit(0xFFFFFFFF));
    assert_eq!(31, get_first_set_bit(0x80000000));
}

#[test]
fn test_div8_ceil() {
    assert_eq!(1, div8_ceil(1));
    assert_eq!(1, div8_ceil(8));
    assert_eq!(2, div8_ceil(9));
    assert_eq!(2, div8_ceil(16));
    assert_eq!(3, div8_ceil(17));
}

#[test]
fn test_calc_median() {
    assert_eq!(2, calc_median(&[1, 2, 3]));
    assert_eq!(2, calc_median(&[3, 1, 2]));
    assert_eq!(2, calc_median(&[1, 2, 3, 4])); // (4-1)/2 = 1 -> index 1 = 2
    assert_eq!(5, calc_median(&[5]));
}

// --- to_int: banker's rounding (CRITICAL for fidelity) ---

#[test]
fn test_to_int_basic() {
    assert_eq!(1, to_int(1.0));
    assert_eq!(-1, to_int(-1.0));
    assert_eq!(0, to_int(0.0));
    assert_eq!(3, to_int(3.3));
    assert_eq!(-3, to_int(-3.3));
}

#[test]
fn test_to_int_rounds_up_normally() {
    assert_eq!(1, to_int(0.6));
    assert_eq!(2, to_int(1.7));
    assert_eq!(-1, to_int(-0.6));
    assert_eq!(-2, to_int(-1.7));
}

#[test]
fn test_to_int_bankers_rounding_half_values() {
    // Banker's rounding: half values round to nearest EVEN
    assert_eq!(0, to_int(0.5)); // 0.5 -> 0 (even)
    assert_eq!(2, to_int(1.5)); // 1.5 -> 2 (even)
    assert_eq!(2, to_int(2.5)); // 2.5 -> 2 (even)
    assert_eq!(4, to_int(3.5)); // 3.5 -> 4 (even)
    assert_eq!(4, to_int(4.5)); // 4.5 -> 4 (even)

    // Negative half values
    assert_eq!(0, to_int(-0.5)); // -0.5 -> 0 (even)
    assert_eq!(-2, to_int(-1.5)); // -1.5 -> -2 (even)
    assert_eq!(-2, to_int(-2.5)); // -2.5 -> -2 (even)
    assert_eq!(-4, to_int(-3.5)); // -3.5 -> -4 (even)
}

#[test]
fn test_to_int_not_round_half_away_from_zero() {
    // Verify we are NOT using round-half-away-from-zero
    // f32::round() would give 1 for 0.5, but banker's gives 0
    assert_ne!(1, to_int(0.5));
    assert_ne!(3, to_int(2.5));
}

#[test]
fn test_calc_energy_empty() {
    assert_eq!(0.0, calc_energy(&[]));
}
