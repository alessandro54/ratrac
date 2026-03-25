use super::*;

// Translated from bitstream_ut.cpp

#[test]
fn test_default_constructor() {
    let bs = BitStream::new();
    assert_eq!(0, bs.size_in_bits());
}

#[test]
fn test_simple_write_read() {
    let mut bs = BitStream::new();
    bs.write(5, 3);
    bs.write(1, 1); // true = 1
    assert_eq!(4, bs.size_in_bits());
    assert_eq!(5, bs.read(3));
    assert_eq!(1, bs.read(1));
}

#[test]
fn test_overlap_write_read() {
    let mut bs = BitStream::new();
    bs.write(101, 22);
    assert_eq!(22, bs.size_in_bits());

    bs.write(212, 22);
    assert_eq!(44, bs.size_in_bits());

    bs.write(323, 22);
    assert_eq!(66, bs.size_in_bits());

    assert_eq!(101, bs.read(22));
    assert_eq!(212, bs.read(22));
    assert_eq!(323, bs.read(22));
}

#[test]
fn test_overlap_write_read_2() {
    let mut bs = BitStream::new();
    bs.write(2, 2);
    bs.write(7, 4);
    bs.write(10003, 16);

    assert_eq!(2, bs.read(2));
    assert_eq!(7, bs.read(4));
    assert_eq!(10003, bs.read(16));
}

#[test]
fn test_overlap_write_read_3() {
    let mut bs = BitStream::new();
    bs.write(40, 6);
    bs.write(3, 2);
    bs.write(0, 3);
    bs.write(0, 3);
    bs.write(0, 3);
    bs.write(0, 3);

    bs.write(3, 5);
    bs.write(1, 2);
    bs.write(1, 1);
    bs.write(1, 1);
    bs.write(1, 1);
    bs.write(1, 1);

    bs.write(0, 3);
    bs.write(4, 3);
    bs.write(35, 6);
    bs.write(25, 6);
    bs.write(3, 3);
    bs.write(32, 6);
    bs.write(29, 6);
    bs.write(3, 3);
    bs.write(36, 6);
    bs.write(49, 6);

    assert_eq!(40, bs.read(6));
    assert_eq!(3, bs.read(2));
    assert_eq!(0, bs.read(3));
    assert_eq!(0, bs.read(3));
    assert_eq!(0, bs.read(3));
    assert_eq!(0, bs.read(3));
    assert_eq!(3, bs.read(5));

    assert_eq!(1, bs.read(2));
    assert_eq!(1, bs.read(1));
    assert_eq!(1, bs.read(1));
    assert_eq!(1, bs.read(1));
    assert_eq!(1, bs.read(1));

    assert_eq!(0, bs.read(3));
    assert_eq!(4, bs.read(3));
    assert_eq!(35, bs.read(6));
    assert_eq!(25, bs.read(6));
    assert_eq!(3, bs.read(3));
    assert_eq!(32, bs.read(6));
    assert_eq!(29, bs.read(6));
    assert_eq!(3, bs.read(3));
    assert_eq!(36, bs.read(6));
    assert_eq!(49, bs.read(6));
}

#[test]
fn test_sign_write_read() {
    let mut bs = BitStream::new();
    bs.write(make_sign(-2, 3) as u32, 3);
    bs.write(make_sign(-1, 3) as u32, 3);
    bs.write(make_sign(1, 2) as u32, 2);
    bs.write(make_sign(-7, 4) as u32, 4);

    assert_eq!(-2, make_sign(bs.read(3) as i32, 3));
    assert_eq!(-1, make_sign(bs.read(3) as i32, 3));
    assert_eq!(1, make_sign(bs.read(2) as i32, 2));
    assert_eq!(-7, make_sign(bs.read(4) as i32, 4));
}

// Additional tests beyond C++ suite

#[test]
fn test_make_sign_values() {
    assert_eq!(-2, make_sign(-2i32 as u32 as i32, 3));
    assert_eq!(-1, make_sign(-1i32 as u32 as i32, 3));
    assert_eq!(1, make_sign(1, 2));
    assert_eq!(-7, make_sign(-7i32 as u32 as i32, 4));

    // Edge cases
    assert_eq!(-1, make_sign(0b111_i32, 3)); // 7 as 3 bits = -1
    assert_eq!(3, make_sign(3, 3)); // 011 = 3 (positive)
    assert_eq!(-4, make_sign(0b100_i32, 3)); // 100 = -4
}

#[test]
fn test_write_read_single_bit() {
    let mut bs = BitStream::new();
    for i in 0..8u32 {
        bs.write(i & 1, 1);
    }
    for i in 0..8u32 {
        assert_eq!(i & 1, bs.read(1));
    }
}

#[test]
fn test_write_read_max_bits() {
    let mut bs = BitStream::new();
    let val = (1u32 << 23) - 1;
    bs.write(val, 23);
    assert_eq!(val, bs.read(23));
}

#[test]
fn test_from_bytes() {
    let mut bs_write = BitStream::new();
    bs_write.write(5, 3);
    bs_write.write(1, 1);
    bs_write.write(10, 6);
    let bytes = bs_write.get_bytes().to_vec();

    let mut bs_read = BitStream::from_bytes(&bytes);
    assert_eq!(5, bs_read.read(3));
    assert_eq!(1, bs_read.read(1));
    assert_eq!(10, bs_read.read(6));
}
