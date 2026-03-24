use super::*;
use std::path::PathBuf;

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("ratrac_test_{name}.aea"))
}

#[test]
fn test_write_and_read_header() {
    let path = temp_path("header");
    let title = "TestTitle";
    let num_channels = 2;
    let num_frames = 100;

    {
        let mut writer = AeaWriter::create(&path, title, num_channels, num_frames).unwrap();
        writer.flush().unwrap();
    }

    let reader = AeaReader::open(&path).unwrap();
    assert_eq!(num_channels, reader.channel_num());
    assert_eq!(title, reader.name());
    assert_eq!(num_frames, reader.num_frames());

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_write_read_frames_roundtrip() {
    let path = temp_path("roundtrip");
    let num_frames = 5;

    // Write
    {
        let mut writer = AeaWriter::create(&path, "test", 1, num_frames).unwrap();

        // First write is skipped (dummy)
        writer.write_frame(&[0xAA; AEA_FRAME_SIZE]).unwrap();

        // These frames are actually written
        for i in 0..num_frames {
            let mut frame = vec![0u8; AEA_FRAME_SIZE];
            frame[0] = i as u8;
            frame[211] = (i + 100) as u8;
            writer.write_frame(&frame).unwrap();
        }
        writer.flush().unwrap();
    }

    // Read
    {
        let mut reader = AeaReader::open(&path).unwrap();

        // First frame in file is the dummy (all zeros)
        let dummy = reader.read_frame().unwrap().unwrap();
        assert_eq!(0, dummy[0]);

        // Read back our written frames
        for i in 0..num_frames {
            let frame = reader.read_frame().unwrap().unwrap();
            assert_eq!(i as u8, frame[0], "Frame {i} first byte mismatch");
            assert_eq!((i + 100) as u8, frame[211], "Frame {i} last byte mismatch");
        }
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_first_write_skipped() {
    let path = temp_path("first_skip");

    {
        let mut writer = AeaWriter::create(&path, "test", 1, 0).unwrap();
        // This write should be silently skipped
        writer.write_frame(&[0xFF; AEA_FRAME_SIZE]).unwrap();
        // This write should be actually written
        writer.write_frame(&[0x42; AEA_FRAME_SIZE]).unwrap();
        writer.flush().unwrap();
    }

    {
        let mut reader = AeaReader::open(&path).unwrap();
        // First frame is the dummy (zeros from header creation)
        let dummy = reader.read_frame().unwrap().unwrap();
        assert_eq!(0, dummy[0]);

        // Second frame should be 0x42 (not 0xFF)
        let frame = reader.read_frame().unwrap().unwrap();
        assert_eq!(0x42, frame[0]);
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_magic_validation() {
    let path = temp_path("bad_magic");

    // Write a file with bad magic
    {
        let mut file = File::create(&path).unwrap();
        let mut header = [0u8; AEA_META_SIZE];
        header[0] = 0xFF; // Wrong magic
        file.write_all(&header).unwrap();
    }

    let result = AeaReader::open(&path);
    assert!(matches!(result, Err(AeaError::InvalidFormat)));

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_channel_validation() {
    let path = temp_path("bad_channels");

    // Write a file with invalid channel count
    {
        let mut file = File::create(&path).unwrap();
        let mut header = [0u8; AEA_META_SIZE];
        header[0..4].copy_from_slice(&AEA_MAGIC);
        header[264] = 5; // Invalid: >= 3
        file.write_all(&header).unwrap();
    }

    let result = AeaReader::open(&path);
    assert!(matches!(result, Err(AeaError::InvalidChannels)));

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_length_in_samples() {
    let path = temp_path("length");

    // Write header + dummy + 10 frames for 1 channel
    {
        let mut writer = AeaWriter::create(&path, "test", 1, 10).unwrap();
        writer.write_frame(&[0; AEA_FRAME_SIZE]).unwrap(); // skipped (first write)
        for _ in 0..10 {
            writer.write_frame(&[0; AEA_FRAME_SIZE]).unwrap();
        }
        writer.flush().unwrap();
    }

    let reader = AeaReader::open(&path).unwrap();
    // file_size = 2048 + 212 * (1 dummy + 10 frames) = 2048 + 2332 = 4380
    // total_frames = (4380 - 2048) / 212 / 1 = 2332 / 212 = 11
    // samples = 512 * (11 - 5) = 512 * 6 = 3072
    assert_eq!(3072, reader.length_in_samples());

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_length_in_samples_stereo() {
    let path = temp_path("length_stereo");

    {
        let mut writer = AeaWriter::create(&path, "test", 2, 20).unwrap();
        writer.write_frame(&[0; AEA_FRAME_SIZE]).unwrap(); // skipped
        for _ in 0..20 {
            writer.write_frame(&[0; AEA_FRAME_SIZE]).unwrap();
        }
        writer.flush().unwrap();
    }

    let reader = AeaReader::open(&path).unwrap();
    // file_size = 2048 + 212 * 21 = 2048 + 4452 = 6500
    // total_frames = (6500 - 2048) / 212 / 2 = 4452 / 212 / 2 = 21 / 2 = 10
    // samples = 512 * (10 - 5) = 512 * 5 = 2560
    assert_eq!(2560, reader.length_in_samples());

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_frame_padding() {
    let path = temp_path("padding");

    {
        let mut writer = AeaWriter::create(&path, "test", 1, 1).unwrap();
        writer.write_frame(&[0; AEA_FRAME_SIZE]).unwrap(); // skipped
        // Write a short frame — should be zero-padded to 212
        writer.write_frame(&[0xAB; 10]).unwrap();
        writer.flush().unwrap();
    }

    {
        let mut reader = AeaReader::open(&path).unwrap();
        let _dummy = reader.read_frame().unwrap().unwrap();
        let frame = reader.read_frame().unwrap().unwrap();
        assert_eq!(AEA_FRAME_SIZE, frame.len());
        // First 10 bytes should be 0xAB
        for i in 0..10 {
            assert_eq!(0xAB, frame[i], "frame[{i}] should be 0xAB");
        }
        // Rest should be zero-padded
        for i in 10..AEA_FRAME_SIZE {
            assert_eq!(0, frame[i], "frame[{i}] should be zero-padded");
        }
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_read_frame_eof() {
    let path = temp_path("eof");

    {
        let mut writer = AeaWriter::create(&path, "test", 1, 0).unwrap();
        writer.flush().unwrap();
    }

    {
        let mut reader = AeaReader::open(&path).unwrap();
        // Dummy frame
        let frame = reader.read_frame().unwrap().unwrap();
        assert_eq!(AEA_FRAME_SIZE, frame.len());

        // Next read should be EOF
        let result = reader.read_frame().unwrap();
        assert!(result.is_none(), "Should return None at EOF");
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_title_truncation() {
    let path = temp_path("long_title");
    let long_title = "This Is A Very Long Title That Exceeds Sixteen";

    {
        let mut writer = AeaWriter::create(&path, long_title, 1, 0).unwrap();
        writer.flush().unwrap();
    }

    let reader = AeaReader::open(&path).unwrap();
    // Title should be truncated to 15 chars (null at byte 19)
    assert!(reader.name().len() <= 15);

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_mono_channel_zero() {
    let path = temp_path("mono_zero");

    // Header with channel byte = 0 should be treated as 1 channel
    {
        let mut file = File::create(&path).unwrap();
        let mut header = [0u8; AEA_META_SIZE];
        header[0..4].copy_from_slice(&AEA_MAGIC);
        header[264] = 0; // Channel 0 -> treated as 1
        file.write_all(&header).unwrap();
    }

    let reader = AeaReader::open(&path).unwrap();
    assert_eq!(1, reader.channel_num());

    std::fs::remove_file(&path).ok();
}
