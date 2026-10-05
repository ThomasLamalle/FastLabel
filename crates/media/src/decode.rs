//! Image decoding to RGB8.

use std::path::Path;

use image::RgbImage;
use zune_jpeg::JpegDecoder;
use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("cannot read {0}: {1}")]
    Io(String, std::io::Error),
    #[error("cannot decode {0}: {1}")]
    Decode(String, String),
}

fn is_jpeg(bytes: &[u8]) -> bool {
    bytes.starts_with(&[0xFF, 0xD8, 0xFF])
}

/// Decodes any supported image to RGB8 (JPEG through zune-jpeg, the rest
/// through the `image` crate).
pub fn decode_rgb(path: &Path) -> Result<RgbImage, DecodeError> {
    let name = || path.display().to_string();
    let bytes = std::fs::read(path).map_err(|e| DecodeError::Io(name(), e))?;
    decode_bytes(&bytes).map_err(|e| DecodeError::Decode(name(), e))
}

pub fn decode_bytes(bytes: &[u8]) -> Result<RgbImage, String> {
    if is_jpeg(bytes) {
        let options = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
        let mut decoder = JpegDecoder::new_with_options(ZCursor::new(bytes), options);
        let pixels = decoder.decode().map_err(|e| e.to_string())?;
        let (w, h) = decoder.dimensions().ok_or("missing JPEG dimensions")?;
        RgbImage::from_raw(w as u32, h as u32, pixels)
            .ok_or_else(|| "JPEG buffer size mismatch".into())
    } else {
        image::load_from_memory(bytes).map(|i| i.to_rgb8()).map_err(|e| e.to_string())
    }
}

/// Encodes a small RGB image as JPEG (thumbnail cache format).
pub fn encode_jpeg(img: &RgbImage, quality: u8) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality)
        .encode_image(img)
        .map_err(|e| e.to_string())?;
    Ok(out)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Deterministic test image: R = x, G = y, B = constant.
    pub fn gradient(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, y| image::Rgb([(x % 256) as u8, (y % 256) as u8, 128]))
    }

    #[test]
    fn jpeg_round_trip_is_close() {
        let img = gradient(64, 48);
        let bytes = encode_jpeg(&img, 95).unwrap();
        let back = decode_bytes(&bytes).unwrap();
        assert_eq!(back.dimensions(), (64, 48));
        let p = back.get_pixel(30, 20);
        assert!((i32::from(p[0]) - 30).abs() < 6 && (i32::from(p[1]) - 20).abs() < 6, "{p:?}");
    }

    #[test]
    fn decodes_png_and_reports_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.png");
        gradient(10, 5).save(&path).unwrap();
        assert_eq!(decode_rgb(&path).unwrap().dimensions(), (10, 5));
        assert!(matches!(decode_rgb(&dir.path().join("missing.jpg")), Err(DecodeError::Io(..))));
        std::fs::write(dir.path().join("bad.jpg"), [0xFF, 0xD8, 0xFF, 0, 1, 2]).unwrap();
        assert!(matches!(decode_rgb(&dir.path().join("bad.jpg")), Err(DecodeError::Decode(..))));
    }
}
