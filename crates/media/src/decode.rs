//! Image decoding to RGB8.

use std::path::{Path, PathBuf};

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

/// Export resizer ([`fastlabel_core::export::ResizeFn`]): each image decoded,
/// resized to exactly `(w, h)` and re-encoded in its own format (JPEG at
/// quality 95), spread over the cores.
pub fn resize_batch(jobs: &[(PathBuf, u32, u32)]) -> Vec<Result<Vec<u8>, String>> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let per = jobs.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        let handles: Vec<_> = jobs
            .chunks(per)
            .map(|c| {
                s.spawn(move || {
                    c.iter().map(|(p, w, h)| resize_file(p, *w, *h)).collect::<Vec<_>>()
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().expect("resize thread")).collect()
    })
}

fn resize_file(path: &Path, w: u32, h: u32) -> Result<Vec<u8>, String> {
    use fast_image_resize::{PixelType, Resizer, images::Image};
    let img = decode_rgb(path).map_err(|e| e.to_string())?;
    let src = Image::from_vec_u8(img.width(), img.height(), img.into_raw(), PixelType::U8x3)
        .map_err(|e| e.to_string())?;
    let mut dst = Image::new(w, h, PixelType::U8x3);
    Resizer::new().resize(&src, &mut dst, None).map_err(|e| e.to_string())?;
    let small = RgbImage::from_raw(w, h, dst.into_vec()).ok_or("resize buffer size mismatch")?;
    match image::ImageFormat::from_path(path).map_err(|e| e.to_string())? {
        image::ImageFormat::Jpeg => encode_jpeg(&small, 95),
        format => {
            let mut out = std::io::Cursor::new(Vec::new());
            small.write_to(&mut out, format).map_err(|e| e.to_string())?;
            Ok(out.into_inner())
        }
    }
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

    #[test]
    fn resize_batch_keeps_order_and_format() {
        let dir = tempfile::tempdir().unwrap();
        let (png, jpg) = (dir.path().join("a.png"), dir.path().join("b.jpg"));
        gradient(40, 20).save(&png).unwrap();
        std::fs::write(&jpg, encode_jpeg(&gradient(30, 60), 95).unwrap()).unwrap();
        let out = resize_batch(&[(png, 20, 10), (jpg.clone(), 15, 30), (jpg, 3, 6)]);
        let dims: Vec<_> =
            out.iter().map(|b| decode_bytes(b.as_ref().unwrap()).unwrap().dimensions()).collect();
        assert_eq!(dims, [(20, 10), (15, 30), (3, 6)]);
        assert!(out[0].as_ref().unwrap().starts_with(b"\x89PNG"));
    }
}
