//! Corrupt images and extreme geometry must give errors, never panics.

use fastlabel_core::BBox;
use fastlabel_media::Region;
use fastlabel_media::crop::thumbnail;
use fastlabel_media::decode::{decode_bytes, encode_jpeg};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    #[test]
    fn random_bytes_do_not_panic(bytes in prop::collection::vec(any::<u8>(), 0..2000)) {
        let _ = decode_bytes(&bytes);
        let mut jpeg = vec![0xFF, 0xD8, 0xFF];
        jpeg.extend_from_slice(&bytes);
        let _ = decode_bytes(&jpeg);
    }

    #[test]
    fn truncated_or_flipped_jpeg_does_not_panic(cut in 0usize..3000, flip in 0usize..3000, v in any::<u8>()) {
        let img = image::RgbImage::from_fn(64, 48, |x, y| image::Rgb([x as u8 * 3, y as u8 * 5, 77]));
        let mut bytes = encode_jpeg(&img, 85).unwrap();
        let n = bytes.len();
        bytes[flip % n] = v;
        bytes.truncate(cut % (n + 1));
        let _ = decode_bytes(&bytes);
    }

    #[test]
    fn regions_and_thumbnails_stay_in_bounds(
        x in -1e7..1e7f64, y in -1e7..1e7f64, w in -10.0..1e7f64, h in -10.0..1e7f64,
        mx in 0.0..5000.0f64, my in 0.0..5000.0f64, side in 1u32..600,
    ) {
        let img = image::RgbImage::new(120, 80);
        if let Some(r) = Region::around(&BBox::new(x, y, w, h), mx, my, 120, 80) {
            prop_assert!(r.x + r.w <= 120 && r.y + r.h <= 80 && r.w > 0 && r.h > 0);
            let t = thumbnail(&img, r, side).unwrap();
            prop_assert!(t.width() <= side.max(r.w.min(side)) && t.width() >= 1);
        }
    }
}
