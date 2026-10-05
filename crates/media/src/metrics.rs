//! Per-box image metrics used for sorting.

use image::RgbImage;

use crate::crop::Region;

/// Standard deviation of the luma (BT.601) inside `region`, in `[0, 127.5]`.
/// Low = uniform (blank, occluded), high = busy. `None` if the region is empty.
pub fn gray_std(img: &RgbImage, region: Region) -> Option<f32> {
    let r = region.clip_to(img.width(), img.height())?;
    let (mut sum, mut sum_sq) = (0f64, 0f64);
    for y in r.y..r.y + r.h {
        for x in r.x..r.x + r.w {
            let [red, green, blue] = img.get_pixel(x, y).0;
            let l = 0.299 * f64::from(red) + 0.587 * f64::from(green) + 0.114 * f64::from(blue);
            sum += l;
            sum_sq += l * l;
        }
    }
    let n = f64::from(r.w) * f64::from(r.h);
    let mean = sum / n;
    Some((sum_sq / n - mean * mean).max(0.0).sqrt() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_is_zero_and_checkerboard_is_high() {
        let flat = RgbImage::from_pixel(10, 10, image::Rgb([80, 80, 80]));
        assert_eq!(gray_std(&flat, Region { x: 0, y: 0, w: 10, h: 10 }), Some(0.0));
        let checker = RgbImage::from_fn(10, 10, |x, y| {
            if (x + y) % 2 == 0 { image::Rgb([0, 0, 0]) } else { image::Rgb([255, 255, 255]) }
        });
        let s = gray_std(&checker, Region { x: 0, y: 0, w: 10, h: 10 }).unwrap();
        assert!((s - 127.5).abs() < 0.5, "{s}");
        assert!(gray_std(&checker, Region { x: 20, y: 0, w: 5, h: 5 }).is_none());
    }
}
