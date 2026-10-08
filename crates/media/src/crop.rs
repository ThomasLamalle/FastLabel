//! Crop regions and thumbnails.

use fastlabel_core::BBox;
use image::RgbImage;

/// Integer pixel rectangle inside an image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Region {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl Region {
    /// The box enlarged by `margin_x` / `margin_y` source pixels on each
    /// side, snapped outward to whole pixels and clipped to the image.
    pub fn around(
        bbox: &BBox,
        margin_x: f64,
        margin_y: f64,
        img_w: u32,
        img_h: u32,
    ) -> Option<Region> {
        if !bbox.is_finite() || img_w == 0 || img_h == 0 {
            return None;
        }
        let x1 = (bbox.x - margin_x).floor().clamp(0.0, f64::from(img_w)) as u32;
        let y1 = (bbox.y - margin_y).floor().clamp(0.0, f64::from(img_h)) as u32;
        let x2 = (bbox.x2() + margin_x).ceil().clamp(0.0, f64::from(img_w)) as u32;
        let y2 = (bbox.y2() + margin_y).ceil().clamp(0.0, f64::from(img_h)) as u32;
        (x2 > x1 && y2 > y1).then(|| Region { x: x1, y: y1, w: x2 - x1, h: y2 - y1 })
    }

    /// Clips to an image actually decoded (its size can differ from metadata).
    pub fn clip_to(&self, img_w: u32, img_h: u32) -> Option<Region> {
        let x2 = (self.x + self.w).min(img_w);
        let y2 = (self.y + self.h).min(img_h);
        (x2 > self.x && y2 > self.y).then(|| Region {
            x: self.x,
            y: self.y,
            w: x2 - self.x,
            h: y2 - self.y,
        })
    }
}

pub use fastlabel_core::geom::fit;

/// Cuts `region` out of `img`, downscaled to fit `max_side`.
pub fn thumbnail(img: &RgbImage, region: Region, max_side: u32) -> Option<RgbImage> {
    let r = region.clip_to(img.width(), img.height())?;
    let view = image::imageops::crop_imm(img, r.x, r.y, r.w, r.h);
    let (tw, th) = fit(r.w, r.h, max_side);
    if (tw, th) == (r.w, r.h) {
        Some(view.to_image())
    } else {
        Some(image::imageops::resize(&*view, tw, th, image::imageops::FilterType::Triangle))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::tests::gradient;

    #[test]
    fn region_adds_margins_and_clips() {
        let r = Region::around(&BBox::new(10.5, 20.0, 40.0, 20.0), 50.0, 30.0, 100, 60).unwrap();
        assert_eq!(r, Region { x: 0, y: 0, w: 100, h: 60 });
        let r = Region::around(&BBox::new(40.2, 20.0, 10.0, 10.0), 5.0, 0.0, 100, 60).unwrap();
        assert_eq!(r, Region { x: 35, y: 20, w: 21, h: 10 });
        assert!(Region::around(&BBox::new(200.0, 0.0, 5.0, 5.0), 0.0, 0.0, 100, 60).is_none());
    }

    #[test]
    fn thumbnail_crops_exact_pixels_and_downscales() {
        let img = gradient(300, 200);
        let t = thumbnail(&img, Region { x: 10, y: 20, w: 30, h: 15 }, 256).unwrap();
        assert_eq!(t.dimensions(), (30, 15));
        assert_eq!(t.get_pixel(0, 0).0, [10, 20, 128]);
        let t = thumbnail(&img, Region { x: 0, y: 0, w: 300, h: 200 }, 150).unwrap();
        assert_eq!(t.dimensions(), (150, 100));
        assert!(thumbnail(&img, Region { x: 400, y: 0, w: 5, h: 5 }, 150).is_none());
    }

    #[test]
    fn fit_keeps_aspect() {
        assert_eq!(fit(1000, 10, 100), (100, 1));
        assert_eq!(fit(50, 80, 100), (50, 80));
    }
}
