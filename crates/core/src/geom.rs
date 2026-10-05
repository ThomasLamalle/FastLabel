//! Axis-aligned bounding boxes in original-image pixel coordinates.

use serde::{Deserialize, Serialize};

/// Boxes narrower or shorter than this (in source pixels) are treated as invalid.
pub const MIN_BOX_SIDE: f64 = 1.0;

/// COCO-style box: top-left corner plus width and height.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(from = "[f64; 4]", into = "[f64; 4]")]
pub struct BBox {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl From<[f64; 4]> for BBox {
    fn from([x, y, w, h]: [f64; 4]) -> Self {
        Self { x, y, w, h }
    }
}

impl From<BBox> for [f64; 4] {
    fn from(b: BBox) -> Self {
        [b.x, b.y, b.w, b.h]
    }
}

impl BBox {
    pub const fn new(x: f64, y: f64, w: f64, h: f64) -> Self {
        Self { x, y, w, h }
    }

    /// Builds a box from two corners given in any order.
    pub fn from_corners(x1: f64, y1: f64, x2: f64, y2: f64) -> Self {
        let (x1, x2) = if x1 <= x2 { (x1, x2) } else { (x2, x1) };
        let (y1, y2) = if y1 <= y2 { (y1, y2) } else { (y2, y1) };
        Self::new(x1, y1, x2 - x1, y2 - y1)
    }

    pub fn x2(&self) -> f64 {
        self.x + self.w
    }

    pub fn y2(&self) -> f64 {
        self.y + self.h
    }

    pub fn area(&self) -> f64 {
        self.w.max(0.0) * self.h.max(0.0)
    }

    pub fn intersection_area(&self, other: &BBox) -> f64 {
        let w = self.x2().min(other.x2()) - self.x.max(other.x);
        let h = self.y2().min(other.y2()) - self.y.max(other.y);
        if w <= 0.0 || h <= 0.0 { 0.0 } else { w * h }
    }

    /// Intersection over union, in `[0, 1]`. Degenerate boxes give 0.
    pub fn iou(&self, other: &BBox) -> f64 {
        let inter = self.intersection_area(other);
        if inter <= 0.0 {
            return 0.0;
        }
        let union = self.area() + other.area() - inter;
        if union <= 0.0 {
            0.0
        } else {
            (inter / union).clamp(0.0, 1.0)
        }
    }

    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.w.is_finite() && self.h.is_finite()
    }

    /// Clips the box to `[0, width] x [0, height]`. Returns `None` when the
    /// result is not finite or smaller than [`MIN_BOX_SIDE`] on either side.
    pub fn clip(&self, width: f64, height: f64) -> Option<BBox> {
        if !self.is_finite() {
            return None;
        }
        let x1 = self.x.clamp(0.0, width);
        let y1 = self.y.clamp(0.0, height);
        let x2 = self.x2().clamp(0.0, width);
        let y2 = self.y2().clamp(0.0, height);
        let b = BBox::new(x1, y1, x2 - x1, y2 - y1);
        (b.w >= MIN_BOX_SIDE && b.h >= MIN_BOX_SIDE).then_some(b)
    }

    /// Bit-exact identity, used to recognise the same prediction across sessions.
    pub fn bits(&self) -> [u64; 4] {
        [
            self.x.to_bits(),
            self.y.to_bits(),
            self.w.to_bits(),
            self.h.to_bits(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn iou_of_identical_boxes_is_one() {
        let b = BBox::new(10.0, 20.0, 30.0, 40.0);
        assert!((b.iou(&b) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn iou_of_disjoint_or_touching_boxes_is_zero() {
        let a = BBox::new(0.0, 0.0, 10.0, 10.0);
        assert_eq!(a.iou(&BBox::new(10.0, 0.0, 10.0, 10.0)), 0.0);
        assert_eq!(a.iou(&BBox::new(50.0, 50.0, 10.0, 10.0)), 0.0);
    }

    #[test]
    fn iou_of_half_overlap() {
        let a = BBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BBox::new(5.0, 0.0, 10.0, 10.0);
        assert!((a.iou(&b) - 50.0 / 150.0).abs() < 1e-12);
    }

    #[test]
    fn clip_keeps_inside_part_and_rejects_tiny_or_outside() {
        let b = BBox::new(-5.0, -5.0, 20.0, 20.0)
            .clip(100.0, 100.0)
            .unwrap();
        assert_eq!(b, BBox::new(0.0, 0.0, 15.0, 15.0));
        assert!(
            BBox::new(200.0, 0.0, 10.0, 10.0)
                .clip(100.0, 100.0)
                .is_none()
        );
        assert!(
            BBox::new(10.0, 10.0, 0.5, 10.0)
                .clip(100.0, 100.0)
                .is_none()
        );
        assert!(
            BBox::new(f64::NAN, 0.0, 1.0, 1.0)
                .clip(100.0, 100.0)
                .is_none()
        );
    }

    #[test]
    fn from_corners_normalises_reverse_drags() {
        assert_eq!(
            BBox::from_corners(10.0, 8.0, 2.0, 4.0),
            BBox::new(2.0, 4.0, 8.0, 4.0)
        );
    }

    #[test]
    fn serializes_as_coco_array() {
        let b = BBox::new(1.5, 2.0, 3.0, 4.0);
        assert_eq!(serde_json::to_string(&b).unwrap(), "[1.5,2.0,3.0,4.0]");
        let back: BBox = serde_json::from_str("[1.5,2,3,4]").unwrap();
        assert_eq!(back, b);
    }

    fn arb_box() -> impl Strategy<Value = BBox> {
        (
            -100.0..100.0f64,
            -100.0..100.0f64,
            0.0..100.0f64,
            0.0..100.0f64,
        )
            .prop_map(|(x, y, w, h)| BBox::new(x, y, w, h))
    }

    proptest! {
        #[test]
        fn iou_is_symmetric_and_bounded(a in arb_box(), b in arb_box()) {
            let ab = a.iou(&b);
            prop_assert!((0.0..=1.0).contains(&ab));
            prop_assert!((ab - b.iou(&a)).abs() < 1e-9);
        }

        #[test]
        fn clipped_box_stays_inside(a in arb_box()) {
            if let Some(c) = a.clip(50.0, 40.0) {
                prop_assert!(c.x >= 0.0 && c.y >= 0.0 && c.x2() <= 50.0 && c.y2() <= 40.0);
                prop_assert!(c.w >= MIN_BOX_SIDE && c.h >= MIN_BOX_SIDE);
            }
        }
    }
}
