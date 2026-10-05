//! Pre- and post-processing, independent of ONNX Runtime (unit-tested).

use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer, images::Image};
use fastlabel_core::BBox;
use image::RgbImage;

/// Ultralytics letterbox padding value.
const PAD: u8 = 114;

/// Mapping between original image pixels and model input pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Letterbox {
    pub scale: f64,
    pub pad_x: f64,
    pub pad_y: f64,
}

impl Letterbox {
    /// Same arithmetic as Ultralytics `LetterBox` (centered padding).
    pub fn new(img_w: u32, img_h: u32, in_w: u32, in_h: u32) -> (Self, u32, u32) {
        let r = (f64::from(in_w) / f64::from(img_w)).min(f64::from(in_h) / f64::from(img_h));
        let new_w = (f64::from(img_w) * r).round() as u32;
        let new_h = (f64::from(img_h) * r).round() as u32;
        let dw = f64::from(in_w - new_w) / 2.0;
        let dh = f64::from(in_h - new_h) / 2.0;
        let left = (dw - 0.1).round();
        let top = (dh - 0.1).round();
        (Self { scale: r, pad_x: left, pad_y: top }, new_w, new_h)
    }

    /// Model input size for dynamic-shape models: the image scaled to fit
    /// `imgsz`, padded only up to a multiple of `stride` (Ultralytics "rect").
    pub fn rect_input(img_w: u32, img_h: u32, imgsz: u32, stride: u32) -> (u32, u32) {
        let r = (f64::from(imgsz) / f64::from(img_w)).min(f64::from(imgsz) / f64::from(img_h));
        let new_w = (f64::from(img_w) * r).round() as u32;
        let new_h = (f64::from(img_h) * r).round() as u32;
        (new_w + (imgsz - new_w) % stride, new_h + (imgsz - new_h) % stride)
    }

    /// Input-space corners back to an original-image box.
    pub fn to_image(&self, x1: f64, y1: f64, x2: f64, y2: f64) -> BBox {
        BBox::from_corners(
            (x1 - self.pad_x) / self.scale,
            (y1 - self.pad_y) / self.scale,
            (x2 - self.pad_x) / self.scale,
            (y2 - self.pad_y) / self.scale,
        )
    }
}

fn resize(img: &RgbImage, w: u32, h: u32) -> RgbImage {
    if img.dimensions() == (w, h) {
        return img.clone();
    }
    let mut dst = Image::new(w, h, PixelType::U8x3);
    let opts = ResizeOptions::new().resize_alg(ResizeAlg::Interpolation(FilterType::Bilinear));
    Resizer::new().resize(img, &mut dst, &opts).expect("RGB8 resize");
    RgbImage::from_raw(w, h, dst.into_vec()).expect("buffer size")
}

/// Letterboxed, normalized (`/255`) CHW tensor data for YOLO models.
pub fn letterbox_chw(img: &RgbImage, in_w: u32, in_h: u32) -> (Vec<f32>, Letterbox) {
    let (lb, new_w, new_h) = Letterbox::new(img.width(), img.height(), in_w, in_h);
    let resized = resize(img, new_w, new_h);
    let plane = (in_w * in_h) as usize;
    let mut data = vec![f32::from(PAD) / 255.0; 3 * plane];
    let (ox, oy) = (lb.pad_x as u32, lb.pad_y as u32);
    for (x, y, p) in resized.enumerate_pixels() {
        let i = ((y + oy) * in_w + (x + ox)) as usize;
        for c in 0..3 {
            data[c * plane + i] = f32::from(p[c]) / 255.0;
        }
    }
    (data, lb)
}

/// Plain resize with ImageNet normalization (DETR-family models).
pub fn imagenet_chw(img: &RgbImage, in_w: u32, in_h: u32) -> Vec<f32> {
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    let resized = resize(img, in_w, in_h);
    let plane = (in_w * in_h) as usize;
    let mut data = vec![0.0; 3 * plane];
    for (i, p) in resized.pixels().enumerate() {
        for c in 0..3 {
            data[c * plane + i] = (f32::from(p[c]) / 255.0 - MEAN[c]) / STD[c];
        }
    }
    data
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detection {
    /// Model class index.
    pub class: usize,
    /// Original-image pixels.
    pub bbox: BBox,
    pub score: f32,
}

/// Raw YOLO head `[4 + nc, anchors]` (cx, cy, w, h in input pixels, then
/// class scores): confidence filter, back to image space, class-wise NMS.
pub fn decode_yolo_raw(
    out: &[f32],
    (channels, anchors): (usize, usize),
    conf: f32,
    lb: &Letterbox,
    img_w: u32,
    img_h: u32,
    (nms_iou, max_det): (f64, usize),
) -> Vec<Detection> {
    let nc = channels.saturating_sub(4);
    let at = |c: usize, a: usize| out[c * anchors + a];
    let mut dets = Vec::new();
    for a in 0..anchors {
        let (class, score) = (0..nc)
            .map(|k| (k, at(4 + k, a)))
            .fold((0, f32::MIN), |b, x| if x.1 > b.1 { x } else { b });
        if score < conf || nc == 0 {
            continue;
        }
        let (cx, cy, w, h) =
            (f64::from(at(0, a)), f64::from(at(1, a)), f64::from(at(2, a)), f64::from(at(3, a)));
        let b = lb.to_image(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0);
        if let Some(bbox) = b.clip(f64::from(img_w), f64::from(img_h)) {
            dets.push(Detection { class, bbox, score });
        }
    }
    nms(dets, nms_iou, max_det)
}

/// End-to-end YOLO output `[n, 6]` = x1, y1, x2, y2, score, class (input pixels).
pub fn decode_yolo_end2end(
    out: &[f32],
    rows: usize,
    conf: f32,
    lb: &Letterbox,
    img_w: u32,
    img_h: u32,
) -> Vec<Detection> {
    let mut dets: Vec<Detection> = (0..rows)
        .filter_map(|r| {
            let v = &out[r * 6..r * 6 + 6];
            if v[4] < conf {
                return None;
            }
            let b = lb.to_image(f64::from(v[0]), f64::from(v[1]), f64::from(v[2]), f64::from(v[3]));
            let bbox = b.clip(f64::from(img_w), f64::from(img_h))?;
            Some(Detection { class: v[5].max(0.0) as usize, bbox, score: v[4] })
        })
        .collect();
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    dets
}

/// DETR-style outputs: `boxes [q, 4]` (cx, cy, w, h normalized) and
/// `logits [q, nc]`; best class per query after a sigmoid.
pub fn decode_detr(
    boxes: &[f32],
    logits: &[f32],
    queries: usize,
    nc: usize,
    conf: f32,
    img_w: u32,
    img_h: u32,
) -> Vec<Detection> {
    let (fw, fh) = (f64::from(img_w), f64::from(img_h));
    let mut dets: Vec<Detection> = (0..queries)
        .filter_map(|q| {
            let row = &logits[q * nc..(q + 1) * nc];
            let (class, logit) = row
                .iter()
                .copied()
                .enumerate()
                .fold((0, f32::MIN), |b, x| if x.1 > b.1 { x } else { b });
            let score = 1.0 / (1.0 + (-logit).exp());
            if score < conf {
                return None;
            }
            let v = &boxes[q * 4..q * 4 + 4];
            let (cx, cy, w, h) = (
                f64::from(v[0]) * fw,
                f64::from(v[1]) * fh,
                f64::from(v[2]) * fw,
                f64::from(v[3]) * fh,
            );
            let bbox = BBox::from_corners(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
                .clip(fw, fh)?;
            Some(Detection { class, bbox, score })
        })
        .collect();
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    dets
}

/// Greedy class-wise non-maximum suppression, highest score first.
pub fn nms(mut dets: Vec<Detection>, iou: f64, max_det: usize) -> Vec<Detection> {
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut keep: Vec<Detection> = Vec::new();
    for d in dets {
        if keep.len() >= max_det {
            break;
        }
        if keep.iter().all(|k| k.class != d.class || k.bbox.iou(&d.bbox) <= iou) {
            keep.push(d);
        }
    }
    keep
}

/// Class names from Ultralytics metadata, e.g. `{0: 'bib', 1: "face"}`.
pub fn parse_names(meta: &str) -> Vec<String> {
    let inner = meta.trim().trim_start_matches('{').trim_end_matches('}');
    let mut pairs: Vec<(usize, String)> = Vec::new();
    let mut rest = inner;
    while let Some(colon) = rest.find(':') {
        let Ok(idx) = rest[..colon].trim().trim_start_matches(',').trim().parse::<usize>() else {
            break;
        };
        let after = rest[colon + 1..].trim_start();
        let Some(quote) = after.chars().next().filter(|c| *c == '\'' || *c == '"') else { break };
        let Some(end) = after[1..].find(quote) else { break };
        pairs.push((idx, after[1..1 + end].to_string()));
        rest = &after[end + 2..];
    }
    pairs.sort();
    pairs.into_iter().map(|(_, n)| n).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_matches_ultralytics_for_1080p_into_1280() {
        let (lb, w, h) = Letterbox::new(1920, 1080, 1280, 1280);
        assert_eq!((w, h), (1280, 720));
        assert_eq!((lb.pad_x, lb.pad_y), (0.0, 280.0));
        let b = lb.to_image(0.0, 280.0, 1280.0, 1000.0);
        assert!((b.w - 1920.0).abs() < 1e-9 && (b.h - 1080.0).abs() < 1e-9 && b.y.abs() < 1e-9);
    }

    #[test]
    fn rect_input_matches_ultralytics_auto_padding() {
        assert_eq!(Letterbox::rect_input(1920, 1080, 1280, 32), (1280, 736));
        assert_eq!(Letterbox::rect_input(1080, 1920, 640, 32), (384, 640));
        let (lb, w, h) = Letterbox::new(1920, 1080, 1280, 736);
        assert_eq!((w, h, lb.pad_y), (1280, 720, 8.0));
    }

    #[test]
    fn letterbox_tensor_layout() {
        let img = RgbImage::from_pixel(4, 2, image::Rgb([255, 0, 51]));
        let (data, lb) = letterbox_chw(&img, 4, 4);
        assert_eq!(lb.pad_y, 1.0);
        let plane = 16;
        assert_eq!(data.len(), 3 * plane);
        assert!((data[0] - 114.0 / 255.0).abs() < 1e-6, "top padding row");
        assert!((data[4] - 1.0).abs() < 1e-6, "R of first image row");
        assert!(data[plane + 4].abs() < 1e-6, "G");
        assert!((data[2 * plane + 4] - 0.2).abs() < 1e-6, "B");
    }

    #[test]
    fn raw_decode_filters_maps_and_suppresses() {
        // 3 anchors, 1 class: two overlapping boxes and one low score.
        let anchors = 3;
        let mut out = vec![0f32; 5 * anchors];
        let set = |out: &mut Vec<f32>, a: usize, v: [f32; 5]| {
            for (c, x) in v.iter().enumerate() {
                out[c * anchors + a] = *x;
            }
        };
        set(&mut out, 0, [100.0, 380.0, 40.0, 20.0, 0.9]);
        set(&mut out, 1, [101.0, 380.0, 40.0, 20.0, 0.8]);
        set(&mut out, 2, [600.0, 600.0, 40.0, 20.0, 0.05]);
        let (lb, ..) = Letterbox::new(1920, 1080, 1280, 1280);
        let dets = decode_yolo_raw(&out, (5, anchors), 0.1, &lb, 1920, 1080, (0.7, 300));
        assert_eq!(dets.len(), 1);
        let b = dets[0].bbox;
        assert!(
            (b.x - 120.0).abs() < 1e-6 && (b.y - 135.0).abs() < 1e-6 && (b.w - 60.0).abs() < 1e-6
        );
    }

    #[test]
    fn nms_is_class_wise() {
        let d = |class, x, score| Detection { class, bbox: BBox::new(x, 0.0, 10.0, 10.0), score };
        let kept =
            nms(vec![d(0, 0.0, 0.9), d(1, 0.0, 0.8), d(0, 1.0, 0.7), d(0, 50.0, 0.6)], 0.5, 300);
        assert_eq!(kept.len(), 3);
        assert_eq!(nms(vec![d(0, 0.0, 0.9), d(0, 50.0, 0.6)], 0.5, 1).len(), 1);
    }

    #[test]
    fn end2end_and_detr_decode() {
        let (lb, ..) = Letterbox::new(640, 640, 640, 640);
        let out = [10.0, 20.0, 30.0, 60.0, 0.8, 0.0, 0.0, 0.0, 5.0, 5.0, 0.01, 0.0];
        let dets = decode_yolo_end2end(&out, 2, 0.25, &lb, 640, 640);
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].bbox, BBox::new(10.0, 20.0, 20.0, 40.0));

        let boxes = [0.5, 0.5, 0.1, 0.2, 0.1, 0.1, 0.05, 0.05];
        let logits = [-5.0, 3.0, -5.0, -6.0];
        let dets = decode_detr(&boxes, &logits, 2, 2, 0.5, 100, 200);
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].class, 1);
        let b = dets[0].bbox;
        let close = |a: f64, b: f64| (a - b).abs() < 1e-4;
        assert!(
            close(b.x, 45.0) && close(b.y, 80.0) && close(b.w, 10.0) && close(b.h, 40.0),
            "{b:?}"
        );
    }

    #[test]
    fn parses_ultralytics_names() {
        assert_eq!(parse_names("{0: 'bib'}"), vec!["bib"]);
        assert_eq!(parse_names("{1: \"face\", 0: 'bib, race'}"), vec!["bib, race", "face"]);
        assert!(parse_names("garbage").is_empty());
    }
}
