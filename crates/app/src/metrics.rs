//! Per-box color uniformity (grayscale standard deviation), computed in the
//! background for boxes that don't have a value yet and cached in
//! `.fastlabel/metrics.json`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use fastlabel_core::{BBox, fsutil};
use fastlabel_infer::Progress;
use fastlabel_media::Region;
use fastlabel_media::cache::Fnv;
use fastlabel_media::decode::decode_rgb;
use fastlabel_media::metrics::gray_std;
use serde::{Deserialize, Serialize};

pub const METRICS_FILE: &str = "metrics.json";
const FORMAT: u32 = 1;

/// Stable key of a box on an image file.
pub fn key(file_name: &str, b: &BBox) -> u64 {
    let mut h = Fnv::default().bytes(file_name.as_bytes());
    for v in [b.x, b.y, b.w, b.h] {
        h = h.u64(v.to_bits());
    }
    h.finish()
}

#[derive(Default, Serialize, Deserialize)]
pub struct Uniformity {
    version: u32,
    /// Hex key → grayscale std dev.
    values: HashMap<String, f32>,
    #[serde(skip)]
    by_key: HashMap<u64, f32>,
}

impl Uniformity {
    pub fn load(path: &Path) -> Self {
        let mut u: Self = fsutil::read_json(path).unwrap_or_default();
        if u.version != FORMAT {
            return Self::default();
        }
        u.by_key = u
            .values
            .iter()
            .filter_map(|(k, v)| Some((u64::from_str_radix(k, 16).ok()?, *v)))
            .collect();
        u
    }

    pub fn save(&mut self, path: &Path) -> fastlabel_core::Result<()> {
        self.version = FORMAT;
        self.values = self.by_key.iter().map(|(k, v)| (format!("{k:016x}"), *v)).collect();
        fsutil::write_json_atomic(path, self)
    }

    pub fn get(&self, k: u64) -> Option<f32> {
        self.by_key.get(&k).copied()
    }

    pub fn is_empty(&self) -> bool {
        self.by_key.is_empty()
    }

    pub fn extend(&mut self, values: impl IntoIterator<Item = (u64, f32)>) {
        self.by_key.extend(values);
    }
}

/// Boxes of one frame still missing a value.
pub struct FrameBoxes {
    pub path: PathBuf,
    pub boxes: Vec<(u64, BBox)>,
}

/// Decodes each frame once (in parallel) and measures all its boxes.
pub fn compute(frames: Vec<FrameBoxes>, threads: usize, progress: &Progress) -> HashMap<u64, f32> {
    progress.total.store(frames.len(), Ordering::Relaxed);
    let next = AtomicUsize::new(0);
    let out = Mutex::new(HashMap::new());
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| {
                loop {
                    if progress.cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(frame) = frames.get(i) else { return };
                    if let Ok(img) = decode_rgb(&frame.path) {
                        let vals: Vec<(u64, f32)> = frame
                            .boxes
                            .iter()
                            .filter_map(|(k, b)| {
                                let r = Region::around(b, 0.0, 0.0, img.width(), img.height())?;
                                Some((*k, gray_std(&img, r)?))
                            })
                            .collect();
                        out.lock().unwrap_or_else(|e| e.into_inner()).extend(vals);
                    }
                    progress.done.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
    });
    out.into_inner().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_and_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let img = image::RgbImage::from_fn(40, 20, |x, _| {
            if x < 20 { image::Rgb([90, 90, 90]) } else { image::Rgb([(x * 12) as u8, 0, 0]) }
        });
        let path = dir.path().join("a.png");
        img.save(&path).unwrap();
        let flat = BBox::new(0.0, 0.0, 10.0, 10.0);
        let busy = BBox::new(20.0, 0.0, 20.0, 20.0);
        let frames = vec![FrameBoxes {
            path,
            boxes: vec![(key("a", &flat), flat), (key("a", &busy), busy)],
        }];
        let vals = compute(frames, 2, &Progress::default());
        assert_eq!(vals[&key("a", &flat)], 0.0);
        assert!(vals[&key("a", &busy)] > 10.0);

        let mut u = Uniformity::default();
        u.extend(vals);
        let file = dir.path().join(METRICS_FILE);
        u.save(&file).unwrap();
        let back = Uniformity::load(&file);
        assert_eq!(back.get(key("a", &flat)), Some(0.0));
        assert_ne!(key("a", &flat), key("b", &flat));
    }
}
