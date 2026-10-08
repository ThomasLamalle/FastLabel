//! Batch inference over many frames: two decoder threads keep the model fed
//! while it runs; progress and cancellation through shared atomics.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use fastlabel_media::decode::decode_rgb;

use crate::detector::Detector;
use crate::prepost::Detection;

pub struct BatchItem {
    /// Caller's identifier (e.g. image index).
    pub key: usize,
    pub path: PathBuf,
}

#[derive(Default)]
pub struct Progress {
    pub done: AtomicUsize,
    pub total: AtomicUsize,
    pub cancel: AtomicBool,
}

#[derive(Debug, Default)]
pub struct BatchResult {
    pub detections: Vec<(usize, Vec<Detection>)>,
    pub failed: Vec<(usize, String)>,
    pub cancelled: bool,
}

const DECODERS: usize = 2;

pub fn run_batch(
    det: &mut Detector,
    items: Vec<BatchItem>,
    conf: f32,
    progress: &Progress,
    notify: &(dyn Fn() + Sync),
) -> BatchResult {
    progress.total.store(items.len(), Ordering::Relaxed);
    progress.done.store(0, Ordering::Relaxed);
    let next = AtomicUsize::new(0);
    let (tx, rx) = crossbeam_channel::bounded(4);
    let mut result = BatchResult::default();
    std::thread::scope(|s| {
        for _ in 0..DECODERS {
            let tx = tx.clone();
            let (items, next) = (&items, &next);
            s.spawn(move || {
                loop {
                    if progress.cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(item) = items.get(i) else { return };
                    let img = decode_rgb(&item.path).map_err(|e| e.to_string());
                    if tx.send((item.key, img)).is_err() {
                        return;
                    }
                }
            });
        }
        drop(tx);
        for (key, img) in rx.iter() {
            if progress.cancel.load(Ordering::Relaxed) {
                result.cancelled = true;
                break;
            }
            match img.and_then(|img| det.detect(&img, conf).map_err(|e| e.to_string())) {
                Ok(d) => result.detections.push((key, d)),
                Err(e) => result.failed.push((key, e)),
            }
            progress.done.fetch_add(1, Ordering::Relaxed);
            notify();
        }
        // Unblock decoders waiting on a full channel.
        drop(rx);
    });
    result.cancelled |= progress.cancel.load(Ordering::Relaxed);
    result
}
