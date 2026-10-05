//! Model runs in a background thread, with progress and cancellation. The
//! loaded detector is kept for the next run.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crossbeam_channel::Receiver;
use eframe::egui;
use fastlabel_infer::{BatchItem, BatchResult, Detector, Progress, run_batch};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    All,
    /// Frames without predictions yet (merged into the existing ones).
    Unpredicted,
    /// One image, by dataset image index.
    Frame(usize),
}

pub struct Outcome {
    pub scope: Scope,
    pub model: PathBuf,
    pub confidence: f64,
    pub class_names: Vec<String>,
    pub result: Result<BatchResult, String>,
}

struct Running {
    scope: Scope,
    progress: Arc<Progress>,
    rx: Receiver<Outcome>,
    thread: JoinHandle<()>,
}

#[derive(Default)]
pub struct Inference {
    detector: Arc<Mutex<Option<Detector>>>,
    running: Option<Running>,
}

/// ONNX Runtime threads: physical cores (hyper-threads don't help).
fn threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| (n.get() / 2).max(1))
}

impl Inference {
    pub fn is_running(&self) -> bool {
        self.running.is_some()
    }

    /// (scope, done, total) of the active run.
    pub fn progress(&self) -> Option<(Scope, usize, usize)> {
        self.running.as_ref().map(|r| {
            (
                r.scope,
                r.progress.done.load(Ordering::Relaxed),
                r.progress.total.load(Ordering::Relaxed),
            )
        })
    }

    pub fn start(
        &mut self,
        ctx: &egui::Context,
        model: &Path,
        confidence: f64,
        scope: Scope,
        items: Vec<BatchItem>,
    ) {
        if self.running.is_some() {
            return;
        }
        let (tx, rx) = crossbeam_channel::bounded(1);
        let progress = Arc::new(Progress::default());
        progress.total.store(items.len(), Ordering::Relaxed);
        let (detector, p, ctx, model) =
            (self.detector.clone(), progress.clone(), ctx.clone(), model.to_path_buf());
        let thread = std::thread::Builder::new()
            .name("fastlabel-inference".into())
            .spawn(move || {
                let mut guard = detector.lock().unwrap_or_else(|e| e.into_inner());
                let loaded = match guard.take() {
                    Some(d) if d.path == model => Ok(d),
                    _ => Detector::load(&model, threads()).map_err(|e| e.to_string()),
                };
                let outcome = match loaded {
                    Ok(mut det) => {
                        let notify = || ctx.request_repaint();
                        let result = run_batch(&mut det, items, confidence as f32, &p, &notify);
                        let class_names = det.class_names().to_vec();
                        *guard = Some(det);
                        Outcome { scope, model, confidence, class_names, result: Ok(result) }
                    }
                    Err(e) => Outcome {
                        scope,
                        model,
                        confidence,
                        class_names: Vec::new(),
                        result: Err(e),
                    },
                };
                let _ = tx.send(outcome);
                ctx.request_repaint();
            })
            .expect("spawn inference thread");
        self.running = Some(Running { scope, progress, rx, thread });
    }

    pub fn cancel(&self) {
        if let Some(r) = &self.running {
            r.progress.cancel.store(true, Ordering::Relaxed);
        }
    }

    /// The finished run, if any.
    pub fn poll(&mut self) -> Option<Outcome> {
        let outcome = self.running.as_ref()?.rx.try_recv().ok()?;
        if let Some(r) = self.running.take() {
            let _ = r.thread.join();
        }
        Some(outcome)
    }

    /// Cancels and waits (at most the image being processed).
    pub fn shutdown(&mut self) {
        self.cancel();
        if let Some(r) = self.running.take() {
            let _ = r.thread.join();
        }
    }
}
