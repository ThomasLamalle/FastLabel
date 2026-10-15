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

/// The ONNX file run for `model`: ultralytics `.pt` weights are exported
/// next to themselves. `None` when that export is missing or older.
pub fn onnx_for(model: &Path) -> Option<PathBuf> {
    if model.extension().is_none_or(|e| e != "pt") {
        return Some(model.to_path_buf());
    }
    let onnx = model.with_extension("onnx");
    let mtime = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    (mtime(&onnx)? >= mtime(model)?).then_some(onnx)
}

/// Exports ultralytics `.pt` weights to `<same name>.onnx` (raw head, the
/// training image size) with `uvx`, CPU torch. Blocks for ~15 s, plus a
/// one-time download of ultralytics and torch.
pub fn export_pt(pt: &Path) -> Result<PathBuf, String> {
    let out = std::process::Command::new("uvx")
        .args(["--torch-backend", "cpu", "--from", "ultralytics", "--with", "onnx"])
        .args(["--with", "onnxslim", "yolo", "export", "format=onnx"])
        .arg(format!("model={}", pt.display()))
        .output()
        .map_err(|e| format!("cannot run uvx (install uv): {e}"))?;
    let onnx = pt.with_extension("onnx");
    if out.status.success() && onnx.is_file() {
        return Ok(onnx);
    }
    let log = String::from_utf8_lossy(&out.stderr) + String::from_utf8_lossy(&out.stdout);
    let tail: Vec<&str> = log.lines().rev().take(3).collect();
    Err(format!("ONNX export of {} failed: {}", pt.display(), tail.join(" | ")))
}

/// ONNX Runtime threads: physical cores (hyper-threads don't help).
pub(crate) fn threads() -> usize {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pt_needs_a_fresh_export() {
        let dir = tempfile::tempdir().unwrap();
        let pt = dir.path().join("best.pt");
        assert_eq!(onnx_for(Path::new("m.onnx")), Some(PathBuf::from("m.onnx")));
        std::fs::write(&pt, "").unwrap();
        assert_eq!(onnx_for(&pt), None);
        std::fs::write(dir.path().join("best.onnx"), "").unwrap();
        assert_eq!(onnx_for(&pt), Some(dir.path().join("best.onnx")));
    }
}
