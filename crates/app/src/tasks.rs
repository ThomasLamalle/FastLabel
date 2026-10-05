//! Generic background task with progress, cancellation and a result.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::thread::JoinHandle;

use crossbeam_channel::Receiver;
use eframe::egui;
use fastlabel_infer::Progress;

pub struct Task<T> {
    pub label: &'static str,
    progress: Arc<Progress>,
    rx: Receiver<T>,
    handle: Option<JoinHandle<()>>,
}

impl<T: Send + 'static> Task<T> {
    /// Runs `work` on its own thread; the UI is repainted on completion.
    pub fn spawn(
        label: &'static str,
        ctx: &egui::Context,
        work: impl FnOnce(&Progress) -> T + Send + 'static,
    ) -> Self {
        let (tx, rx) = crossbeam_channel::bounded(1);
        let progress = Arc::new(Progress::default());
        let (p, ctx) = (progress.clone(), ctx.clone());
        let handle = std::thread::Builder::new()
            .name(format!("fastlabel-{label}"))
            .spawn(move || {
                let _ = tx.send(work(&p));
                ctx.request_repaint();
            })
            .expect("spawn task thread");
        Self { label, progress, rx, handle: Some(handle) }
    }

    pub fn poll(&mut self) -> Option<T> {
        let out = self.rx.try_recv().ok()?;
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        Some(out)
    }

    pub fn progress(&self) -> (usize, usize) {
        (self.progress.done.load(Ordering::Relaxed), self.progress.total.load(Ordering::Relaxed))
    }

    pub fn cancel(&self) {
        self.progress.cancel.store(true, Ordering::Relaxed);
    }
}

impl<T> Drop for Task<T> {
    fn drop(&mut self) {
        self.progress.cancel.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}
