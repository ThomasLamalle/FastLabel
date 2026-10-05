//! GPU textures for crop thumbnails, fed by the background loader and
//! bounded by a memory budget (least recently drawn evicted first).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use eframe::egui::{self, ColorImage, TextureHandle, TextureOptions};
use fastlabel_media::{DiskCache, Job, JobId, Loader};

pub enum Slot {
    Ready(TextureHandle),
    Failed(String),
}

struct Entry {
    slot: Slot,
    last_used: u64,
    bytes: usize,
}

pub struct Thumbs {
    loader: Loader,
    entries: HashMap<JobId, Entry>,
    bytes: usize,
    budget: usize,
    frame: u64,
    wanted: Vec<JobId>,
}

impl Thumbs {
    pub fn new(ctx: &egui::Context, cache_dir: PathBuf, budget_bytes: usize) -> Self {
        let ctx = ctx.clone();
        let loader = Loader::new(
            0,
            Some(DiskCache::new(cache_dir)),
            Arc::new(move || ctx.request_repaint()),
        );
        Self {
            loader,
            entries: HashMap::new(),
            bytes: 0,
            budget: budget_bytes,
            frame: 0,
            wanted: Vec::new(),
        }
    }

    /// Uploads finished images; call once per frame before drawing.
    pub fn begin_frame(&mut self, ctx: &egui::Context) {
        self.frame += 1;
        for loaded in self.loader.poll() {
            let (slot, bytes) = match loaded.image {
                Ok(img) => {
                    let size = [img.width() as usize, img.height() as usize];
                    let color = ColorImage::from_rgb(size, img.as_raw());
                    let tex = ctx.load_texture(
                        format!("thumb-{:x}", loaded.id),
                        color,
                        TextureOptions::LINEAR,
                    );
                    (Slot::Ready(tex), size[0] * size[1] * 4)
                }
                Err(e) => (Slot::Failed(e), 0),
            };
            self.bytes += bytes;
            if let Some(old) =
                self.entries.insert(loaded.id, Entry { slot, last_used: self.frame, bytes })
            {
                self.bytes -= old.bytes;
            }
        }
        self.evict();
    }

    pub fn get(&mut self, id: JobId) -> Option<&Slot> {
        let e = self.entries.get_mut(&id)?;
        e.last_used = self.frame;
        Some(&e.slot)
    }

    /// The jobs the view needs, most urgent first. Only the missing ones are
    /// queued, and the loader is only told when the list changes.
    pub fn want(&mut self, jobs: Vec<Job>) {
        let missing: Vec<(Job, u64)> = jobs
            .into_iter()
            .filter(|j| !self.entries.contains_key(&j.id()))
            .enumerate()
            .map(|(i, j)| (j, i as u64))
            .collect();
        let ids: Vec<JobId> = missing.iter().map(|(j, _)| j.id()).collect();
        if ids != self.wanted {
            self.wanted = ids;
            self.loader.set_wanted(missing);
        }
    }

    /// Jobs queued or in progress.
    pub fn busy(&self) -> usize {
        self.loader.busy()
    }

    /// Forgets failed loads so they are retried (e.g. after a save).
    pub fn retry_failed(&mut self) {
        self.entries.retain(|_, e| !matches!(e.slot, Slot::Failed(_)));
        self.wanted.clear();
    }

    fn evict(&mut self) {
        if self.bytes <= self.budget {
            return;
        }
        let mut old: Vec<(u64, JobId)> = self
            .entries
            .iter()
            // Never evict what was drawn in the last two frames.
            .filter(|(_, e)| e.last_used + 2 < self.frame)
            .map(|(id, e)| (e.last_used, *id))
            .collect();
        old.sort_unstable();
        let target = self.budget / 10 * 9;
        for (_, id) in old {
            if self.bytes <= target {
                break;
            }
            if let Some(e) = self.entries.remove(&id) {
                self.bytes -= e.bytes;
            }
        }
    }
}
