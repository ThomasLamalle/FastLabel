//! Background image loading: a worker pool fed by a priority queue.
//!
//! The UI states what it wants (`set_wanted`, lowest priority value first);
//! anything no longer wanted is dropped before being decoded. Workers take
//! the most urgent job plus every other queued crop of the same frame, so a
//! frame is decoded once for all its crops. Crops go through the disk cache.

use std::cell::OnceCell;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender};
use image::RgbImage;

use crate::cache::{DiskCache, Fnv};
use crate::crop::{Region, fit, thumbnail};
use crate::decode::decode_rgb;

pub type Notify = Arc<dyn Fn() + Send + Sync>;
pub type JobId = u64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Job {
    /// A crop of `region`, downscaled to fit `max_side`.
    Thumb { path: PathBuf, region: Region, max_side: u32 },
    /// A whole frame, downscaled to fit `max_side`.
    Frame { path: PathBuf, max_side: u32 },
}

impl Job {
    pub fn id(&self) -> JobId {
        match self {
            Job::Thumb { path, region, max_side } => Fnv::default()
                .u64(1)
                .bytes(path.as_os_str().as_encoded_bytes())
                .u64(u64::from(region.x))
                .u64(u64::from(region.y))
                .u64(u64::from(region.w))
                .u64(u64::from(region.h))
                .u64(u64::from(*max_side))
                .finish(),
            Job::Frame { path, max_side } => Fnv::default()
                .u64(2)
                .bytes(path.as_os_str().as_encoded_bytes())
                .u64(u64::from(*max_side))
                .finish(),
        }
    }

    fn path(&self) -> &PathBuf {
        match self {
            Job::Thumb { path, .. } | Job::Frame { path, .. } => path,
        }
    }
}

#[derive(Debug)]
pub struct Loaded {
    pub id: JobId,
    pub image: Result<RgbImage, String>,
}

/// Pending jobs ordered by priority, with a per-frame index so all crops of
/// a frame can be taken together. Every operation is O(log n).
#[derive(Default)]
struct Queue {
    pending: HashMap<JobId, (Job, u64)>,
    order: BTreeSet<(u64, JobId)>,
    by_path: HashMap<PathBuf, HashSet<JobId>>,
    in_flight: HashSet<JobId>,
    shutdown: bool,
}

impl Queue {
    fn insert(&mut self, job: Job, prio: u64) {
        let id = job.id();
        if self.in_flight.contains(&id) {
            return;
        }
        if let Some((_, old)) = self.pending.get(&id) {
            if *old <= prio {
                return;
            }
            self.order.remove(&(*old, id));
        }
        self.order.insert((prio, id));
        self.by_path.entry(job.path().clone()).or_default().insert(id);
        self.pending.insert(id, (job, prio));
    }

    fn remove(&mut self, id: JobId) -> Option<Job> {
        let (job, prio) = self.pending.remove(&id)?;
        self.order.remove(&(prio, id));
        if let Some(set) = self.by_path.get_mut(job.path()) {
            set.remove(&id);
            if set.is_empty() {
                self.by_path.remove(job.path());
            }
        }
        Some(job)
    }

    fn clear(&mut self) {
        self.pending.clear();
        self.order.clear();
        self.by_path.clear();
    }

    /// The most urgent job plus every queued crop of the same frame.
    fn take_batch(&mut self) -> Option<Vec<(JobId, Job)>> {
        let &(_, first_id) = self.order.first()?;
        let first = self.remove(first_id)?;
        let mut batch = vec![(first_id, first)];
        let siblings: Vec<JobId> = self
            .by_path
            .get(batch[0].1.path())
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default();
        for id in siblings {
            if matches!(self.pending.get(&id), Some((Job::Thumb { .. }, _))) {
                let job = self.remove(id).expect("present");
                batch.push((id, job));
            }
        }
        self.in_flight.extend(batch.iter().map(|(id, _)| *id));
        Some(batch)
    }
}

struct Shared {
    queue: Mutex<Queue>,
    cv: Condvar,
    cache: Option<DiskCache>,
    notify: Notify,
    tx: Sender<Loaded>,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, Queue> {
        // A panicking worker must not take the whole loader down.
        self.queue.lock().unwrap_or_else(|e| e.into_inner())
    }
}

pub struct Loader {
    shared: Arc<Shared>,
    rx: Receiver<Loaded>,
    workers: Vec<JoinHandle<()>>,
}

impl Loader {
    /// `threads = 0` uses all cores but one.
    pub fn new(threads: usize, cache: Option<DiskCache>, notify: Notify) -> Self {
        let threads = if threads == 0 {
            std::thread::available_parallelism().map_or(2, |n| n.get().saturating_sub(1).max(1))
        } else {
            threads
        };
        let (tx, rx) = crossbeam_channel::unbounded();
        let shared =
            Arc::new(Shared { queue: Mutex::default(), cv: Condvar::new(), cache, notify, tx });
        let workers = (0..threads)
            .map(|i| {
                let shared = shared.clone();
                std::thread::Builder::new()
                    .name(format!("fastlabel-loader-{i}"))
                    .spawn(move || worker(&shared))
                    .expect("spawn loader thread")
            })
            .collect();
        Self { shared, rx, workers }
    }

    /// Replaces the queue with `jobs` (lower priority value = sooner). Jobs
    /// already being processed are not restarted.
    pub fn set_wanted(&self, jobs: impl IntoIterator<Item = (Job, u64)>) {
        let mut q = self.shared.lock();
        q.clear();
        for (job, prio) in jobs {
            q.insert(job, prio);
        }
        drop(q);
        self.shared.cv.notify_all();
    }

    /// Adds one job, keeping the rest of the queue.
    pub fn request(&self, job: Job, prio: u64) {
        self.shared.lock().insert(job, prio);
        self.shared.cv.notify_one();
    }

    /// Results delivered so far (non-blocking).
    pub fn poll(&self) -> Vec<Loaded> {
        self.rx.try_iter().collect()
    }

    pub fn recv_timeout(&self, timeout: Duration) -> Option<Loaded> {
        self.rx.recv_timeout(timeout).ok()
    }

    /// Jobs queued or being processed.
    pub fn busy(&self) -> usize {
        let q = self.shared.lock();
        q.pending.len() + q.in_flight.len()
    }
}

impl Drop for Loader {
    fn drop(&mut self) {
        self.shared.lock().shutdown = true;
        self.shared.cv.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

fn worker(shared: &Shared) {
    loop {
        let batch = {
            let mut q = shared.lock();
            while q.pending.is_empty() && !q.shutdown {
                q = shared.cv.wait(q).unwrap_or_else(|e| e.into_inner());
            }
            if q.shutdown {
                return;
            }
            q.take_batch().expect("non-empty")
        };

        let ids: Vec<JobId> = batch.iter().map(|(id, _)| *id).collect();
        process(shared, batch);
        shared.lock().in_flight.retain(|id| !ids.contains(id));
        (shared.notify)();
    }
}

fn process(shared: &Shared, batch: Vec<(JobId, Job)>) {
    // Decoded at most once per batch, only if some job is not cached.
    let frame: OnceCell<Result<RgbImage, String>> = OnceCell::new();
    let decode = |path: &PathBuf| {
        frame
            .get_or_init(|| decode_rgb(path).map_err(|e| e.to_string()))
            .as_ref()
            .map_err(Clone::clone)
    };
    for (id, job) in batch {
        let image = match &job {
            Job::Thumb { path, region, max_side } => {
                let key =
                    shared.cache.as_ref().and_then(|_| DiskCache::key(path, *region, *max_side));
                let cached = key.and_then(|k| shared.cache.as_ref()?.get(k));
                match cached {
                    Some(img) => Ok(img),
                    None => decode(path).and_then(|full| {
                        let thumb =
                            thumbnail(full, *region, *max_side).ok_or("crop outside the image")?;
                        if let (Some(cache), Some(k)) = (&shared.cache, key) {
                            cache.put(k, &thumb);
                        }
                        Ok(thumb)
                    }),
                }
            }
            Job::Frame { path, max_side } => decode(path).map(|full| {
                let (w, h) = fit(full.width(), full.height(), *max_side);
                if (w, h) == full.dimensions() {
                    full.clone()
                } else {
                    image::imageops::resize(full, w, h, image::imageops::FilterType::Triangle)
                }
            }),
        };
        if shared.tx.send(Loaded { id, image }).is_err() {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::tests::gradient;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn setup() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("frame.png");
        gradient(200, 100).save(&path).unwrap();
        (dir, path)
    }

    fn collect(loader: &Loader, n: usize) -> HashMap<JobId, Result<RgbImage, String>> {
        (0..n)
            .map(|_| loader.recv_timeout(Duration::from_secs(10)).expect("result in time"))
            .map(|l| (l.id, l.image))
            .collect()
    }

    #[test]
    fn loads_thumbs_and_frames_and_notifies() {
        let (dir, path) = setup();
        let hits = Arc::new(AtomicUsize::new(0));
        let h = hits.clone();
        let loader = Loader::new(
            2,
            Some(DiskCache::new(dir.path().join("cache"))),
            Arc::new(move || {
                h.fetch_add(1, Ordering::SeqCst);
            }),
        );
        let t1 = Job::Thumb {
            path: path.clone(),
            region: Region { x: 10, y: 10, w: 20, h: 10 },
            max_side: 256,
        };
        let t2 = Job::Thumb {
            path: path.clone(),
            region: Region { x: 0, y: 0, w: 200, h: 100 },
            max_side: 50,
        };
        let fr = Job::Frame { path: path.clone(), max_side: 100 };
        let missing = Job::Thumb {
            path: dir.path().join("nope.jpg"),
            region: Region { x: 0, y: 0, w: 1, h: 1 },
            max_side: 9,
        };
        loader.set_wanted([
            (t1.clone(), 0),
            (t2.clone(), 1),
            (fr.clone(), 2),
            (missing.clone(), 3),
        ]);
        let out = collect(&loader, 4);
        assert_eq!(out[&t1.id()].as_ref().unwrap().get_pixel(0, 0).0, [10, 10, 128]);
        assert_eq!(out[&t2.id()].as_ref().unwrap().dimensions(), (50, 25));
        assert_eq!(out[&fr.id()].as_ref().unwrap().dimensions(), (100, 50));
        assert!(out[&missing.id()].is_err());
        assert!(hits.load(Ordering::SeqCst) >= 1);

        // The cache key includes the source size and mtime: once the source is
        // gone, its stale cache entry must not be served.
        std::fs::remove_file(&path).unwrap();
        loader.set_wanted([(t1.clone(), 0)]);
        let out = collect(&loader, 1);
        let img = out[&t1.id()].as_ref();
        assert!(img.is_err(), "deleted source must not be served from a stale cache entry");
    }

    #[test]
    fn set_wanted_drops_stale_jobs() {
        let (_dir, path) = setup();
        let loader = Loader::new(1, None, Arc::new(|| {}));
        let jobs: Vec<_> = (0..50)
            .map(|i| {
                (
                    Job::Thumb {
                        path: path.clone(),
                        region: Region { x: i, y: 0, w: 10, h: 10 },
                        max_side: 256,
                    },
                    u64::from(i),
                )
            })
            .collect();
        loader.set_wanted(jobs);
        loader.set_wanted([]);
        std::thread::sleep(Duration::from_millis(200));
        // At most the batch already in flight was processed.
        assert!(loader.poll().len() <= 50);
        assert_eq!(loader.busy(), 0);
    }
}
