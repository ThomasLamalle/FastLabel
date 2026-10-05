//! Cold / warm crop throughput on a real dataset (cache in a temp dir):
//! `cargo run --release -p fastlabel-media --example thumb_bench -- <dataset dir> [limit]`

use std::sync::Arc;
use std::time::{Duration, Instant};

use fastlabel_core::Project;
use fastlabel_media::{DiskCache, Job, Loader, Region};

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("dataset dir"));
    let limit: usize = std::env::args().nth(2).map_or(usize::MAX, |s| s.parse().unwrap());
    let project = Project::open(&dir).unwrap();
    let ds = &project.dataset;
    let jobs: Vec<(Job, u64)> = ds
        .annotations()
        .iter()
        .take(limit)
        .enumerate()
        .filter_map(|(i, a)| {
            let img = ds.image(a.image_id)?;
            let region = Region::around(&a.bbox, 50.0, 50.0, img.width, img.height)?;
            Some((Job::Thumb { path: ds.image_path(img), region, max_side: 256 }, i as u64))
        })
        .collect();
    let cache_dir = tempfile::tempdir().unwrap();
    for pass in ["cold", "warm"] {
        let loader = Loader::new(0, Some(DiskCache::new(cache_dir.path())), Arc::new(|| {}));
        let t = Instant::now();
        loader.set_wanted(jobs.iter().cloned());
        let mut errors = 0;
        for _ in 0..jobs.len() {
            let l = loader.recv_timeout(Duration::from_secs(60)).expect("result");
            errors += usize::from(l.image.is_err());
        }
        let el = t.elapsed();
        println!(
            "{pass}: {} crops in {:?} → {:.0} crops/s ({errors} errors)",
            jobs.len(),
            el,
            jobs.len() as f64 / el.as_secs_f64()
        );
    }
}
