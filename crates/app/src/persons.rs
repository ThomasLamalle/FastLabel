//! Person boxes added to exports (not to the dataset): COCO-pretrained
//! yolo26x run with ONNX Runtime on the CPU. Each frame's detections are
//! cached in one small file, so a frame is never detected twice.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::UNIX_EPOCH;

use fastlabel_core::coco::{CocoAnnotation, CocoCategory, CocoFile};
use fastlabel_core::{BBox, Split, fsutil};
use fastlabel_infer::{Detector, Progress};
use fastlabel_media::cache::Fnv;

/// Ultralytics asset, downloaded and exported to ONNX on first use.
const MODEL: &str = "yolo26x.pt";
/// COCO class index of "person".
const PERSON: usize = 0;
/// Detections are cached down to this score, so a higher export threshold
/// needs no rerun.
const CACHE_CONF: f32 = 0.25;
/// Score of the person boxes written to exports.
pub const CONF: f32 = 0.5;
pub const CATEGORY: &str = "person";

/// Cached detections of one frame: (box, score).
type Persons = Vec<(BBox, f32)>;

/// The export's annotations with persons added; `None` when cancelled.
pub type Outcome = Result<Option<CocoFile>, String>;

/// The yolo26x ONNX file under `dir`, exported with `uvx` when missing.
fn model(dir: &Path) -> Result<PathBuf, String> {
    let onnx = dir.join(MODEL).with_extension("onnx");
    if onnx.is_file() {
        return Ok(onnx);
    }
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    crate::inference::export_pt(&dir.join(MODEL))
}

/// Where downloaded models are kept (shared by all datasets).
pub fn models_dir() -> PathBuf {
    dirs::cache_dir().unwrap_or_else(std::env::temp_dir).join("fastlabel").join("models")
}

/// Cache file of `image`: model, path, size and mtime; `None` when the
/// file cannot be stat'ed.
fn cache_file(cache: &Path, image: &Path) -> Option<PathBuf> {
    let meta = std::fs::metadata(image).ok()?;
    let mtime = meta.modified().ok()?.duration_since(UNIX_EPOCH).ok()?.as_nanos() as u64;
    let key = Fnv::default()
        .bytes(MODEL.as_bytes())
        .bytes(image.as_os_str().as_encoded_bytes())
        .u64(meta.len())
        .u64(mtime)
        .finish();
    let hex = format!("{key:016x}");
    Some(cache.join(&hex[..2]).join(format!("{hex}.json")))
}

/// `coco` plus a "person" category (reused when one exists) and its boxes
/// on the frames of `splits`. Only uncached frames are detected; progress
/// counts them. `Ok(None)` when cancelled (the work done stays cached).
pub fn add(
    root: &Path,
    coco: &CocoFile,
    splits: &[Split],
    cache: &Path,
    models: &Path,
    progress: &Progress,
) -> Outcome {
    let frames: Vec<(usize, PathBuf, PathBuf)> = coco
        .images
        .iter()
        .enumerate()
        .filter(|(_, im)| Split::of_image(im).is_some_and(|s| splits.contains(&s)))
        .filter_map(|(i, im)| {
            let path = root.join(&im.file_name);
            Some((i, cache_file(cache, &path)?, path))
        })
        .collect();
    let mut found: Vec<Option<Persons>> =
        frames.iter().map(|(_, file, _)| fsutil::read_json(file).ok()).collect();
    let todo = found.iter().filter(|f| f.is_none()).count();
    progress.total.store(todo, Ordering::Relaxed);
    progress.done.store(0, Ordering::Relaxed);
    if todo > 0 {
        let onnx = model(models)?;
        let mut det = Detector::load(&onnx, crate::inference::threads())
            .map_err(|e| format!("{}: {e}", onnx.display()))?;
        // ponytail: decode and detect on one thread; decoding is a few % of a yolo26x run.
        for ((_, file, path), slot) in frames.iter().zip(&mut found).filter(|(_, s)| s.is_none()) {
            if progress.cancel.load(Ordering::Relaxed) {
                return Ok(None);
            }
            let img = fastlabel_media::decode::decode_rgb(path)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            let persons: Persons = det
                .detect(&img, CACHE_CONF)
                .map_err(|e| format!("{}: {e}", path.display()))?
                .into_iter()
                .filter(|d| d.class == PERSON)
                .map(|d| (d.bbox, d.score))
                .collect();
            fsutil::write_json_atomic(file, &persons).map_err(|e| e.to_string())?;
            *slot = Some(persons);
            progress.done.fetch_add(1, Ordering::Relaxed);
        }
    }
    Ok(Some(merge(coco, frames.iter().map(|f| f.0).zip(found.into_iter().flatten()))))
}

/// `coco` with the person boxes scoring at least [`CONF`] added.
fn merge(coco: &CocoFile, persons: impl Iterator<Item = (usize, Persons)>) -> CocoFile {
    let mut out = coco.clone();
    let cat = match out.categories.iter().find(|c| c.name == CATEGORY) {
        Some(c) => c.id,
        None => {
            let id = out.categories.iter().map(|c| c.id).max().unwrap_or(0) + 1;
            out.categories.push(CocoCategory {
                id,
                name: CATEGORY.into(),
                supercategory: String::new(),
                extra: Default::default(),
            });
            id
        }
    };
    let mut id = out.next_annotation_id();
    for (i, boxes) in persons {
        let image_id = coco.images[i].id;
        for (bbox, _) in boxes.into_iter().filter(|(_, s)| *s >= CONF) {
            out.annotations.push(CocoAnnotation::new(id, image_id, cat, bbox));
            id += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_frames_need_no_model_and_low_scores_are_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("a.jpg"), b"x").unwrap();
        std::fs::write(root.join("b.jpg"), b"y").unwrap();
        let coco: CocoFile = serde_json::from_str(
            r#"{"images": [
                {"id": 1, "file_name": "a.jpg", "width": 100, "height": 100, "split": "train"},
                {"id": 2, "file_name": "b.jpg", "width": 100, "height": 100, "split": "test"}],
              "annotations": [{"id": 5, "image_id": 1, "category_id": 3, "bbox": [1, 1, 5, 5]}],
              "categories": [{"id": 3, "name": "bib"}]}"#,
        )
        .unwrap();
        let cache = root.join("cache");
        let persons: Persons =
            vec![(BBox::new(10.0, 10.0, 20.0, 40.0), 0.9), (BBox::new(0.0, 0.0, 9.0, 9.0), 0.3)];
        let file = cache_file(&cache, &root.join("a.jpg")).unwrap();
        fsutil::write_json_atomic(&file, &persons).unwrap();

        // Frame b is uncached but not exported; the missing model is never needed.
        let progress = Progress::default();
        let out = add(root, &coco, &[Split::Train], &cache, &root.join("none"), &progress)
            .unwrap()
            .unwrap();
        assert_eq!(progress.total.load(Ordering::Relaxed), 0);
        assert_eq!(out.categories[1].id, 4);
        assert_eq!(out.categories[1].name, CATEGORY);
        assert_eq!(out.annotations.len(), 2);
        let p = &out.annotations[1];
        assert_eq!((p.id, p.image_id, p.category_id), (6, 1, 4));
        assert_eq!(p.bbox, persons[0].0);
        assert_ne!(
            cache_file(&cache, &root.join("a.jpg")),
            cache_file(&cache, &root.join("b.jpg"))
        );
    }
}
