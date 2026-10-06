//! Applying pending edits to produce the next dataset version, and moving
//! deleted frames to the trash.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

use crate::coco::{AnnId, CocoAnnotation, CocoFile, ImageId};
use crate::dataset::{Dataset, SPLIT_FIELD};
use crate::edits::{AnnMark, PendingState, PredMark};
use crate::error::{Error, Result};
use crate::geom::BBox;
use crate::review::ReviewState;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    pub deleted_annotations: usize,
    pub confirmed: usize,
    pub replaced: usize,
    pub edited: usize,
    pub added: usize,
    pub accepted: usize,
    pub rejected: usize,
    pub deleted_frames: usize,
    /// Frames moved to another split.
    pub moved_frames: usize,
    /// Edits dropped because the box became invalid (outside the image, < 1 px).
    pub invalid_skipped: usize,
}

#[derive(Clone, Debug)]
pub struct Applied {
    pub coco: CocoFile,
    pub review: ReviewState,
    /// `file_name`s of deleted frames, to move to the trash once the new
    /// annotation file is safely written.
    pub trashed_files: Vec<String>,
    pub summary: Summary,
}

/// Pure function: base dataset + pending edits → next version.
///
/// Order: geometry/category edits, marks (confirm / replace / delete), drawn
/// boxes, accepted predictions (an accepted prediction identical to an
/// existing box keeps that box, even if it was marked for deletion), then
/// whole-frame deletion, which removes everything on the frame. Every
/// remaining frame with a known split gets its `split` field.
pub fn apply(ds: &Dataset, pending: &PendingState, review: &ReviewState) -> Applied {
    let mut coco = ds.coco.clone();
    let mut review = review.clone();
    let mut summary = Summary::default();
    let dims: HashMap<ImageId, (f64, f64)> =
        coco.images.iter().map(|i| (i.id, (f64::from(i.width), f64::from(i.height)))).collect();
    let clip = |image: ImageId, b: &BBox| dims.get(&image).and_then(|&(w, h)| b.clip(w, h));

    let mut deleted: HashSet<AnnId> = HashSet::new();
    for ann in &mut coco.annotations {
        let mut touched = false;
        if let Some(&cat) = pending.categories.get(&ann.id) {
            ann.category_id = cat;
            touched = true;
        }
        if let Some(b) = pending.boxes.get(&ann.id) {
            match clip(ann.image_id, b) {
                Some(b) => set_bbox(ann, b),
                None => summary.invalid_skipped += 1,
            }
            touched = true;
        }
        summary.edited += usize::from(touched);
        match pending.ann_marks.get(&ann.id) {
            Some(AnnMark::Delete) => {
                deleted.insert(ann.id);
            }
            Some(AnnMark::Confirm) => {
                summary.confirmed += 1;
                touched = true;
            }
            Some(AnnMark::Replace(pred)) => {
                match clip(ann.image_id, &pred.bbox) {
                    Some(b) => {
                        set_bbox(ann, b);
                        summary.replaced += 1;
                    }
                    None => summary.invalid_skipped += 1,
                }
                touched = true;
            }
            None => {}
        }
        if touched {
            review.reviewed.insert(ann.id);
        }
    }

    let mut next_id =
        coco.next_annotation_id().max(pending.added.keys().max().map_or(0, |m| m + 1));
    for (&id, new) in &pending.added {
        if pending.ann_marks.get(&id) == Some(&AnnMark::Delete) {
            continue;
        }
        let Some(b) = clip(new.image, &new.bbox) else {
            summary.invalid_skipped += 1;
            continue;
        };
        coco.annotations.push(CocoAnnotation::new(id, new.image, new.category, b));
        review.reviewed.insert(id);
        summary.added += 1;
    }

    let mut accepted: Vec<_> = pending.pred_marks.iter().collect();
    // Deterministic ids regardless of hash order.
    accepted.sort_by(|(a, _), (b, _)| {
        a.image_id
            .cmp(&b.image_id)
            .then(a.bbox.bits().cmp(&b.bbox.bits()))
            .then(a.category_id.cmp(&b.category_id))
    });
    for (pred, mark) in accepted {
        match mark {
            PredMark::Reject => {
                review.reject(pred);
                summary.rejected += 1;
            }
            PredMark::Accept => {
                let Some(b) = clip(pred.image_id, &pred.bbox) else {
                    summary.invalid_skipped += 1;
                    continue;
                };
                let same = coco.annotations.iter().find(|a| {
                    a.image_id == pred.image_id
                        && a.category_id == pred.category_id
                        && a.bbox.bits() == b.bits()
                });
                if let Some(existing) = same {
                    deleted.remove(&existing.id);
                    review.reviewed.insert(existing.id);
                } else {
                    coco.annotations.push(CocoAnnotation::new(
                        next_id,
                        pred.image_id,
                        pred.category_id,
                        b,
                    ));
                    review.reviewed.insert(next_id);
                    next_id += 1;
                }
                summary.accepted += 1;
            }
        }
    }

    for img in &mut coco.images {
        if let Some(tags) = pending.tags.get(&img.id) {
            img.tags = tags.clone();
        }
    }

    let frames = &pending.deleted_frames;
    let trashed_files = coco
        .images
        .iter()
        .filter(|i| frames.contains(&i.id))
        .map(|i| i.file_name.clone())
        .collect();
    summary.deleted_frames = frames.iter().filter(|id| dims.contains_key(id)).count();
    coco.images.retain(|i| !frames.contains(&i.id));
    let before = coco.annotations.len();
    coco.annotations.retain(|a| !frames.contains(&a.image_id) && !deleted.contains(&a.id));
    summary.deleted_annotations = before - coco.annotations.len();

    write_splits(ds, &mut coco, pending, &mut summary);

    let kept: HashSet<AnnId> = coco.annotations.iter().map(|a| a.id).collect();
    review.retain(|a| kept.contains(&a), |i| !frames.contains(&i));
    Applied { coco, review, trashed_files, summary }
}

/// Writes each frame's split (pending moves included) as its `split` field,
/// so the annotation file no longer depends on folder names.
fn write_splits(ds: &Dataset, coco: &mut CocoFile, pending: &PendingState, summary: &mut Summary) {
    for img in &mut coco.images {
        let saved = ds.image_idx(img.id).and_then(|i| ds.split_of(i));
        let split = pending.splits.get(&img.id).copied().or(saved);
        summary.moved_frames += usize::from(split != saved);
        if let Some(s) = split {
            img.extra.insert(SPLIT_FIELD.into(), s.name().into());
        }
    }
}

fn set_bbox(ann: &mut CocoAnnotation, b: BBox) {
    ann.bbox = b;
    ann.area = b.area();
    // A stored score no longer describes the edited geometry.
    ann.extra.remove("score");
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TrashReport {
    pub moved: Vec<PathBuf>,
    /// Files that could not be moved, with the reason (manual cleanup needed).
    pub failed: Vec<(PathBuf, String)>,
}

/// Moves deleted frames from `root` into `trash_dir`, keeping their relative
/// path. Refuses paths escaping `root` and files still used by `retained`.
pub fn move_to_trash(
    root: &Path,
    trash_dir: &Path,
    files: &[String],
    retained: &CocoFile,
) -> TrashReport {
    let still_used: HashSet<&str> = retained.images.iter().map(|i| i.file_name.as_str()).collect();
    let mut report = TrashReport::default();
    for name in files {
        let src = root.join(name);
        let rel = Path::new(name);
        let safe = rel.components().all(|c| matches!(c, Component::Normal(_)));
        if !safe {
            report.failed.push((src, "path is outside the dataset folder".into()));
            continue;
        }
        if still_used.contains(name.as_str()) {
            report.failed.push((src, "image is still used by a retained frame".into()));
            continue;
        }
        if !src.exists() {
            continue;
        }
        let dst = trash_dir.join(rel);
        match move_file(&src, &dst) {
            Ok(()) => report.moved.push(dst),
            Err(e) => report.failed.push((src, e.to_string())),
        }
    }
    report
}

fn move_file(src: &Path, dst: &Path) -> Result<()> {
    if let Some(dir) = dst.parent() {
        fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
    }
    if fs::rename(src, dst).is_ok() {
        return Ok(());
    }
    // Cross-device: copy then remove.
    fs::copy(src, dst).map_err(|e| Error::io(dst, e))?;
    fs::remove_file(src).map_err(|e| Error::io(src, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Split;
    use crate::dataset::test_util::dataset;
    use crate::edits::{EditLog, Op};
    use crate::predictions::PredRef;

    fn b(x: f64) -> BBox {
        BBox::new(x, 10.0, 20.0, 20.0)
    }

    fn pred(image_id: ImageId, bbox: BBox) -> PredRef {
        PredRef { image_id, category_id: 1, bbox, score: 0.8 }
    }

    #[test]
    fn applies_marks_edits_and_frame_deletion() {
        let ds = dataset(3, &[(1, b(0.0)), (1, b(30.0)), (2, b(0.0)), (3, b(0.0))]);
        let mut log = EditLog::new(ds.coco.next_annotation_id());
        log.push(Op::AnnMark { ann: 1, mark: AnnMark::Delete });
        log.push(Op::AnnMark {
            ann: 2,
            mark: AnnMark::Replace(pred(1, BBox::new(31.0, 11.0, 18.0, 18.0))),
        });
        log.push(Op::AnnMark { ann: 3, mark: AnnMark::Confirm });
        log.push(Op::SetCategory { ann: 3, category: 2 });
        log.push(Op::PredMark { pred: pred(2, b(60.0)), mark: PredMark::Accept });
        log.push(Op::PredMark { pred: pred(2, b(40.0)), mark: PredMark::Reject });
        let id = log.alloc_ann_id();
        log.push(Op::AddBox {
            ann: id,
            image: 1,
            category: 1,
            bbox: BBox::new(90.0, 90.0, 20.0, 20.0),
        });
        log.push(Op::FrameDelete { image: 3 });
        log.push(Op::SetTags { image: 2, tags: vec!["blurry".into()] });

        let out = apply(&ds, log.state(), &ReviewState::default());
        let anns: Vec<_> = out
            .coco
            .annotations
            .iter()
            .map(|a| (a.id, a.image_id, a.category_id, a.bbox))
            .collect();
        assert_eq!(
            anns,
            vec![
                (2, 1, 1, BBox::new(31.0, 11.0, 18.0, 18.0)),
                (3, 2, 2, b(0.0)),
                (5, 1, 1, BBox::new(90.0, 90.0, 10.0, 10.0)), // clipped to image
                (6, 2, 1, b(60.0)),
            ]
        );
        assert_eq!(out.coco.images.iter().map(|i| i.id).collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(out.coco.images[1].tags, vec!["blurry"]);
        assert_eq!(out.trashed_files, vec!["images/train/0003.jpg"]);
        assert_eq!(out.review.reviewed.iter().copied().collect::<Vec<_>>(), vec![2, 3, 5, 6]);
        assert!(out.review.is_rejected(&pred(2, b(40.0))));
        assert_eq!(
            out.summary,
            Summary {
                deleted_annotations: 2,
                confirmed: 1,
                replaced: 1,
                edited: 1,
                added: 1,
                accepted: 1,
                rejected: 1,
                deleted_frames: 1,
                moved_frames: 0,
                invalid_skipped: 0
            }
        );
        // The base dataset is untouched.
        assert_eq!(ds.annotations().len(), 4);
    }

    #[test]
    fn accepting_an_identical_box_keeps_it_instead_of_duplicating() {
        let ds = dataset(1, &[(1, b(0.0))]);
        let mut log = EditLog::new(10);
        log.push(Op::AnnMark { ann: 1, mark: AnnMark::Delete });
        log.push(Op::PredMark { pred: pred(1, b(0.0)), mark: PredMark::Accept });
        let out = apply(&ds, log.state(), &ReviewState::default());
        assert_eq!(out.coco.annotations.len(), 1);
        assert_eq!(out.coco.annotations[0].id, 1);
    }

    #[test]
    fn empty_pending_state_only_records_splits() {
        let ds = dataset(2, &[(1, b(0.0)), (2, b(5.0))]);
        let mut out = apply(&ds, &PendingState::default(), &ReviewState::default());
        for img in &mut out.coco.images {
            assert_eq!(img.extra.remove(SPLIT_FIELD), Some("train".into()));
        }
        assert_eq!(out.coco, ds.coco);
        assert_eq!(out.summary, Summary::default());
        assert!(out.trashed_files.is_empty());
    }

    #[test]
    fn splits_are_written_as_fields_without_touching_paths() {
        let mut coco = dataset(3, &[(1, b(0.0))]).coco;
        coco.images[2].file_name = "images/0003.jpg".into(); // no split
        let ds = Dataset::new("/d".into(), coco, Path::new("t")).unwrap().0;
        let mut log = EditLog::new(10);
        log.push(Op::SetSplit { image: 1, split: Some(Split::Test) });
        log.push(Op::SetSplit { image: 3, split: Some(Split::Val) });
        let out = apply(&ds, log.state(), &ReviewState::default());
        let fields: Vec<_> =
            out.coco.images.iter().map(|i| i.extra[SPLIT_FIELD].as_str().unwrap()).collect();
        assert_eq!(fields, ["test", "train", "val"]);
        let names: Vec<_> = out.coco.images.iter().map(|i| i.file_name.as_str()).collect();
        assert_eq!(names, ["images/train/0001.jpg", "images/train/0002.jpg", "images/0003.jpg"]);
        assert_eq!(out.summary.moved_frames, 2);
        let again = Dataset::new("/d".into(), out.coco, Path::new("t")).unwrap().0;
        assert_eq!(again.split_of(0), Some(Split::Test), "the field wins over the path");
    }

    #[test]
    fn trash_moves_files_and_refuses_unsafe_paths() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::create_dir_all(root.join("images/train")).unwrap();
        fs::write(root.join("images/train/a.jpg"), b"x").unwrap();
        fs::write(root.join("images/train/b.jpg"), b"y").unwrap();
        let mut retained = dataset(0, &[]).coco;
        retained.images = dataset(1, &[]).coco.images;
        retained.images[0].file_name = "images/train/b.jpg".into();

        let trash = root.join(".fastlabel/trash/v002");
        let files = [
            "images/train/a.jpg",
            "images/train/b.jpg",
            "../etc/passwd",
            "images/train/missing.jpg",
        ]
        .map(String::from);
        let report = move_to_trash(root, &trash, &files, &retained);
        assert_eq!(report.moved, vec![trash.join("images/train/a.jpg")]);
        assert_eq!(report.failed.len(), 2);
        assert!(!root.join("images/train/a.jpg").exists());
        assert!(root.join("images/train/b.jpg").exists());
        assert_eq!(fs::read(trash.join("images/train/a.jpg")).unwrap(), b"x");
    }
}
