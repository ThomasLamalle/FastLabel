//! An open dataset in the UI: project, evaluation, crop list, selection and
//! the review actions. No egui here, so all of it is unit-tested.

use std::path::Path;

use fastlabel_core::matching::evaluate;
use fastlabel_core::project::SaveReport;
use fastlabel_core::{AnnMark, Evaluation, Op, PredMark, Project};
use fastlabel_core::{PredRef, Predictions};
use fastlabel_infer::{BatchItem, BatchResult};

use crate::crops::{self, Crop, CropId, ViewOptions};
use crate::metrics::{self, FrameBoxes, Uniformity};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mark {
    /// D: delete a GT / reject a prediction.
    Delete,
    /// G: confirm a GT / accept a prediction.
    Good,
    /// R: replace the GT box with its matched prediction.
    Replace,
    /// F: delete the whole frame.
    Frame,
}

/// What a crop currently shows as pending.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CropMarks {
    pub mark: Option<Mark>,
    pub frame_deleted: bool,
    pub reviewed: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PendingCounts {
    pub delete: usize,
    pub good: usize,
    pub replace: usize,
    pub frames: usize,
    pub other: usize,
}

pub struct Workspace {
    pub project: Project,
    pub eval: Option<Evaluation>,
    pub opts: ViewOptions,
    pub crops: Vec<Crop>,
    pub selected: usize,
    pub columns: usize,
    /// Frame mode / editor state.
    pub frame: crate::frame::FrameState,
    pub uniformity: Uniformity,
    /// Image file dates, read when sorting by date (indexed like images).
    image_dates: Option<Vec<Option<f64>>>,
}

/// Outcome of a user action, for the status line.
pub type ActionResult = Result<(), String>;

impl Workspace {
    pub fn open(dir: &Path, opts: ViewOptions, columns: usize) -> anyhow::Result<Self> {
        let project = Project::open(dir)?;
        let uniformity = Uniformity::load(&project.state_dir().join(metrics::METRICS_FILE));
        let mut ws = Self {
            project,
            eval: None,
            opts,
            crops: Vec::new(),
            selected: 0,
            columns: columns.max(1),
            frame: Default::default(),
            uniformity,
            image_dates: None,
        };
        ws.rebuild();
        Ok(ws)
    }

    pub fn has_predictions(&self) -> bool {
        self.project.predictions.as_ref().is_some_and(|p| !p.is_empty())
    }

    /// Recomputes the evaluation and the crop list, keeping the selected crop
    /// when it still exists.
    pub fn rebuild(&mut self) {
        let keep = self.selected_crop().map(|c| c.id);
        let p = &self.project;
        self.eval = p.predictions.as_ref().map(|preds| evaluate(&p.dataset, preds));
        if !self.has_predictions() {
            self.opts.fp_only = false;
            self.opts.fn_only = false;
            if self.opts.sort.needs_predictions() {
                self.opts.sort = crops::SortKey::Id;
            }
        }
        if self.opts.sort == crops::SortKey::Uniformity && self.uniformity.is_empty() {
            self.opts.sort = crops::SortKey::Id;
        }
        if self.opts.sort == crops::SortKey::ImageDate && self.image_dates.is_none() {
            self.image_dates = Some(file_dates(&p.dataset));
        }
        self.crops = crops::build(
            &p.dataset,
            p.predictions.as_ref(),
            self.eval.as_ref(),
            &p.review,
            &self.uniformity,
            self.image_dates.as_deref(),
            &self.opts,
        );
        self.selected = keep
            .and_then(|id| self.index_of(id))
            .unwrap_or(self.selected)
            .min(self.crops.len().saturating_sub(1));
    }

    pub fn selected_crop(&self) -> Option<&Crop> {
        self.crops.get(self.selected)
    }

    pub fn index_of(&self, id: CropId) -> Option<usize> {
        self.crops.iter().position(|c| c.id == id)
    }

    pub fn select(&mut self, i: usize) {
        self.selected = i.min(self.crops.len().saturating_sub(1));
    }

    /// Grid movement: `dx` crops sideways, `dy` rows; clamped to the list.
    pub fn move_selection(&mut self, dx: isize, dy: isize) {
        if self.crops.is_empty() {
            return;
        }
        let target = self.selected as isize + dx + dy * self.columns as isize;
        let last = self.crops.len() as isize - 1;
        // Moving down from the last partial row lands on the last crop.
        self.selected = target.clamp(0, last) as usize;
    }

    pub fn marks_of(&self, crop: &Crop) -> CropMarks {
        let st = self.project.log.state();
        let review = &self.project.review;
        let (mark, reviewed) = match crop.id {
            CropId::Gt(ann) => (
                st.ann_mark(ann).map(|m| match m {
                    AnnMark::Delete => Mark::Delete,
                    AnnMark::Confirm => Mark::Good,
                    AnnMark::Replace(_) => Mark::Replace,
                }),
                review.is_reviewed(ann),
            ),
            CropId::Pred(p) => (
                st.pred_mark(&p).map(|m| match m {
                    PredMark::Reject => Mark::Delete,
                    PredMark::Accept => Mark::Good,
                }),
                review.is_rejected(&p),
            ),
        };
        CropMarks { mark, frame_deleted: st.is_frame_deleted(crop.image_id), reviewed }
    }

    pub fn pending_counts(&self) -> PendingCounts {
        let st = self.project.log.state();
        let mut c = PendingCounts { frames: st.deleted_frames.len(), ..Default::default() };
        for m in st.ann_marks.values() {
            match m {
                AnnMark::Delete => c.delete += 1,
                AnnMark::Confirm => c.good += 1,
                AnnMark::Replace(_) => c.replace += 1,
            }
        }
        for m in st.pred_marks.values() {
            match m {
                PredMark::Reject => c.delete += 1,
                PredMark::Accept => c.good += 1,
            }
        }
        c.other = st.added.len() + st.boxes.len() + st.categories.len() + st.tags.len();
        c
    }

    fn op_for(crop: &Crop, mark: Mark) -> Option<Op> {
        match (mark, crop.id) {
            (Mark::Delete, CropId::Gt(ann)) => Some(Op::AnnMark { ann, mark: AnnMark::Delete }),
            (Mark::Delete, CropId::Pred(pred)) => {
                Some(Op::PredMark { pred, mark: PredMark::Reject })
            }
            (Mark::Good, CropId::Gt(ann)) => Some(Op::AnnMark { ann, mark: AnnMark::Confirm }),
            (Mark::Good, CropId::Pred(pred)) => Some(Op::PredMark { pred, mark: PredMark::Accept }),
            (Mark::Replace, CropId::Gt(ann)) => {
                crop.pred.map(|p| Op::AnnMark { ann, mark: AnnMark::Replace(p) })
            }
            (Mark::Replace, CropId::Pred(_)) => None,
            (Mark::Frame, _) => Some(Op::FrameDelete { image: crop.image_id }),
        }
    }

    pub(crate) fn record(&mut self, op: Op) -> ActionResult {
        self.project.record(op).map_err(|e| format!("action applied but not journaled: {e}"))
    }

    /// Ops toggling the deletion of a frame. Marking a frame for deletion
    /// also clears the D / G / R marks of its boxes and predictions (they
    /// would be meaningless), in the same undo step.
    pub(crate) fn frame_delete_ops(&self, image_id: fastlabel_core::ImageId) -> Vec<Op> {
        let mut ops = vec![Op::FrameDelete { image: image_id }];
        let st = self.project.log.state();
        if st.is_frame_deleted(image_id) {
            return ops;
        }
        let ds = &self.project.dataset;
        if let Some(idx) = ds.image_idx(image_id) {
            let anns = ds.annotations_of(idx).map(|a| a.id);
            let added = st.added.iter().filter(|(_, b)| b.image == image_id).map(|(id, _)| *id);
            for ann in anns.chain(added) {
                if let Some(mark) = st.ann_mark(ann) {
                    ops.push(Op::AnnMark { ann, mark });
                }
            }
        }
        let mut preds: Vec<_> =
            st.pred_marks.iter().filter(|(p, _)| p.image_id == image_id).collect();
        preds.sort_by_key(|(p, _)| p.bbox.bits());
        ops.extend(preds.into_iter().map(|(pred, mark)| Op::PredMark { pred: *pred, mark: *mark }));
        ops
    }

    fn ops_for(&self, crop: &Crop, mark: Mark) -> Result<Vec<Op>, String> {
        if mark == Mark::Frame {
            return Ok(self.frame_delete_ops(crop.image_id));
        }
        let op = Self::op_for(crop, mark).ok_or("R needs a GT box paired with a prediction")?;
        // Deciding on a box of a frame marked F keeps the frame after all.
        if self.project.log.state().is_frame_deleted(crop.image_id) {
            return Ok(vec![Op::FrameDelete { image: crop.image_id }, op]);
        }
        Ok(vec![op])
    }

    fn record_all(&mut self, mut ops: Vec<Op>) -> ActionResult {
        match ops.len() {
            0 => Ok(()),
            1 => self.record(ops.pop().expect("one op")),
            _ => self.record(Op::Group { ops }),
        }
    }

    /// A crop still waiting for a decision: no mark, frame not deleted.
    fn is_pending(&self, crop: &Crop) -> bool {
        let m = self.marks_of(crop);
        m.mark.is_none() && !m.frame_deleted
    }

    /// Next crop after `from` still waiting for a decision (for F, also on
    /// another frame); stays on `from` when there is none.
    fn next_pending(&self, from: usize, skip_image: Option<fastlabel_core::ImageId>) -> usize {
        (from + 1..self.crops.len())
            .find(|&i| {
                let c = &self.crops[i];
                skip_image != Some(c.image_id) && self.is_pending(c)
            })
            .unwrap_or(from)
    }

    /// Toggles `mark` on the selected crop, then advances to the next crop
    /// still waiting for a decision.
    pub fn mark_selected(&mut self, mark: Mark) -> ActionResult {
        let Some(crop) = self.selected_crop().copied() else { return Ok(()) };
        let ops = self.ops_for(&crop, mark)?;
        let result = self.record_all(ops);
        let skip = (mark == Mark::Frame).then_some(crop.image_id);
        self.selected = self.next_pending(self.selected, skip);
        result
    }

    /// Toggles `mark` on each crop from the selection to the end of its row
    /// (each frame once for F; D / G / R unmark frames marked F),
    /// as one undo step, then selects the first crop of the next row.
    pub fn mark_row(&mut self, mark: Mark) -> ActionResult {
        if self.crops.is_empty() {
            return Ok(());
        }
        let row_end = (self.selected / self.columns + 1) * self.columns;
        let end = row_end.min(self.crops.len());
        let mut ops = Vec::new();
        let mut frames = Vec::new();
        for i in self.selected..end {
            let crop = self.crops[i];
            let first_of_frame = !frames.contains(&crop.image_id);
            frames.push(crop.image_id);
            if mark == Mark::Frame && !first_of_frame {
                continue;
            }
            if let Ok(mut o) = self.ops_for(&crop, mark) {
                // Unmark a frame only once, even with several crops on it.
                if !first_of_frame {
                    o.retain(|op| !matches!(op, Op::FrameDelete { .. }));
                }
                ops.extend(o);
            }
        }
        let result = self.record_all(ops);
        if end < self.crops.len() {
            self.selected = end;
        }
        result
    }

    /// Undoes the last action and selects the crop it affected.
    pub fn undo(&mut self) -> ActionResult {
        let op = self.project.undo().map_err(|e| e.to_string())?;
        self.select_target(op.as_ref());
        Ok(())
    }

    pub fn redo(&mut self) -> ActionResult {
        let op = self.project.redo().map_err(|e| e.to_string())?;
        self.select_target(op.as_ref());
        Ok(())
    }

    fn select_target(&mut self, op: Option<&Op>) {
        let found = match op {
            Some(Op::AnnMark { ann, .. }) => self.index_of(CropId::Gt(*ann)),
            Some(Op::PredMark { pred, .. }) => self.index_of(CropId::Pred(*pred)),
            Some(Op::FrameDelete { image }) => self.crops.iter().position(|c| c.image_id == *image),
            Some(Op::Group { ops }) => {
                return self.select_target(ops.first());
            }
            _ => None,
        };
        if let Some(i) = found {
            self.selected = i;
        }
    }

    /// Frames to run the model on: all images except frames pending deletion.
    pub fn inference_items(&self) -> Vec<BatchItem> {
        let ds = &self.project.dataset;
        let st = self.project.log.state();
        ds.images()
            .iter()
            .enumerate()
            .filter(|(_, img)| !st.is_frame_deleted(img.id))
            .map(|(key, img)| BatchItem { key, path: ds.image_path(img) })
            .collect()
    }

    /// Frames the model has not been run on yet (e.g. newly added images).
    pub fn unpredicted_items(&self) -> Vec<BatchItem> {
        let preds = self.project.predictions.as_ref();
        self.inference_items()
            .into_iter()
            .filter(|it| preds.is_none_or(|p| p.of(it.key).is_none()))
            .collect()
    }

    pub fn frame_item(&self, image_idx: usize) -> BatchItem {
        let ds = &self.project.dataset;
        BatchItem { key: image_idx, path: ds.image_path(&ds.images()[image_idx]) }
    }

    /// Stores model output as the predictions (all frames, or one frame) and
    /// rebuilds. A cancelled full run keeps the previous predictions.
    pub fn apply_detections(
        &mut self,
        all: bool,
        model: &str,
        confidence: f64,
        class_names: &[String],
        result: BatchResult,
    ) -> Result<String, String> {
        if result.cancelled && all {
            return Err("Model run cancelled; previous predictions kept".into());
        }
        let ds = &self.project.dataset;
        let mapping: Vec<_> = class_names
            .iter()
            .map(|n| {
                ds.category_by_name(n)
                    .ok_or_else(|| format!("model class '{n}' has no matching dataset category"))
            })
            .collect::<Result<_, _>>()?;
        let to_refs = |idx: usize,
                       dets: &[fastlabel_infer::Detection]|
         -> Result<Vec<PredRef>, String> {
            let image_id = ds.images()[idx].id;
            dets.iter()
                .map(|d| {
                    let category_id = *mapping
                        .get(d.class)
                        .ok_or_else(|| format!("unknown model class index {}", d.class))?;
                    Ok(PredRef { image_id, category_id, bbox: d.bbox, score: f64::from(d.score) })
                })
                .collect()
        };
        let boxes: usize = result.detections.iter().map(|(_, d)| d.len()).sum();
        if all {
            let mut preds = Predictions::empty(model, confidence);
            for (idx, dets) in &result.detections {
                preds.set_image(*idx, to_refs(*idx, dets)?);
            }
            self.project.set_predictions(preds).map_err(|e| e.to_string())?;
            self.opts.sort = crops::SortKey::Iou;
            self.opts.descending = false;
        } else if let Some((idx, dets)) = result.detections.first() {
            let refs = to_refs(*idx, dets)?;
            self.project
                .set_image_predictions(*idx, refs, model, confidence)
                .map_err(|e| e.to_string())?;
        }
        self.rebuild();
        let mut msg = format!("{boxes} predictions on {} frames", result.detections.len());
        if !result.failed.is_empty() {
            msg.push_str(&format!(
                ", {} frames failed ({})",
                result.failed.len(),
                result.failed[0].1
            ));
        }
        Ok(msg)
    }

    /// Boxes (GT and unmatched predictions) without a uniformity value yet,
    /// grouped by frame.
    pub fn missing_uniformity(&self) -> Vec<FrameBoxes> {
        let ds = &self.project.dataset;
        let preds = self.project.predictions.as_ref();
        ds.images()
            .iter()
            .enumerate()
            .filter_map(|(idx, img)| {
                let gt = ds.annotations_of(idx).map(|a| a.bbox);
                let pr = preds.and_then(|p| p.of(idx)).unwrap_or(&[]).iter().map(|p| p.bbox);
                let boxes: Vec<_> = gt
                    .chain(pr)
                    .map(|b| (metrics::key(&img.file_name, &b), b))
                    .filter(|(k, _)| self.uniformity.get(*k).is_none())
                    .collect();
                (!boxes.is_empty()).then(|| FrameBoxes { path: ds.image_path(img), boxes })
            })
            .collect()
    }

    /// Stores computed values, persists them and re-sorts.
    pub fn add_uniformity(
        &mut self,
        values: std::collections::HashMap<u64, f32>,
    ) -> anyhow::Result<()> {
        self.uniformity.extend(values);
        self.uniformity.save(&self.project.state_dir().join(metrics::METRICS_FILE))?;
        self.rebuild();
        Ok(())
    }

    /// Saves a new version and rebuilds the view.
    pub fn save(&mut self) -> anyhow::Result<Option<SaveReport>> {
        let ds = &self.project.dataset;
        let current = ds.images().get(self.frame.image_idx).map(|i| i.id);
        let report = self.project.save()?;
        if report.is_some() {
            let ds = &self.project.dataset;
            let n = ds.images().len();
            // Deleted frames shift image indices: stay on the same image.
            self.image_dates = None;
            self.frame.image_idx = current
                .and_then(|id| ds.image_idx(id))
                .unwrap_or(self.frame.image_idx)
                .min(n.saturating_sub(1));
            self.rebuild();
        }
        Ok(report)
    }
}

/// File date of each image: creation time when the filesystem records it
/// (a copied file gets a new one), else modification time.
fn file_dates(ds: &fastlabel_core::Dataset) -> Vec<Option<f64>> {
    ds.images()
        .iter()
        .map(|img| {
            let meta = std::fs::metadata(ds.image_path(img)).ok()?;
            let t = meta.created().or_else(|_| meta.modified()).ok()?;
            Some(t.duration_since(std::time::UNIX_EPOCH).ok()?.as_secs_f64())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastlabel_core::coco::CocoFile;
    use fastlabel_core::predictions::Predictions;
    use std::fs;

    /// Writes the crops fixture as a dataset folder.
    fn open_fixture(columns: usize) -> (tempfile::TempDir, Workspace) {
        let (ds, preds, _) = crate::crops::tests::fixture();
        let dir = tempfile::tempdir().unwrap();
        let coco: &CocoFile = &ds.coco;
        coco.save(&dir.path().join("annotations.v001.json")).unwrap();
        Predictions::to_file(&preds, &ds).save(&dir.path().join("predictions.json")).unwrap();
        for f in ["images/train/a.jpg", "images/val/b.jpg"] {
            let p = dir.path().join(f);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(p, b"x").unwrap();
        }
        let ws = Workspace::open(dir.path(), ViewOptions::default(), columns).unwrap();
        (dir, ws)
    }

    #[test]
    fn marks_toggle_advance_and_undo_reselects() {
        let (_d, mut ws) = open_fixture(2);
        assert_eq!(ws.crops.len(), 4); // #1 #2 #3 P150
        ws.mark_selected(Mark::Delete).unwrap();
        assert_eq!(ws.selected, 1);
        assert_eq!(ws.marks_of(&ws.crops[0]).mark, Some(Mark::Delete));
        ws.mark_selected(Mark::Good).unwrap();
        assert_eq!(ws.marks_of(&ws.crops[1]).mark, Some(Mark::Good));

        ws.undo().unwrap();
        assert_eq!(ws.selected, 1);
        assert_eq!(ws.marks_of(&ws.crops[1]).mark, None);
        ws.redo().unwrap();
        assert_eq!(ws.marks_of(&ws.crops[1]).mark, Some(Mark::Good));

        // R only works on a TP crop and does not advance otherwise.
        ws.select(1);
        assert!(ws.mark_selected(Mark::Replace).is_err());
        assert_eq!(ws.selected, 1);
        ws.select(0);
        ws.mark_selected(Mark::Replace).unwrap();
        assert_eq!(ws.marks_of(&ws.crops[0]).mark, Some(Mark::Replace));
        // Pred crops: D rejects, G accepts.
        ws.select(3);
        ws.mark_selected(Mark::Good).unwrap();
        assert_eq!(ws.marks_of(&ws.crops[3]).mark, Some(Mark::Good));
        assert_eq!(ws.selected, 3, "last crop stays selected");
    }

    #[test]
    fn frame_mark_covers_siblings_and_skips_them() {
        let (_d, mut ws) = open_fixture(2);
        ws.mark_selected(Mark::Frame).unwrap();
        // #1, #2 and P150 share image 1; next crop of another frame is #3.
        assert_eq!(ws.selected, 2);
        let flagged: Vec<bool> = ws.crops.iter().map(|c| ws.marks_of(c).frame_deleted).collect();
        assert_eq!(flagged, vec![true, true, false, true]);
        let counts = ws.pending_counts();
        assert_eq!(counts.frames, 1);
    }

    #[test]
    fn row_actions_go_from_selection_to_row_end_then_next_row_start() {
        let (_d, mut ws) = open_fixture(3); // rows: [#1 #2 #3] [P150]
        ws.select(1);
        ws.mark_row(Mark::Delete).unwrap(); // #2 and #3 only
        assert_eq!(ws.marks_of(&ws.crops[0]).mark, None);
        assert_eq!(ws.marks_of(&ws.crops[1]).mark, Some(Mark::Delete));
        assert_eq!(ws.marks_of(&ws.crops[2]).mark, Some(Mark::Delete));
        assert_eq!(ws.selected, 3, "first crop of the next row");
        ws.select(1);
        ws.mark_row(Mark::Delete).unwrap(); // toggles both off again
        assert!(ws.crops.iter().all(|c| ws.marks_of(c).mark.is_none()));
        ws.undo().unwrap();
        assert_eq!(ws.marks_of(&ws.crops[2]).mark, Some(Mark::Delete), "row undone in one step");

        // F on a row toggles each frame once (#1 and #2 share a frame).
        ws.select(0);
        ws.mark_row(Mark::Frame).unwrap();
        assert!(ws.crops.iter().all(|c| ws.marks_of(c).frame_deleted));
        assert_eq!(ws.selected, 3);
    }

    #[test]
    fn auto_advance_skips_decided_crops() {
        let (_d, mut ws) = open_fixture(2); // #1 #2 #3 P150; #1 #2 P150 share frame 1
        ws.select(2);
        ws.mark_selected(Mark::Good).unwrap(); // decide #3 first
        ws.select(3);
        ws.mark_selected(Mark::Frame).unwrap(); // frame 1: #1 #2 P150
        ws.select(0);
        assert_eq!(ws.next_pending(0, None), 0, "nothing left to decide: stay");
        ws.undo().unwrap(); // frame 1 back
        ws.select(0);
        ws.mark_selected(Mark::Delete).unwrap();
        assert_eq!(ws.selected, 1);
        ws.mark_selected(Mark::Delete).unwrap();
        assert_eq!(ws.selected, 3, "skips #3, already decided");
    }

    #[test]
    fn frame_deletion_and_box_marks_exclude_each_other() {
        let (_d, mut ws) = open_fixture(2);
        ws.mark_selected(Mark::Good).unwrap(); // #1 confirmed
        ws.select(3);
        ws.mark_selected(Mark::Good).unwrap(); // P150 accepted
        ws.select(0);
        ws.mark_selected(Mark::Frame).unwrap();
        assert!(ws.marks_of(&ws.crops[0]).frame_deleted);
        assert_eq!(ws.marks_of(&ws.crops[0]).mark, None, "F cleared the G mark");
        assert_eq!(ws.marks_of(&ws.crops[3]).mark, None, "and the accepted prediction");

        // D on a box of the deleted frame: applied, and the frame is kept.
        ws.select(1);
        ws.mark_selected(Mark::Delete).unwrap();
        assert_eq!(ws.marks_of(&ws.crops[1]).mark, Some(Mark::Delete));
        assert!(!ws.marks_of(&ws.crops[1]).frame_deleted);
        ws.undo().unwrap(); // one step: frame deleted again, no D
        assert!(ws.marks_of(&ws.crops[1]).frame_deleted);
        assert_eq!(ws.marks_of(&ws.crops[1]).mark, None);

        // A row action unmarks the frame once and marks every crop.
        ws.select(0);
        ws.mark_row(Mark::Delete).unwrap();
        assert!(!ws.marks_of(&ws.crops[0]).frame_deleted);
        assert_eq!(ws.marks_of(&ws.crops[0]).mark, Some(Mark::Delete));
        assert_eq!(ws.marks_of(&ws.crops[1]).mark, Some(Mark::Delete));
    }

    #[test]
    fn navigation_clamps() {
        let (_d, mut ws) = open_fixture(3);
        ws.move_selection(0, 1);
        assert_eq!(ws.selected, 3);
        ws.move_selection(0, 1);
        assert_eq!(ws.selected, 3);
        ws.move_selection(-1, -1);
        assert_eq!(ws.selected, 0);
        ws.move_selection(1, 0);
        assert_eq!(ws.selected, 1);
    }

    #[test]
    fn save_applies_and_rebuilds_keeping_selection() {
        let (dir, mut ws) = open_fixture(2);
        ws.mark_selected(Mark::Delete).unwrap(); // delete #1, select #2
        ws.select(3);
        ws.mark_selected(Mark::Good).unwrap(); // accept P150
        ws.select(1);
        let report = ws.save().unwrap().unwrap();
        assert_eq!(report.summary.deleted_annotations, 1);
        assert_eq!(report.summary.accepted, 1);
        assert!(dir.path().join("annotations.v002.json").exists());
        assert_eq!(ws.selected_crop().unwrap().id, CropId::Gt(2));
        assert!(ws.crops.iter().all(|c| ws.marks_of(c).mark.is_none()));
        // The accepted prediction is now a reviewed TP annotation.
        let new = ws.crops.iter().find(|c| c.ann_id() == Some(4)).unwrap();
        assert!(ws.marks_of(new).reviewed);
    }

    #[test]
    fn detections_become_predictions_with_category_mapping() {
        use fastlabel_infer::Detection;
        let (_d, mut ws) = open_fixture(2);
        assert_eq!(ws.inference_items().len(), 2);
        let det = |x| Detection {
            class: 0,
            bbox: fastlabel_core::BBox::new(x, 10.0, 20.0, 20.0),
            score: 0.6,
        };
        let result = BatchResult { detections: vec![(1, vec![det(100.0)])], ..Default::default() };
        let msg = ws.apply_detections(false, "m.onnx", 0.2, &["BIB".into()], result).unwrap();
        assert!(msg.starts_with("1 predictions"));
        let preds = ws.project.predictions.as_ref().unwrap();
        assert_eq!(preds.of(1).unwrap().len(), 1, "frame 2 now predicted");
        assert_eq!(preds.of(0).unwrap().len(), 2, "frame 1 untouched");

        let bad = BatchResult { detections: vec![(0, vec![det(0.0)])], ..Default::default() };
        assert!(ws.apply_detections(true, "m", 0.2, &["car".into()], bad).is_err());
        let cancelled = BatchResult { cancelled: true, ..Default::default() };
        assert!(ws.apply_detections(true, "m", 0.2, &["bib".into()], cancelled).is_err());
        assert_eq!(ws.project.predictions.as_ref().unwrap().of(0).unwrap().len(), 2, "kept");

        let full = BatchResult {
            detections: vec![(0, vec![]), (1, vec![det(5.0)])],
            ..Default::default()
        };
        ws.apply_detections(true, "m2", 0.3, &["bib".into()], full).unwrap();
        let preds = ws.project.predictions.as_ref().unwrap();
        assert_eq!(preds.model, "m2");
        assert_eq!(preds.of(0).unwrap().len(), 0);
        assert_eq!(ws.opts.sort, crops::SortKey::Iou);
    }

    #[test]
    fn replace_and_reject_persist_through_save_and_hide_reviewed() {
        let (_d, mut ws) = open_fixture(2);
        ws.mark_selected(Mark::Replace).unwrap(); // #1 takes the prediction box
        ws.select(3);
        ws.mark_selected(Mark::Delete).unwrap(); // reject P150
        ws.save().unwrap().unwrap();
        let replaced = ws.project.dataset.annotation(1).unwrap().bbox;
        assert_eq!(replaced, fastlabel_core::BBox::new(1.0, 0.0, 10.0, 10.0));
        assert_eq!(ws.crops[0].iou, Some(1.0));

        ws.opts.hide_reviewed = true;
        ws.rebuild();
        let left: Vec<_> = ws.crops.iter().map(|c| c.id).collect();
        assert_eq!(left, vec![CropId::Gt(2), CropId::Gt(3)]);
    }

    #[test]
    fn date_sort_puts_newest_images_first_and_unpredicted_frames_are_listed() {
        let (dir, mut ws) = open_fixture(2);
        // Make the second image clearly newer.
        std::thread::sleep(std::time::Duration::from_millis(30));
        let b = dir.path().join("images/val/b.jpg");
        std::fs::remove_file(&b).unwrap();
        std::fs::write(&b, b"y").unwrap();
        ws.opts.sort = crops::SortKey::ImageDate;
        ws.opts.descending = true;
        ws.rebuild();
        assert_eq!(ws.crops[0].id, CropId::Gt(3), "image b's crop first");
        assert!(ws.crops.iter().all(|c| c.date.is_some()));

        let items = ws.unpredicted_items();
        assert_eq!(items.iter().map(|i| i.key).collect::<Vec<_>>(), vec![1]);
    }
}
