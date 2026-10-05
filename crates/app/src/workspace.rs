//! An open dataset in the UI: project, evaluation, crop list, selection and
//! the review actions. No egui here, so all of it is unit-tested.

use std::path::Path;

use fastlabel_core::matching::{Thresholds, evaluate};
use fastlabel_core::project::SaveReport;
use fastlabel_core::{AnnMark, Evaluation, Op, PredMark, Project};

use crate::crops::{self, Crop, CropId, ViewOptions};

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
    pub thresholds: Thresholds,
    pub opts: ViewOptions,
    pub crops: Vec<Crop>,
    pub selected: usize,
    pub columns: usize,
}

/// Outcome of a user action, for the status line.
pub type ActionResult = Result<(), String>;

impl Workspace {
    pub fn open(
        dir: &Path,
        opts: ViewOptions,
        thresholds: Thresholds,
        columns: usize,
    ) -> anyhow::Result<Self> {
        let project = Project::open(dir)?;
        let mut ws = Self {
            project,
            eval: None,
            thresholds,
            opts,
            crops: Vec::new(),
            selected: 0,
            columns: columns.max(1),
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
        self.eval =
            p.predictions.as_ref().map(|preds| evaluate(&p.dataset, preds, self.thresholds));
        if !self.has_predictions() {
            self.opts.fp_only = false;
            self.opts.fn_only = false;
            if self.opts.sort.needs_predictions() {
                self.opts.sort = crops::SortKey::Id;
            }
        }
        self.crops = crops::build(
            &p.dataset,
            p.predictions.as_ref(),
            self.eval.as_ref(),
            &p.review,
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

    fn record(&mut self, op: Op) -> ActionResult {
        self.project.record(op).map_err(|e| format!("action applied but not journaled: {e}"))
    }

    /// Toggles `mark` on the selected crop and advances: to the next crop, or
    /// for F to the next crop of another frame.
    pub fn mark_selected(&mut self, mark: Mark) -> ActionResult {
        let Some(crop) = self.selected_crop().copied() else { return Ok(()) };
        let Some(op) = Self::op_for(&crop, mark) else {
            return Err("R needs a GT box with a matched prediction".into());
        };
        let result = self.record(op);
        self.selected = if mark == Mark::Frame {
            (self.selected + 1..self.crops.len())
                .find(|&i| self.crops[i].image_id != crop.image_id)
                .unwrap_or(self.selected)
        } else {
            (self.selected + 1).min(self.crops.len() - 1)
        };
        result
    }

    /// Toggles `mark` on every crop of the selected row (each frame once for
    /// F), as one undo step, then moves to the same column of the next row.
    pub fn mark_row(&mut self, mark: Mark) -> ActionResult {
        if self.crops.is_empty() {
            return Ok(());
        }
        let start = self.selected / self.columns * self.columns;
        let end = (start + self.columns).min(self.crops.len());
        let mut ops = Vec::new();
        let mut frames = Vec::new();
        for crop in &self.crops[start..end] {
            if mark == Mark::Frame {
                if frames.contains(&crop.image_id) {
                    continue;
                }
                frames.push(crop.image_id);
            }
            ops.extend(Self::op_for(crop, mark));
        }
        let result = if ops.is_empty() { Ok(()) } else { self.record(Op::Group { ops }) };
        if end < self.crops.len() {
            self.selected = (self.selected + self.columns).min(self.crops.len() - 1);
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

    /// Saves a new version and rebuilds the view.
    pub fn save(&mut self) -> anyhow::Result<Option<SaveReport>> {
        let report = self.project.save()?;
        if report.is_some() {
            self.rebuild();
        }
        Ok(report)
    }
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
        let ws =
            Workspace::open(dir.path(), ViewOptions::default(), Thresholds::default(), columns)
                .unwrap();
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
    fn row_actions_toggle_each_and_frames_once() {
        let (_d, mut ws) = open_fixture(2);
        ws.mark_selected(Mark::Delete).unwrap(); // #1 marked, selection on #2
        ws.mark_row(Mark::Delete).unwrap(); // row 0: toggles #1 off, #2 on
        assert_eq!(ws.marks_of(&ws.crops[0]).mark, None);
        assert_eq!(ws.marks_of(&ws.crops[1]).mark, Some(Mark::Delete));
        assert_eq!(ws.selected, 3, "same column, next row");
        ws.undo().unwrap();
        assert_eq!(ws.marks_of(&ws.crops[0]).mark, Some(Mark::Delete), "row undone in one step");

        ws.select(0);
        ws.mark_row(Mark::Frame).unwrap(); // #1 and #2 share a frame → toggled once
        assert!(ws.marks_of(&ws.crops[0]).frame_deleted);
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
    fn thresholds_change_classification() {
        let (_d, mut ws) = open_fixture(2);
        assert_eq!(ws.crops[0].status, crate::crops::CropStatus::Tp);
        ws.thresholds.match_iou = 0.95; // TP had IoU 0.82
        ws.rebuild();
        assert_eq!(ws.crops[0].status, crate::crops::CropStatus::Fn);
        // The once-matched prediction is now an extra FP crop.
        assert_eq!(ws.crops.iter().filter(|c| c.is_pred()).count(), 2);
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
}
