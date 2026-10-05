//! Frame-level editing on top of the workspace: the effective boxes of a
//! frame (base annotations + pending edits), and the editing actions.
//! Every action is a pending [`Op`]; nothing changes on disk until save.

use fastlabel_core::{AnnId, AnnMark, BBox, CategoryId, Op, PredMark, PredRef};

use crate::crops::CropId;
use crate::workspace::{ActionResult, Mark, Workspace};

/// Selected item in the frame editor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sel {
    Gt(AnnId),
    Pred(PredRef),
}

#[derive(Clone, Debug, Default)]
pub struct FrameState {
    /// Dataset image index shown in frame mode / the editor.
    pub image_idx: usize,
    pub sel: Option<Sel>,
    /// Category for new boxes (`None` = first category).
    pub draw_category: Option<CategoryId>,
}

/// A GT box as currently edited.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EditBox {
    pub id: AnnId,
    pub category: CategoryId,
    pub bbox: BBox,
    pub mark: Option<Mark>,
    /// Drawn in this session (not saved yet).
    pub added: bool,
}

fn ann_mark(m: AnnMark) -> Mark {
    match m {
        AnnMark::Delete => Mark::Delete,
        AnnMark::Confirm => Mark::Good,
        AnnMark::Replace(_) => Mark::Replace,
    }
}

impl Workspace {
    fn image_dims(&self, image_idx: usize) -> (f64, f64) {
        let img = &self.project.dataset.images()[image_idx];
        (f64::from(img.width), f64::from(img.height))
    }

    pub fn frame_boxes(&self, image_idx: usize) -> Vec<EditBox> {
        let ds = &self.project.dataset;
        let st = self.project.log.state();
        let image_id = ds.images()[image_idx].id;
        let mut out: Vec<EditBox> = ds
            .annotations_of(image_idx)
            .map(|a| EditBox {
                id: a.id,
                category: st.categories.get(&a.id).copied().unwrap_or(a.category_id),
                bbox: st.boxes.get(&a.id).copied().unwrap_or(a.bbox),
                mark: st.ann_mark(a.id).map(ann_mark),
                added: false,
            })
            .collect();
        out.extend(st.added.iter().filter(|(_, b)| b.image == image_id).map(|(&id, b)| EditBox {
            id,
            category: b.category,
            bbox: b.bbox,
            mark: st.ann_mark(id).map(ann_mark),
            added: true,
        }));
        out
    }

    /// Predictions of a frame with their pending mark (D = reject, G = accept).
    pub fn frame_preds(&self, image_idx: usize) -> Vec<(PredRef, Option<Mark>)> {
        let st = self.project.log.state();
        let Some(preds) = self.project.predictions.as_ref().and_then(|p| p.of(image_idx)) else {
            return Vec::new();
        };
        preds
            .iter()
            .map(|p| {
                let mark = st.pred_mark(p).map(|m| match m {
                    PredMark::Reject => Mark::Delete,
                    PredMark::Accept => Mark::Good,
                });
                (*p, mark)
            })
            .collect()
    }

    /// D / G on the editor selection.
    pub fn mark_sel(&mut self, mark: Mark) -> ActionResult {
        let op = match (self.frame.sel, mark) {
            (Some(Sel::Gt(ann)), Mark::Delete) => Op::AnnMark { ann, mark: AnnMark::Delete },
            (Some(Sel::Gt(ann)), Mark::Good) => Op::AnnMark { ann, mark: AnnMark::Confirm },
            (Some(Sel::Pred(pred)), Mark::Delete) => Op::PredMark { pred, mark: PredMark::Reject },
            (Some(Sel::Pred(pred)), Mark::Good) => Op::PredMark { pred, mark: PredMark::Accept },
            (None, _) => return Err("Select a box first".into()),
            _ => return Ok(()),
        };
        self.record(op)
    }

    /// Moves / resizes a GT box (clipped to the image).
    pub fn set_box(&mut self, id: AnnId, bbox: BBox) -> ActionResult {
        let (w, h) = self.image_dims(self.frame.image_idx);
        let bbox = bbox.clip(w, h).ok_or("Box too small or outside the image")?;
        self.record(Op::SetBox { ann: id, bbox })
    }

    /// Draws a new box on the current frame and selects it.
    pub fn add_box(&mut self, bbox: BBox) -> Result<AnnId, String> {
        let (w, h) = self.image_dims(self.frame.image_idx);
        let bbox = bbox.clip(w, h).ok_or("Box too small or outside the image")?;
        let category = self.draw_category().ok_or("The dataset has no category")?;
        let image = self.project.dataset.images()[self.frame.image_idx].id;
        let ann = self.project.log.alloc_ann_id();
        self.record(Op::AddBox { ann, image, category, bbox })?;
        self.frame.sel = Some(Sel::Gt(ann));
        Ok(ann)
    }

    pub fn draw_category(&self) -> Option<CategoryId> {
        self.frame.draw_category.or_else(|| self.project.dataset.categories().first().map(|c| c.id))
    }

    /// Keys 1-9: category of the selected box, also used for new boxes.
    pub fn set_category_index(&mut self, n: usize) -> ActionResult {
        let cat =
            self.project.dataset.categories().get(n).map(|c| c.id).ok_or("No such category")?;
        self.frame.draw_category = Some(cat);
        match self.frame.sel {
            Some(Sel::Gt(ann)) => self.record(Op::SetCategory { ann, category: cat }),
            _ => Ok(()),
        }
    }

    /// Effective tags of a frame (pending edit or saved value).
    pub fn tags_of(&self, image_idx: usize) -> Vec<String> {
        let img = &self.project.dataset.images()[image_idx];
        self.project.log.state().tags.get(&img.id).cloned().unwrap_or_else(|| img.tags.clone())
    }

    /// Sets tags from comma-separated text; no-op if unchanged.
    pub fn set_tags_text(&mut self, image_idx: usize, text: &str) -> ActionResult {
        let tags = fastlabel_core::edits::normalize_tags(
            &text.split(',').map(String::from).collect::<Vec<_>>(),
        );
        if tags == self.tags_of(image_idx) {
            return Ok(());
        }
        let image = self.project.dataset.images()[image_idx].id;
        self.record(Op::SetTags { image, tags })
    }

    pub fn toggle_frame_delete(&mut self) -> ActionResult {
        let image = self.project.dataset.images()[self.frame.image_idx].id;
        self.record(Op::FrameDelete { image })
    }

    pub fn is_frame_deleted(&self, image_idx: usize) -> bool {
        let image = self.project.dataset.images()[image_idx].id;
        self.project.log.state().is_frame_deleted(image)
    }

    /// Previous / next frame (`dir` = ±1), optionally only frames with GT.
    /// Stops at the ends (returns `None`).
    pub fn step_frame(&self, dir: isize, annotated_only: bool) -> Option<usize> {
        let n = self.project.dataset.images().len() as isize;
        let mut i = self.frame.image_idx as isize + dir;
        while (0..n).contains(&i) {
            if !annotated_only || self.project.dataset.annotations_of(i as usize).next().is_some() {
                return Some(i as usize);
            }
            i += dir;
        }
        None
    }

    pub fn go_to_frame(&mut self, image_idx: usize) {
        if image_idx != self.frame.image_idx {
            self.frame.image_idx = image_idx;
            self.frame.sel = None;
        }
    }

    /// Opens the selected crop's frame with its box selected.
    pub fn focus_selected_crop(&mut self) {
        if let Some(crop) = self.selected_crop().copied() {
            self.frame.image_idx = crop.image_idx;
            self.frame.sel = Some(match crop.id {
                CropId::Gt(a) => Sel::Gt(a),
                CropId::Pred(p) => Sel::Pred(p),
            });
            self.frame.draw_category = Some(crop.category_id);
        }
    }

    /// Selects, in the mosaic, the first crop of the current frame.
    pub fn select_crop_of_frame(&mut self) {
        let want = self.frame.sel;
        let idx = self.frame.image_idx;
        let exact = want.and_then(|s| {
            self.index_of(match s {
                Sel::Gt(a) => CropId::Gt(a),
                Sel::Pred(p) => CropId::Pred(p),
            })
        });
        if let Some(i) = exact.or_else(|| self.crops.iter().position(|c| c.image_idx == idx)) {
            self.selected = i;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crops::ViewOptions;
    use fastlabel_core::matching::Thresholds;
    use std::fs;

    fn open() -> (tempfile::TempDir, Workspace) {
        let (ds, preds, _) = crate::crops::tests::fixture();
        let dir = tempfile::tempdir().unwrap();
        ds.coco.save(&dir.path().join("annotations.v001.json")).unwrap();
        preds.to_file(&ds).save(&dir.path().join("predictions.json")).unwrap();
        for f in ["images/train/a.jpg", "images/val/b.jpg"] {
            let p = dir.path().join(f);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(p, b"x").unwrap();
        }
        let ws =
            Workspace::open(dir.path(), ViewOptions::default(), Thresholds::default(), 2).unwrap();
        (dir, ws)
    }

    #[test]
    fn edits_show_in_frame_boxes_and_apply_on_save() {
        let (_d, mut ws) = open();
        ws.select(0);
        ws.focus_selected_crop();
        assert_eq!(ws.frame.sel, Some(Sel::Gt(1)));
        ws.set_box(1, BBox::new(-5.0, 0.0, 20.0, 10.0)).unwrap(); // clipped
        let id = ws.add_box(BBox::new(150.0, 50.0, 30.0, 20.0)).unwrap();
        assert_eq!(ws.frame.sel, Some(Sel::Gt(id)));
        assert!(ws.add_box(BBox::new(500.0, 0.0, 5.0, 5.0)).is_err(), "outside the image");

        let boxes = ws.frame_boxes(0);
        assert_eq!(boxes.len(), 3);
        assert_eq!(boxes[0].bbox, BBox::new(0.0, 0.0, 15.0, 10.0));
        assert!(boxes[2].added);

        ws.frame.sel = Some(Sel::Gt(2));
        ws.mark_sel(Mark::Delete).unwrap();
        assert_eq!(ws.frame_boxes(0)[1].mark, Some(Mark::Delete));
        let preds = ws.frame_preds(0);
        ws.frame.sel = Some(Sel::Pred(preds[1].0));
        ws.mark_sel(Mark::Good).unwrap();
        assert_eq!(ws.frame_preds(0)[1].1, Some(Mark::Good));

        ws.set_tags_text(0, " blurry, ,occluded,blurry").unwrap();
        assert_eq!(ws.tags_of(0), vec!["blurry", "occluded"]);
        let before = ws.project.log.len();
        ws.set_tags_text(0, "blurry,occluded").unwrap();
        assert_eq!(ws.project.log.len(), before, "unchanged tags record nothing");

        let report = ws.save().unwrap().unwrap();
        assert_eq!(report.summary.added, 1);
        assert_eq!(report.summary.accepted, 1);
        assert_eq!(report.summary.deleted_annotations, 1);
        let ds = &ws.project.dataset;
        assert_eq!(ds.annotation(1).unwrap().bbox, BBox::new(0.0, 0.0, 15.0, 10.0));
        assert_eq!(ds.images()[0].tags, vec!["blurry", "occluded"]);
    }

    #[test]
    fn category_keys_and_frame_navigation() {
        let (_d, mut ws) = open();
        ws.frame.sel = Some(Sel::Gt(1));
        assert!(ws.set_category_index(1).is_err(), "only one category");
        ws.set_category_index(0).unwrap();
        assert_eq!(ws.draw_category(), Some(1));

        assert_eq!(ws.step_frame(1, false), Some(1));
        assert_eq!(ws.step_frame(-1, false), None);
        ws.go_to_frame(1);
        assert_eq!(ws.frame.sel, None);
        assert_eq!(ws.step_frame(1, true), None);
        assert_eq!(ws.step_frame(-1, true), Some(0));

        ws.toggle_frame_delete().unwrap();
        assert!(ws.is_frame_deleted(1));
        ws.select_crop_of_frame();
        assert_eq!(ws.selected_crop().unwrap().image_idx, 1);
    }
}
