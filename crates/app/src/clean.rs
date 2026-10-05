//! Cleaning workflows. First one: overlapping GT boxes (double
//! annotations). Boxes of the same category that overlap at all are grouped
//! per frame; the reviewer keeps one box (or all of them, after editing).

use fastlabel_core::{AnnId, AnnMark, BBox, Op};

use crate::frame::EditBox;
use crate::workspace::{ActionResult, Mark, Workspace};

#[derive(Clone, Debug, PartialEq)]
pub struct OverlapGroup {
    pub image_idx: usize,
    /// Left to right; box `n` is selected with key `n + 1`.
    pub boxes: Vec<AnnId>,
}

/// Connected components of `boxes` under "overlaps (IoU > 0), same category".
fn components(boxes: &[EditBox]) -> Vec<Vec<usize>> {
    let n = boxes.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], i: usize) -> usize {
        let mut r = i;
        while p[r] != r {
            r = p[r];
        }
        p[i] = r;
        r
    }
    for i in 0..n {
        for j in i + 1..n {
            if boxes[i].category == boxes[j].category && boxes[i].bbox.iou(&boxes[j].bbox) > 0.0 {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                parent[a] = b;
            }
        }
    }
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut root_of: Vec<Option<usize>> = vec![None; n];
    for i in 0..n {
        let r = find(&mut parent, i);
        match root_of[r] {
            Some(g) => groups[g].push(i),
            None => {
                root_of[r] = Some(groups.len());
                groups.push(vec![i]);
            }
        }
    }
    groups.retain(|g| g.len() > 1);
    groups
}

impl Workspace {
    /// Groups of overlapping GT boxes, as currently edited. Boxes already
    /// marked for deletion, frames marked F and groups whose boxes were all
    /// kept (confirmed now or reviewed in an earlier session) are left out.
    pub fn overlap_groups(&self) -> Vec<OverlapGroup> {
        let mut out = Vec::new();
        for image_idx in 0..self.project.dataset.images().len() {
            if self.is_frame_deleted(image_idx) {
                continue;
            }
            let boxes: Vec<EditBox> = self
                .frame_boxes(image_idx)
                .into_iter()
                .filter(|b| b.mark != Some(Mark::Delete))
                .collect();
            if boxes.len() < 2 {
                continue;
            }
            for comp in components(&boxes) {
                let mut ids: Vec<(f64, AnnId)> =
                    comp.iter().map(|&i| (boxes[i].bbox.x, boxes[i].id)).collect();
                ids.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                let kept = |&(_, id): &(f64, AnnId)| {
                    let b = boxes.iter().find(|b| b.id == id).expect("in frame");
                    b.mark == Some(Mark::Good) || self.project.review.is_reviewed(id)
                };
                if ids.iter().all(kept) {
                    continue;
                }
                out.push(OverlapGroup {
                    image_idx,
                    boxes: ids.into_iter().map(|(_, id)| id).collect(),
                });
            }
        }
        out
    }

    /// Region to show for a group: all its boxes.
    pub fn group_extent(&self, group: &OverlapGroup) -> Option<BBox> {
        let boxes = self.frame_boxes(group.image_idx);
        group
            .boxes
            .iter()
            .filter_map(|id| boxes.iter().find(|b| b.id == *id).map(|b| b.bbox))
            .reduce(|a, b| {
                BBox::from_corners(
                    a.x.min(b.x),
                    a.y.min(b.y),
                    a.x2().max(b.x2()),
                    a.y2().max(b.y2()),
                )
            })
    }

    /// Sets each box of the group to `mark_of(id)` (no-op where already
    /// set), keeping the frame, as one undo step.
    fn resolve_group(
        &mut self,
        group: &OverlapGroup,
        mark_of: impl Fn(AnnId) -> AnnMark,
    ) -> ActionResult {
        let st = self.project.log.state();
        let image = self.project.dataset.images()[group.image_idx].id;
        let mut ops = Vec::new();
        if st.is_frame_deleted(image) {
            ops.push(Op::FrameDelete { image });
        }
        for &ann in &group.boxes {
            let want = mark_of(ann);
            if st.ann_mark(ann) != Some(want) {
                ops.push(Op::AnnMark { ann, mark: want });
            }
        }
        if ops.is_empty() { Ok(()) } else { self.record(Op::Group { ops }) }
    }

    /// Keeps box `keep` (confirmed) and marks the other boxes of the group
    /// for deletion.
    pub fn keep_only(&mut self, group: &OverlapGroup, keep: AnnId) -> ActionResult {
        self.resolve_group(group, |id| if id == keep { AnnMark::Confirm } else { AnnMark::Delete })
    }

    /// Keeps every box of the group (e.g. after fixing them by hand).
    pub fn keep_all(&mut self, group: &OverlapGroup) -> ActionResult {
        self.resolve_group(group, |_| AnnMark::Confirm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crops::ViewOptions;
    use crate::frame::Sel;
    use fastlabel_core::coco::{CocoAnnotation, CocoFile};
    use std::fs;

    /// Image 1: boxes 1-2-3 chained by overlaps + 4 alone; box 5 overlaps 1
    /// but is another category. Image 2: 6 and 7 overlap.
    fn open() -> (tempfile::TempDir, Workspace) {
        let (ds, ..) = crate::crops::tests::fixture();
        let mut coco: CocoFile = ds.coco.clone();
        coco.categories.push(fastlabel_core::coco::CocoCategory {
            id: 2,
            name: "face".into(),
            supercategory: String::new(),
            extra: Default::default(),
        });
        let b = |x, y, w, h| BBox::new(x, y, w, h);
        coco.annotations = vec![
            CocoAnnotation::new(1, 1, 1, b(0.0, 0.0, 20.0, 20.0)),
            CocoAnnotation::new(2, 1, 1, b(15.0, 0.0, 20.0, 20.0)),
            CocoAnnotation::new(3, 1, 1, b(30.0, 0.0, 20.0, 20.0)),
            CocoAnnotation::new(4, 1, 1, b(100.0, 50.0, 10.0, 10.0)),
            CocoAnnotation::new(5, 1, 2, b(2.0, 2.0, 5.0, 5.0)),
            CocoAnnotation::new(6, 2, 1, b(10.0, 10.0, 30.0, 20.0)),
            CocoAnnotation::new(7, 2, 1, b(12.0, 11.0, 30.0, 20.0)),
        ];
        let dir = tempfile::tempdir().unwrap();
        coco.save(&dir.path().join("annotations.v001.json")).unwrap();
        for f in ["images/train/a.jpg", "images/val/b.jpg"] {
            let p = dir.path().join(f);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(p, b"x").unwrap();
        }
        let ws = Workspace::open(dir.path(), ViewOptions::default(), 4).unwrap();
        (dir, ws)
    }

    #[test]
    fn groups_overlapping_boxes_of_the_same_category() {
        let (_d, ws) = open();
        let groups = ws.overlap_groups();
        assert_eq!(
            groups,
            vec![
                OverlapGroup { image_idx: 0, boxes: vec![1, 2, 3] },
                OverlapGroup { image_idx: 1, boxes: vec![6, 7] },
            ]
        );
        assert_eq!(ws.group_extent(&groups[0]), Some(BBox::new(0.0, 0.0, 50.0, 20.0)));
    }

    #[test]
    fn keep_one_deletes_the_others_in_one_step() {
        let (_d, mut ws) = open();
        let groups = ws.overlap_groups();
        ws.keep_only(&groups[0], 2).unwrap();
        let marks: Vec<_> = ws.frame_boxes(0).iter().map(|b| (b.id, b.mark)).collect();
        assert_eq!(
            &marks[..4],
            &[(1, Some(Mark::Delete)), (2, Some(Mark::Good)), (3, Some(Mark::Delete)), (4, None)]
        );
        // Changing one's mind: keep 1 instead.
        ws.keep_only(&groups[0], 1).unwrap();
        assert_eq!(ws.frame_boxes(0)[0].mark, Some(Mark::Good));
        assert_eq!(ws.frame_boxes(0)[1].mark, Some(Mark::Delete));
        ws.undo().unwrap();
        assert_eq!(ws.frame_boxes(0)[1].mark, Some(Mark::Good), "one undo step per decision");
        // Resolved groups disappear from a fresh scan (deleted boxes left out).
        assert_eq!(ws.overlap_groups().len(), 1);
    }

    #[test]
    fn keep_all_after_editing_and_frame_deletion_is_undone() {
        let (_d, mut ws) = open();
        let groups = ws.overlap_groups();
        // The reviewer fixes box 7 by hand, then keeps both.
        ws.frame.image_idx = 1;
        ws.frame.sel = Some(Sel::Gt(7));
        ws.set_box(7, BBox::new(60.0, 10.0, 30.0, 20.0)).unwrap();
        ws.toggle_frame_delete().unwrap();
        ws.keep_all(&groups[1]).unwrap();
        assert!(!ws.is_frame_deleted(1), "deciding keeps the frame");
        assert!(ws.frame_boxes(1).iter().all(|b| b.mark == Some(Mark::Good)));
        assert_eq!(ws.overlap_groups().len(), 1, "boxes no longer overlap");
        // Kept groups do not come back.
        ws.keep_all(&groups[0]).unwrap();
        assert!(ws.overlap_groups().is_empty());
    }
}
