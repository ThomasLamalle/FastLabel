//! Pending edits. Every user action is an [`Op`] appended to the [`EditLog`];
//! the dataset itself only changes when the log is applied on save.
//!
//! Marks on one target are mutually exclusive and toggle: applying the mark
//! the target already has clears it, applying another mark replaces it.

use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::coco::{AnnId, CategoryId, ImageId};
use crate::geom::BBox;
use crate::predictions::PredRef;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnnMark {
    Delete,
    Confirm,
    /// Replace the annotation geometry with this prediction's box.
    Replace(PredRef),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredMark {
    Accept,
    Reject,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Op {
    AnnMark {
        ann: AnnId,
        mark: AnnMark,
    },
    PredMark {
        pred: PredRef,
        mark: PredMark,
    },
    FrameDelete {
        image: ImageId,
    },
    AddBox {
        ann: AnnId,
        image: ImageId,
        category: CategoryId,
        bbox: BBox,
    },
    SetBox {
        ann: AnnId,
        bbox: BBox,
    },
    SetCategory {
        ann: AnnId,
        category: CategoryId,
    },
    SetTags {
        image: ImageId,
        tags: Vec<String>,
    },
    /// Several ops undone/redone as one step (row actions).
    Group {
        ops: Vec<Op>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NewBox {
    pub image: ImageId,
    pub category: CategoryId,
    pub bbox: BBox,
}

/// The effect of all ops up to the log cursor.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PendingState {
    pub ann_marks: HashMap<AnnId, AnnMark>,
    pub pred_marks: HashMap<PredRef, PredMark>,
    pub deleted_frames: HashSet<ImageId>,
    /// Boxes drawn in the editor, by provisional id.
    pub added: IndexMap<AnnId, NewBox>,
    /// Geometry edits of existing annotations.
    pub boxes: HashMap<AnnId, BBox>,
    /// Category edits of existing annotations.
    pub categories: HashMap<AnnId, CategoryId>,
    pub tags: HashMap<ImageId, Vec<String>>,
}

impl PendingState {
    fn apply(&mut self, op: &Op) {
        match op {
            Op::AnnMark { ann, mark } => toggle(&mut self.ann_marks, *ann, *mark),
            Op::PredMark { pred, mark } => toggle(&mut self.pred_marks, *pred, *mark),
            Op::FrameDelete { image } => {
                if !self.deleted_frames.remove(image) {
                    self.deleted_frames.insert(*image);
                }
            }
            Op::AddBox {
                ann,
                image,
                category,
                bbox,
            } => {
                self.added.insert(
                    *ann,
                    NewBox {
                        image: *image,
                        category: *category,
                        bbox: *bbox,
                    },
                );
            }
            Op::SetBox { ann, bbox } => match self.added.get_mut(ann) {
                Some(b) => b.bbox = *bbox,
                None => {
                    self.boxes.insert(*ann, *bbox);
                }
            },
            Op::SetCategory { ann, category } => match self.added.get_mut(ann) {
                Some(b) => b.category = *category,
                None => {
                    self.categories.insert(*ann, *category);
                }
            },
            Op::SetTags { image, tags } => {
                self.tags.insert(*image, normalize_tags(tags));
            }
            Op::Group { ops } => ops.iter().for_each(|op| self.apply(op)),
        }
    }

    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    pub fn ann_mark(&self, ann: AnnId) -> Option<AnnMark> {
        self.ann_marks.get(&ann).copied()
    }

    pub fn pred_mark(&self, pred: &PredRef) -> Option<PredMark> {
        self.pred_marks.get(pred).copied()
    }

    pub fn is_frame_deleted(&self, image: ImageId) -> bool {
        self.deleted_frames.contains(&image)
    }
}

fn toggle<K: Eq + std::hash::Hash, V: PartialEq>(map: &mut HashMap<K, V>, key: K, value: V) {
    if map.get(&key) == Some(&value) {
        map.remove(&key);
    } else {
        map.insert(key, value);
    }
}

/// Trims, drops empty entries and removes duplicates (first occurrence wins).
pub fn normalize_tags(tags: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    tags.iter()
        .map(|t| t.trim())
        .filter(|t| !t.is_empty() && seen.insert(t.to_string()))
        .map(str::to_string)
        .collect()
}

/// Linear undo/redo history of ops since the last save.
#[derive(Clone, Debug)]
pub struct EditLog {
    ops: Vec<Op>,
    cursor: usize,
    state: PendingState,
    next_ann_id: AnnId,
}

impl EditLog {
    /// `next_ann_id` must exceed every annotation id of the base dataset.
    pub fn new(next_ann_id: AnnId) -> Self {
        Self {
            ops: Vec::new(),
            cursor: 0,
            state: PendingState::default(),
            next_ann_id,
        }
    }

    pub fn state(&self) -> &PendingState {
        &self.state
    }

    /// True when there is something to save.
    pub fn is_dirty(&self) -> bool {
        !self.state.is_empty()
    }

    pub fn len(&self) -> usize {
        self.cursor
    }

    pub fn is_empty(&self) -> bool {
        self.cursor == 0
    }

    /// Provisional id for a box drawn in the editor.
    pub fn alloc_ann_id(&mut self) -> AnnId {
        let id = self.next_ann_id;
        self.next_ann_id += 1;
        id
    }

    pub fn push(&mut self, op: Op) {
        if let Op::AddBox { ann, .. } = &op {
            self.next_ann_id = self.next_ann_id.max(ann + 1);
        }
        self.ops.truncate(self.cursor);
        self.state.apply(&op);
        self.ops.push(op);
        self.cursor += 1;
    }

    pub fn can_undo(&self) -> bool {
        self.cursor > 0
    }

    pub fn can_redo(&self) -> bool {
        self.cursor < self.ops.len()
    }

    /// Undoes the last op and returns it (so the UI can reselect its target).
    pub fn undo(&mut self) -> Option<&Op> {
        if self.cursor == 0 {
            return None;
        }
        self.cursor -= 1;
        // Toggles depend on history, so rebuild from scratch: ops since the
        // last save are few and applying them is cheap.
        let mut state = PendingState::default();
        self.ops[..self.cursor]
            .iter()
            .for_each(|op| state.apply(op));
        self.state = state;
        self.ops.get(self.cursor)
    }

    pub fn redo(&mut self) -> Option<&Op> {
        let op = self.ops.get(self.cursor)?;
        self.state.apply(op);
        self.cursor += 1;
        Some(op)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn pred(x: f64) -> PredRef {
        PredRef {
            image_id: 1,
            category_id: 1,
            bbox: BBox::new(x, 0.0, 10.0, 10.0),
            score: 0.5,
        }
    }

    #[test]
    fn marks_toggle_and_replace_each_other() {
        let mut log = EditLog::new(100);
        log.push(Op::AnnMark {
            ann: 1,
            mark: AnnMark::Delete,
        });
        assert_eq!(log.state().ann_mark(1), Some(AnnMark::Delete));
        log.push(Op::AnnMark {
            ann: 1,
            mark: AnnMark::Confirm,
        });
        assert_eq!(log.state().ann_mark(1), Some(AnnMark::Confirm));
        log.push(Op::AnnMark {
            ann: 1,
            mark: AnnMark::Confirm,
        });
        assert_eq!(log.state().ann_mark(1), None);
        assert!(!log.is_dirty());

        log.push(Op::AnnMark {
            ann: 2,
            mark: AnnMark::Replace(pred(0.0)),
        });
        log.push(Op::AnnMark {
            ann: 2,
            mark: AnnMark::Replace(pred(1.0)),
        });
        assert_eq!(log.state().ann_mark(2), Some(AnnMark::Replace(pred(1.0))));
    }

    #[test]
    fn undo_redo_and_new_op_clears_redo() {
        let mut log = EditLog::new(100);
        log.push(Op::FrameDelete { image: 3 });
        log.push(Op::PredMark {
            pred: pred(0.0),
            mark: PredMark::Accept,
        });
        assert_eq!(
            log.undo(),
            Some(&Op::PredMark {
                pred: pred(0.0),
                mark: PredMark::Accept
            })
        );
        assert_eq!(log.state().pred_mark(&pred(0.0)), None);
        assert!(log.state().is_frame_deleted(3));
        log.redo();
        assert_eq!(log.state().pred_mark(&pred(0.0)), Some(PredMark::Accept));
        log.undo();
        log.push(Op::SetTags {
            image: 3,
            tags: vec![" a ".into(), "".into(), "a".into(), "b".into()],
        });
        assert!(!log.can_redo());
        assert_eq!(log.state().tags[&3], vec!["a", "b"]);
    }

    #[test]
    fn group_undoes_as_one_step() {
        let mut log = EditLog::new(100);
        log.push(Op::Group {
            ops: vec![
                Op::AnnMark {
                    ann: 1,
                    mark: AnnMark::Delete,
                },
                Op::AnnMark {
                    ann: 2,
                    mark: AnnMark::Delete,
                },
            ],
        });
        assert_eq!(log.state().ann_marks.len(), 2);
        log.undo();
        assert!(log.state().is_empty());
    }

    #[test]
    fn edits_on_added_boxes_update_the_new_box() {
        let mut log = EditLog::new(100);
        let id = log.alloc_ann_id();
        log.push(Op::AddBox {
            ann: id,
            image: 1,
            category: 1,
            bbox: BBox::new(0.0, 0.0, 5.0, 5.0),
        });
        log.push(Op::SetBox {
            ann: id,
            bbox: BBox::new(1.0, 1.0, 5.0, 5.0),
        });
        log.push(Op::SetCategory {
            ann: id,
            category: 2,
        });
        log.push(Op::SetBox {
            ann: 7,
            bbox: BBox::new(2.0, 2.0, 5.0, 5.0),
        });
        let st = log.state();
        assert_eq!(
            st.added[&id],
            NewBox {
                image: 1,
                category: 2,
                bbox: BBox::new(1.0, 1.0, 5.0, 5.0)
            }
        );
        assert_eq!(st.boxes[&7], BBox::new(2.0, 2.0, 5.0, 5.0));
        assert!(!st.boxes.contains_key(&id));
        assert_eq!(log.alloc_ann_id(), 101);
    }

    #[test]
    fn ops_serialize_as_tagged_json() {
        let op = Op::AnnMark {
            ann: 1,
            mark: AnnMark::Replace(pred(2.0)),
        };
        let json = serde_json::to_string(&op).unwrap();
        assert!(json.starts_with(r#"{"op":"ann_mark""#), "{json}");
        assert_eq!(serde_json::from_str::<Op>(&json).unwrap(), op);
    }

    fn arb_op() -> impl Strategy<Value = Op> {
        prop_oneof![
            (0..5i64, 0..3u8).prop_map(|(ann, m)| Op::AnnMark {
                ann,
                mark: match m {
                    0 => AnnMark::Delete,
                    1 => AnnMark::Confirm,
                    _ => AnnMark::Replace(pred(ann as f64)),
                },
            }),
            (0..5u8, any::<bool>()).prop_map(|(x, a)| Op::PredMark {
                pred: pred(f64::from(x)),
                mark: if a {
                    PredMark::Accept
                } else {
                    PredMark::Reject
                },
            }),
            (0..5i64).prop_map(|image| Op::FrameDelete { image }),
        ]
    }

    proptest! {
        #[test]
        fn undo_all_returns_to_clean_and_redo_all_restores(ops in prop::collection::vec(arb_op(), 0..40)) {
            let mut log = EditLog::new(100);
            ops.iter().cloned().for_each(|op| log.push(op));
            let full = log.state().clone();
            while log.undo().is_some() {}
            prop_assert!(log.state().is_empty());
            while log.redo().is_some() {}
            prop_assert_eq!(log.state(), &full);
        }

        #[test]
        fn same_mark_twice_restores_or_clears(ops in prop::collection::vec(arb_op(), 0..20), extra in arb_op()) {
            let mut log = EditLog::new(100);
            ops.iter().cloned().for_each(|op| log.push(op));
            let before = log.state().clone();
            log.push(extra.clone());
            log.push(extra.clone());
            let st = log.state();
            // X, X restores the previous mark if it was none or X; a different
            // previous mark Y is cleared (X replaced it, then X toggled off).
            match extra {
                Op::AnnMark { ann, mark } => {
                    let prev = before.ann_mark(ann);
                    let expected = if prev.is_none() || prev == Some(mark) { prev } else { None };
                    prop_assert_eq!(st.ann_mark(ann), expected);
                    prop_assert_eq!(&st.pred_marks, &before.pred_marks);
                }
                Op::PredMark { pred, mark } => {
                    let prev = before.pred_mark(&pred);
                    let expected = if prev.is_none() || prev == Some(mark) { prev } else { None };
                    prop_assert_eq!(st.pred_mark(&pred), expected);
                    prop_assert_eq!(&st.ann_marks, &before.ann_marks);
                }
                // Frame deletion is a plain toggle: twice is identity.
                _ => prop_assert_eq!(st, &before),
            }
        }
    }
}
