//! Persistent review state (`.fastlabel/review.json`): which annotations were
//! already reviewed and which predictions were rejected.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::coco::{AnnId, CategoryId, ImageId};
use crate::error::Result;
use crate::fsutil;
use crate::geom::BBox;
use crate::predictions::PredRef;

pub const REVIEW_FILE: &str = "review.json";

/// A rejected prediction still "covers" a new one at this IoU or above.
pub const REJECT_MATCH_IOU: f64 = 0.9;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RejectedPred {
    pub image_id: ImageId,
    pub category_id: CategoryId,
    pub bbox: BBox,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ReviewState {
    pub reviewed: BTreeSet<AnnId>,
    pub rejected: Vec<RejectedPred>,
    #[serde(skip)]
    rejected_by_image: HashMap<ImageId, Vec<usize>>,
}

impl ReviewState {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let mut state: Self = fsutil::read_json(path)?;
        state.reindex();
        Ok(state)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        fsutil::write_json_atomic(path, self)
    }

    fn reindex(&mut self) {
        self.rejected_by_image.clear();
        for (i, r) in self.rejected.iter().enumerate() {
            self.rejected_by_image
                .entry(r.image_id)
                .or_default()
                .push(i);
        }
    }

    pub fn is_reviewed(&self, ann: AnnId) -> bool {
        self.reviewed.contains(&ann)
    }

    /// True when a previously rejected prediction overlaps this one enough.
    pub fn is_rejected(&self, pred: &PredRef) -> bool {
        self.rejected_by_image
            .get(&pred.image_id)
            .is_some_and(|idx| {
                idx.iter().map(|&i| &self.rejected[i]).any(|r| {
                    r.category_id == pred.category_id && r.bbox.iou(&pred.bbox) >= REJECT_MATCH_IOU
                })
            })
    }

    pub fn reject(&mut self, pred: &PredRef) {
        if !self.is_rejected(pred) {
            self.rejected.push(RejectedPred {
                image_id: pred.image_id,
                category_id: pred.category_id,
                bbox: pred.bbox,
            });
            self.rejected_by_image
                .entry(pred.image_id)
                .or_default()
                .push(self.rejected.len() - 1);
        }
    }

    /// Drops everything about removed annotations and images.
    pub fn retain(
        &mut self,
        keep_ann: impl Fn(AnnId) -> bool,
        keep_image: impl Fn(ImageId) -> bool,
    ) {
        self.reviewed.retain(|&a| keep_ann(a));
        self.rejected.retain(|r| keep_image(r.image_id));
        self.reindex();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pred(x: f64) -> PredRef {
        PredRef {
            image_id: 1,
            category_id: 1,
            bbox: BBox::new(x, 0.0, 100.0, 100.0),
            score: 0.3,
        }
    }

    #[test]
    fn rejection_survives_small_geometry_changes_and_reload() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(REVIEW_FILE);
        let mut st = ReviewState::default();
        st.reject(&pred(0.0));
        st.reject(&pred(1.0)); // already covered
        st.reviewed.insert(5);
        st.save(&path).unwrap();

        let st = ReviewState::load(&path).unwrap();
        assert_eq!(st.rejected.len(), 1);
        assert!(st.is_rejected(&pred(2.0)));
        assert!(!st.is_rejected(&pred(30.0)));
        assert!(!st.is_rejected(&PredRef {
            category_id: 2,
            ..pred(0.0)
        }));
        assert!(st.is_reviewed(5));
    }

    #[test]
    fn retain_prunes_removed_items() {
        let mut st = ReviewState::default();
        st.reject(&pred(0.0));
        st.reviewed.extend([1, 2]);
        st.retain(|a| a != 1, |i| i != 1);
        assert!(st.rejected.is_empty());
        assert!(!st.is_rejected(&pred(0.0)));
        assert_eq!(st.reviewed.iter().copied().collect::<Vec<_>>(), vec![2]);
    }
}
