//! GT ↔ prediction comparison.
//!
//! Matching is one-to-one, same category, greedy by highest IoU, accepting
//! pairs with IoU ≥ `match_iou` (TP). Independently, each box records its best
//! IoU against any box of the other kind, which drives the FP / FN filters.

use serde::{Deserialize, Serialize};

use crate::dataset::Dataset;
use crate::predictions::Predictions;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Thresholds {
    pub match_iou: f64,
    pub fp_iou: f64,
    pub fn_iou: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            match_iou: 0.5,
            fp_iou: 0.3,
            fn_iou: 0.3,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GtStatus {
    Tp,
    Fn,
    /// The image has no evaluated predictions.
    Unpredicted,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GtEval {
    pub status: GtStatus,
    /// Index of the matched prediction within its image.
    pub matched: Option<usize>,
    /// Matched IoU (TP), 0 (FN) or `None` (unpredicted).
    pub iou: Option<f64>,
    /// Best IoU against any same-category prediction, `None` when unpredicted.
    pub best_iou: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PredEval {
    /// Index of the matched annotation in `Dataset::annotations()`.
    pub matched: Option<usize>,
    /// Matched IoU, 0 when unmatched.
    pub iou: f64,
    pub best_iou: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub tp: usize,
    pub fp: usize,
    pub fn_: usize,
    pub unpredicted_images: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Evaluation {
    pub thresholds: Thresholds,
    /// Indexed like `Dataset::annotations()`.
    pub gt: Vec<GtEval>,
    /// Per image index, per prediction (empty for unpredicted images).
    pub preds: Vec<Vec<PredEval>>,
    pub counts: Counts,
}

impl Evaluation {
    pub fn is_fp_candidate(&self, image_idx: usize, pred_idx: usize) -> bool {
        self.preds[image_idx][pred_idx].best_iou < self.thresholds.fp_iou
    }

    pub fn is_fn_candidate(&self, ann_idx: usize) -> bool {
        self.gt[ann_idx]
            .best_iou
            .is_some_and(|b| b < self.thresholds.fn_iou)
    }
}

pub fn evaluate(ds: &Dataset, preds: &Predictions, thresholds: Thresholds) -> Evaluation {
    let unpredicted = GtEval {
        status: GtStatus::Unpredicted,
        matched: None,
        iou: None,
        best_iou: None,
    };
    let mut gt = vec![unpredicted; ds.annotations().len()];
    let mut per_image = Vec::with_capacity(ds.images().len());
    let mut counts = Counts::default();
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();

    for image_idx in 0..ds.images().len() {
        let Some(image_preds) = preds.of(image_idx) else {
            counts.unpredicted_images += 1;
            per_image.push(Vec::new());
            continue;
        };
        let anns: Vec<(usize, _)> = ds
            .annotations_of(image_idx)
            .map(|a| (ds.ann_idx(a.id).expect("indexed annotation"), a))
            .collect();
        let mut pevals = vec![
            PredEval {
                matched: None,
                iou: 0.0,
                best_iou: 0.0
            };
            image_preds.len()
        ];
        for &(ai, _) in &anns {
            gt[ai] = GtEval {
                status: GtStatus::Fn,
                matched: None,
                iou: Some(0.0),
                best_iou: Some(0.0),
            };
        }

        pairs.clear();
        for (ai, ann) in &anns {
            for (pi, pred) in image_preds.iter().enumerate() {
                if pred.category_id != ann.category_id {
                    continue;
                }
                let iou = ann.bbox.iou(&pred.bbox);
                let g = &mut gt[*ai];
                g.best_iou = Some(g.best_iou.unwrap_or(0.0).max(iou));
                pevals[pi].best_iou = pevals[pi].best_iou.max(iou);
                if iou >= thresholds.match_iou && iou > 0.0 {
                    pairs.push((iou, *ai, pi));
                }
            }
        }
        pairs.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        for &(iou, ai, pi) in &pairs {
            if gt[ai].matched.is_none() && pevals[pi].matched.is_none() {
                gt[ai].status = GtStatus::Tp;
                gt[ai].matched = Some(pi);
                gt[ai].iou = Some(iou);
                pevals[pi].matched = Some(ai);
                pevals[pi].iou = iou;
            }
        }
        for &(ai, _) in &anns {
            match gt[ai].status {
                GtStatus::Tp => counts.tp += 1,
                GtStatus::Fn => counts.fn_ += 1,
                GtStatus::Unpredicted => {}
            }
        }
        counts.fp += pevals.iter().filter(|p| p.matched.is_none()).count();
        per_image.push(pevals);
    }
    Evaluation {
        thresholds,
        gt,
        preds: per_image,
        counts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::test_util::dataset;
    use crate::geom::BBox;
    use crate::predictions::PredRef;

    fn preds_for(ds: &Dataset, boxes: &[(usize, BBox)], evaluated: &[usize]) -> Predictions {
        let mut p = Predictions::default();
        for &idx in evaluated {
            let image_id = ds.images()[idx].id;
            let list = boxes
                .iter()
                .filter(|(i, _)| *i == idx)
                .map(|&(_, bbox)| PredRef {
                    image_id,
                    category_id: 1,
                    bbox,
                    score: 0.9,
                })
                .collect();
            p.set_image(idx, list);
        }
        p
    }

    #[test]
    fn classifies_tp_fn_fp_and_unpredicted() {
        let a = BBox::new(0.0, 0.0, 10.0, 10.0);
        let far = BBox::new(50.0, 50.0, 10.0, 10.0);
        let ds = dataset(2, &[(1, a), (1, far), (2, a)]);
        let preds = preds_for(
            &ds,
            &[
                (0, BBox::new(1.0, 0.0, 10.0, 10.0)),
                (0, BBox::new(80.0, 80.0, 5.0, 5.0)),
            ],
            &[0],
        );
        let ev = evaluate(&ds, &preds, Thresholds::default());
        assert_eq!(ev.gt[0].status, GtStatus::Tp);
        assert!((ev.gt[0].iou.unwrap() - 90.0 / 110.0).abs() < 1e-9);
        assert_eq!(ev.gt[1].status, GtStatus::Fn);
        assert_eq!(ev.gt[2].status, GtStatus::Unpredicted);
        assert_eq!(
            ev.counts,
            Counts {
                tp: 1,
                fp: 1,
                fn_: 1,
                unpredicted_images: 1
            }
        );
        assert!(ev.is_fp_candidate(0, 1));
        assert!(!ev.is_fp_candidate(0, 0));
        assert!(ev.is_fn_candidate(1));
        assert!(!ev.is_fn_candidate(2), "unpredicted is never FN");
    }

    #[test]
    fn matching_is_one_to_one_highest_iou_first() {
        let gt = BBox::new(0.0, 0.0, 10.0, 10.0);
        let ds = dataset(1, &[(1, gt)]);
        let preds = preds_for(&ds, &[(0, BBox::new(2.0, 0.0, 10.0, 10.0)), (0, gt)], &[0]);
        let ev = evaluate(&ds, &preds, Thresholds::default());
        assert_eq!(ev.gt[0].matched, Some(1));
        assert_eq!(
            ev.preds[0][0].matched, None,
            "duplicate stays FP even though it overlaps"
        );
        assert!(
            !ev.is_fp_candidate(0, 0),
            "but it is not an FP-filter candidate (best IoU ≥ 0.3)"
        );
        assert_eq!(ev.counts.fp, 1);
    }

    #[test]
    fn loose_overlap_is_fn_but_not_filter_candidate() {
        let ds = dataset(1, &[(1, BBox::new(0.0, 0.0, 10.0, 10.0))]);
        // IoU = 60/140 ≈ 0.43: below match, above filter threshold.
        let preds = preds_for(&ds, &[(0, BBox::new(4.0, 0.0, 10.0, 10.0))], &[0]);
        let ev = evaluate(&ds, &preds, Thresholds::default());
        assert_eq!(ev.gt[0].status, GtStatus::Fn);
        assert!(!ev.is_fn_candidate(0));
        assert!(!ev.is_fp_candidate(0, 0));
    }
}
