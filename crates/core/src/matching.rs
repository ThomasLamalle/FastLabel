//! GT ↔ prediction comparison, without thresholds.
//!
//! Each GT box is paired one-to-one with a same-category prediction, highest
//! IoU first, as long as they overlap at all (IoU > 0). Then:
//! - FP: a prediction that overlaps no GT box (IoU 0 with all of them);
//! - FN: a GT box that overlaps no prediction;
//! - extra: an unpaired prediction overlapping a GT already paired with a
//!   better prediction (a duplicate), and symmetrically an unpaired GT.
//!
//! How well a pair agrees is its IoU, used for sorting.

use crate::dataset::Dataset;
use crate::predictions::Predictions;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GtStatus {
    /// Paired with a prediction.
    Matched,
    /// Overlaps no prediction.
    Fn,
    /// Overlaps only predictions paired with other GT boxes.
    Unmatched,
    /// The image has no evaluated predictions.
    Unpredicted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredStatus {
    Matched,
    /// Overlaps no GT box.
    Fp,
    /// Overlaps only GT boxes paired with other predictions (duplicate).
    Extra,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GtEval {
    pub status: GtStatus,
    /// Index of the paired prediction within its image.
    pub matched: Option<usize>,
    /// Paired IoU, 0 when unpaired, `None` when unpredicted.
    pub iou: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PredEval {
    pub status: PredStatus,
    /// Index of the paired annotation in `Dataset::annotations()`.
    pub matched: Option<usize>,
    /// Paired IoU, 0 when unpaired.
    pub iou: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub matched: usize,
    pub fp: usize,
    pub fn_: usize,
    /// Unpaired predictions and GT boxes that still overlap something.
    pub extra: usize,
    pub unpredicted_images: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Evaluation {
    /// Indexed like `Dataset::annotations()`.
    pub gt: Vec<GtEval>,
    /// Per image index, per prediction (empty for unpredicted images).
    pub preds: Vec<Vec<PredEval>>,
    pub counts: Counts,
}

impl Evaluation {
    pub fn is_fp(&self, image_idx: usize, pred_idx: usize) -> bool {
        self.preds[image_idx][pred_idx].status == PredStatus::Fp
    }

    pub fn is_fn(&self, ann_idx: usize) -> bool {
        self.gt[ann_idx].status == GtStatus::Fn
    }
}

pub fn evaluate(ds: &Dataset, preds: &Predictions) -> Evaluation {
    let unpredicted = GtEval { status: GtStatus::Unpredicted, matched: None, iou: None };
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
        let anns: Vec<(usize, _)> =
            ds.annotations_of(image_idx).map(|a| (ds.ann_idx(a.id).expect("indexed"), a)).collect();
        let mut pevals =
            vec![PredEval { status: PredStatus::Fp, matched: None, iou: 0.0 }; image_preds.len()];
        let mut gt_overlaps = vec![false; anns.len()];
        pairs.clear();
        for (k, (ai, ann)) in anns.iter().enumerate() {
            for (pi, pred) in image_preds.iter().enumerate() {
                if pred.category_id != ann.category_id {
                    continue;
                }
                let iou = ann.bbox.iou(&pred.bbox);
                if iou > 0.0 {
                    gt_overlaps[k] = true;
                    pevals[pi].status = PredStatus::Extra;
                    pairs.push((iou, *ai, pi));
                }
            }
        }
        pairs.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        for (k, (ai, _)) in anns.iter().enumerate() {
            let status = if gt_overlaps[k] { GtStatus::Unmatched } else { GtStatus::Fn };
            gt[*ai] = GtEval { status, matched: None, iou: Some(0.0) };
        }
        for &(iou, ai, pi) in &pairs {
            if gt[ai].matched.is_none() && pevals[pi].matched.is_none() {
                gt[ai] = GtEval { status: GtStatus::Matched, matched: Some(pi), iou: Some(iou) };
                pevals[pi] = PredEval { status: PredStatus::Matched, matched: Some(ai), iou };
            }
        }
        for (ai, _) in &anns {
            match gt[*ai].status {
                GtStatus::Matched => counts.matched += 1,
                GtStatus::Fn => counts.fn_ += 1,
                GtStatus::Unmatched => counts.extra += 1,
                GtStatus::Unpredicted => {}
            }
        }
        for p in &pevals {
            match p.status {
                PredStatus::Fp => counts.fp += 1,
                PredStatus::Extra => counts.extra += 1,
                PredStatus::Matched => {}
            }
        }
        per_image.push(pevals);
    }
    Evaluation { gt, preds: per_image, counts }
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
                .map(|&(_, bbox)| PredRef { image_id, category_id: 1, bbox, score: 0.9 })
                .collect();
            p.set_image(idx, list);
        }
        p
    }

    #[test]
    fn classifies_matched_fp_fn_and_unpredicted() {
        let a = BBox::new(0.0, 0.0, 10.0, 10.0);
        let far = BBox::new(50.0, 50.0, 10.0, 10.0);
        let ds = dataset(2, &[(1, a), (1, far), (2, a)]);
        let preds = preds_for(
            &ds,
            &[(0, BBox::new(1.0, 0.0, 10.0, 10.0)), (0, BBox::new(80.0, 80.0, 5.0, 5.0))],
            &[0],
        );
        let ev = evaluate(&ds, &preds);
        assert_eq!(ev.gt[0].status, GtStatus::Matched);
        assert!((ev.gt[0].iou.unwrap() - 90.0 / 110.0).abs() < 1e-9);
        assert!(ev.is_fn(1));
        assert_eq!(ev.gt[2].status, GtStatus::Unpredicted);
        assert!(!ev.is_fn(2), "unpredicted is never FN");
        assert!(ev.is_fp(0, 1));
        assert!(!ev.is_fp(0, 0));
        assert_eq!(
            ev.counts,
            Counts { matched: 1, fp: 1, fn_: 1, extra: 0, unpredicted_images: 1 }
        );
    }

    #[test]
    fn any_overlap_pairs_and_duplicates_are_extra() {
        let gt = BBox::new(0.0, 0.0, 10.0, 10.0);
        let ds = dataset(1, &[(1, gt)]);
        // A loose box (IoU 0.05) still pairs: it is reviewed through the IoU sort.
        let loose = preds_for(&ds, &[(0, BBox::new(9.0, 0.0, 10.0, 10.0))], &[0]);
        let ev = evaluate(&ds, &loose);
        assert_eq!(ev.gt[0].status, GtStatus::Matched);
        assert!(ev.gt[0].iou.unwrap() < 0.1);

        let dup = preds_for(&ds, &[(0, BBox::new(2.0, 0.0, 10.0, 10.0)), (0, gt)], &[0]);
        let ev = evaluate(&ds, &dup);
        assert_eq!(ev.gt[0].matched, Some(1), "best overlap wins");
        assert_eq!(ev.preds[0][0].status, PredStatus::Extra);
        assert!(!ev.is_fp(0, 0));
        assert_eq!(ev.counts.extra, 1);
    }
}
