//! The mosaic's crop list: which crops exist, filtering and sorting.

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::sync::Arc;

use fastlabel_core::matching::{GtStatus, PredStatus};
use fastlabel_core::review::ReviewState;
use fastlabel_core::{
    AnnId, BBox, CategoryId, Dataset, Evaluation, ImageId, PredRef, Predictions, Split,
};
use serde::{Deserialize, Serialize};

use crate::metrics::{self, Uniformity};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SortKey {
    #[default]
    Id,
    Width,
    Height,
    AspectRatio,
    Area,
    Iou,
    Confidence,
    Uniformity,
    ImageDate,
}

impl SortKey {
    pub const ALL: [SortKey; 9] = [
        SortKey::Id,
        SortKey::Width,
        SortKey::Height,
        SortKey::AspectRatio,
        SortKey::Area,
        SortKey::Iou,
        SortKey::Confidence,
        SortKey::Uniformity,
        SortKey::ImageDate,
    ];

    pub fn label(self) -> &'static str {
        match self {
            SortKey::Id => "Id",
            SortKey::Width => "Width",
            SortKey::Height => "Height",
            SortKey::AspectRatio => "Aspect ratio (w / h)",
            SortKey::Area => "Area",
            SortKey::Iou => "IoU with prediction",
            SortKey::Confidence => "Prediction confidence",
            SortKey::Uniformity => "Color uniformity",
            SortKey::ImageDate => "Image date (file)",
        }
    }

    pub fn needs_predictions(self) -> bool {
        matches!(self, SortKey::Iou | SortKey::Confidence)
    }
}

/// Split filter: all frames, one split, or frames without a split.
/// Stored like the former `Option<Split>` (`null`, `"Train"`…) plus `"None"`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "Option<String>", into = "Option<String>")]
pub enum SplitFilter {
    #[default]
    All,
    Only(Split),
    NoSplit,
}

impl SplitFilter {
    pub fn matches(self, split: Option<Split>) -> bool {
        match self {
            SplitFilter::All => true,
            SplitFilter::Only(s) => split == Some(s),
            SplitFilter::NoSplit => split.is_none(),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            SplitFilter::All => "All splits",
            SplitFilter::Only(s) => s.name(),
            SplitFilter::NoSplit => "no split",
        }
    }

    pub fn choices() -> [SplitFilter; 5] {
        [
            SplitFilter::All,
            SplitFilter::Only(Split::Train),
            SplitFilter::Only(Split::Val),
            SplitFilter::Only(Split::Test),
            SplitFilter::NoSplit,
        ]
    }
}

impl From<Option<String>> for SplitFilter {
    fn from(v: Option<String>) -> Self {
        match v.as_deref() {
            Some("None") => SplitFilter::NoSplit,
            Some(s) => Split::parse(s).map_or(SplitFilter::All, SplitFilter::Only),
            None => SplitFilter::All,
        }
    }
}

impl From<SplitFilter> for Option<String> {
    fn from(f: SplitFilter) -> Self {
        match f {
            SplitFilter::All => None,
            SplitFilter::Only(s) => Some(format!("{s:?}")),
            SplitFilter::NoSplit => Some("None".into()),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ViewOptions {
    pub sort: SortKey,
    pub descending: bool,
    pub fp_only: bool,
    pub fn_only: bool,
    pub hide_reviewed: bool,
    pub split: SplitFilter,
    /// Only crops of these frames (a cluster selection); not persisted.
    #[serde(skip)]
    pub frames: Option<Arc<BTreeSet<ImageId>>>,
}

/// Per-frame values kept outside the dataset, indexed like its images.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameInfo<'a> {
    /// File dates, when sorting by date.
    pub dates: Option<&'a [Option<f64>]>,
    /// Splits including pending moves; the dataset's when `None`.
    pub splits: Option<&'a [Option<Split>]>,
}

/// Stable identity of a crop across rebuilds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CropId {
    Gt(AnnId),
    Pred(PredRef),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CropStatus {
    /// GT paired with a prediction (both shown).
    Matched,
    /// GT overlapping no prediction.
    Fn,
    /// Prediction overlapping no GT.
    Fp,
    /// Unpaired box overlapping a box already paired with another one
    /// (typically a duplicate prediction).
    Extra,
    /// GT on an image the model was not run on.
    Unpredicted,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Crop {
    pub id: CropId,
    pub image_idx: usize,
    pub image_id: ImageId,
    pub category_id: CategoryId,
    /// The GT box (GT crops only).
    pub gt: Option<BBox>,
    /// The matched prediction (TP crops) or the prediction itself (FP crops).
    pub pred: Option<PredRef>,
    pub status: CropStatus,
    /// IoU with the paired box, 0 when unpaired, `None` when unpredicted.
    pub iou: Option<f64>,
    /// FP (prediction) or FN (GT): shown by the FP-only / FN-only filters.
    pub filter_candidate: bool,
    /// Grayscale standard deviation inside the box, once computed.
    pub uniformity: Option<f32>,
    /// Image file date (creation, else modification), seconds since epoch.
    pub date: Option<f64>,
}

impl Crop {
    pub fn ann_id(&self) -> Option<AnnId> {
        match self.id {
            CropId::Gt(a) => Some(a),
            CropId::Pred(_) => None,
        }
    }

    pub fn is_pred(&self) -> bool {
        matches!(self.id, CropId::Pred(_))
    }

    /// The box the crop is about: GT for GT crops, the prediction otherwise.
    pub fn primary(&self) -> BBox {
        self.gt.or(self.pred.map(|p| p.bbox)).expect("a crop has a box")
    }

    /// Smallest box containing every drawn box, used to frame the crop.
    pub fn frame_box(&self) -> BBox {
        let a = self.primary();
        match (self.gt, self.pred) {
            (Some(g), Some(p)) => BBox::from_corners(
                g.x.min(p.bbox.x),
                g.y.min(p.bbox.y),
                g.x2().max(p.bbox.x2()),
                g.y2().max(p.bbox.y2()),
            ),
            _ => a,
        }
    }

    fn sort_value(&self, key: SortKey) -> Option<f64> {
        match key {
            SortKey::Id => self.ann_id().map(|a| a as f64),
            SortKey::Width => Some(self.primary().w),
            SortKey::Height => Some(self.primary().h),
            SortKey::AspectRatio => {
                let b = self.primary();
                (b.h > 0.0).then(|| b.w / b.h)
            }
            SortKey::Area => Some(self.primary().area()),
            SortKey::Iou => self.iou,
            SortKey::Confidence => self.pred.map(|p| p.score),
            SortKey::Uniformity => self.uniformity.map(f64::from),
            SortKey::ImageDate => self.date,
        }
    }
}

/// Builds the crop list: one crop per GT annotation (with its matched
/// prediction), plus one per unmatched prediction, filtered and sorted.
pub fn build(
    ds: &Dataset,
    preds: Option<&Predictions>,
    eval: Option<&Evaluation>,
    review: &ReviewState,
    uniformity: &Uniformity,
    frames: FrameInfo,
    opts: &ViewOptions,
) -> Vec<Crop> {
    let mut crops = Vec::with_capacity(ds.annotations().len());
    let dates = frames.dates;
    let split_of = |idx: usize| frames.splits.map_or_else(|| ds.split_of(idx), |s| s[idx]);
    for (image_idx, img) in ds.images().iter().enumerate() {
        if !opts.split.matches(split_of(image_idx)) {
            continue;
        }
        if opts.frames.as_ref().is_some_and(|f| !f.contains(&img.id)) {
            continue;
        }
        let image_preds = preds.and_then(|p| p.of(image_idx));
        for ann in ds.annotations_of(image_idx) {
            let ann_idx = ds.ann_idx(ann.id).expect("indexed");
            let gt_eval = eval.map(|e| e.gt[ann_idx]);
            let (status, iou, pred) = match gt_eval.map(|g| (g, g.status)) {
                Some((g, GtStatus::Matched)) => {
                    let pred = g.matched.and_then(|pi| image_preds.map(|p| p[pi]));
                    (CropStatus::Matched, g.iou, pred)
                }
                Some((g, GtStatus::Fn)) => (CropStatus::Fn, g.iou, None),
                Some((g, GtStatus::Unmatched)) => (CropStatus::Extra, g.iou, None),
                _ => (CropStatus::Unpredicted, None, None),
            };
            crops.push(Crop {
                id: CropId::Gt(ann.id),
                image_idx,
                image_id: img.id,
                category_id: ann.category_id,
                gt: Some(ann.bbox),
                pred,
                status,
                iou,
                filter_candidate: eval.is_some_and(|e| e.is_fn(ann_idx)),
                uniformity: uniformity.get(metrics::key(&img.file_name, &ann.bbox)),
                date: dates.and_then(|d| d[image_idx]),
            });
        }
        if let (Some(image_preds), Some(e)) = (image_preds, eval) {
            for (pi, p) in image_preds.iter().enumerate() {
                let status = match e.preds[image_idx][pi].status {
                    PredStatus::Matched => continue,
                    PredStatus::Fp => CropStatus::Fp,
                    PredStatus::Extra => CropStatus::Extra,
                };
                crops.push(Crop {
                    id: CropId::Pred(*p),
                    image_idx,
                    image_id: img.id,
                    category_id: p.category_id,
                    gt: None,
                    pred: Some(*p),
                    status,
                    iou: Some(0.0),
                    filter_candidate: status == CropStatus::Fp,
                    uniformity: uniformity.get(metrics::key(&img.file_name, &p.bbox)),
                    date: dates.and_then(|d| d[image_idx]),
                });
            }
        }
    }

    crops.retain(|c| {
        let reviewed = match c.id {
            CropId::Gt(a) => review.is_reviewed(a),
            CropId::Pred(p) => review.is_rejected(&p),
        };
        if opts.hide_reviewed && reviewed {
            return false;
        }
        if opts.fp_only || opts.fn_only {
            let wanted = if c.is_pred() { opts.fp_only } else { opts.fn_only };
            return wanted && c.filter_candidate;
        }
        true
    });

    // Missing values always go last; ties keep dataset order (GT before preds).
    crops.sort_by(|a, b| match (a.sort_value(opts.sort), b.sort_value(opts.sort)) {
        (Some(x), Some(y)) => {
            let o = x.total_cmp(&y);
            if opts.descending { o.reverse() } else { o }
        }
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    });
    crops
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use fastlabel_core::coco::{CocoAnnotation, CocoCategory, CocoFile, CocoImage};
    use fastlabel_core::matching::evaluate;
    use serde_json::Map;
    use std::path::{Path, PathBuf};

    pub fn dataset(images: &[&str], anns: &[(ImageId, BBox)]) -> Dataset {
        let coco = CocoFile {
            images: images
                .iter()
                .enumerate()
                .map(|(i, f)| CocoImage {
                    id: i as i64 + 1,
                    file_name: (*f).into(),
                    width: 200,
                    height: 100,
                    tags: vec![],
                    extra: Map::new(),
                })
                .collect(),
            annotations: anns
                .iter()
                .enumerate()
                .map(|(i, &(img, b))| CocoAnnotation::new(i as i64 + 1, img, 1, b))
                .collect(),
            categories: vec![CocoCategory {
                id: 1,
                name: "bib".into(),
                supercategory: String::new(),
                extra: Map::new(),
            }],
            extra: Map::new(),
        };
        Dataset::new(PathBuf::from("/d"), coco, Path::new("t")).unwrap().0
    }

    pub fn pred(image_id: ImageId, bbox: BBox, score: f64) -> PredRef {
        PredRef { image_id, category_id: 1, bbox, score }
    }

    /// 2 images: image 1 has a TP (iou .82), an FN and an FP; image 2 (val)
    /// has a GT and no predictions run.
    pub fn fixture() -> (Dataset, Predictions, Evaluation) {
        let ds = dataset(
            &["images/train/a.jpg", "images/val/b.jpg"],
            &[
                (1, BBox::new(0.0, 0.0, 10.0, 10.0)),
                (1, BBox::new(50.0, 50.0, 20.0, 30.0)),
                (2, BBox::new(5.0, 5.0, 40.0, 5.0)),
            ],
        );
        let mut preds = Predictions::empty("m", 0.1);
        preds.set_image(
            0,
            vec![
                pred(1, BBox::new(1.0, 0.0, 10.0, 10.0), 0.9),
                pred(1, BBox::new(150.0, 10.0, 8.0, 8.0), 0.4),
            ],
        );
        let ev = evaluate(&ds, &preds);
        (ds, preds, ev)
    }

    fn ids(crops: &[Crop]) -> Vec<String> {
        crops
            .iter()
            .map(|c| match c.id {
                CropId::Gt(a) => format!("#{a}"),
                CropId::Pred(p) => format!("P{}", p.bbox.x),
            })
            .collect()
    }

    #[test]
    fn builds_gt_and_unmatched_pred_crops() {
        let (ds, preds, ev) = fixture();
        let crops = build(
            &ds,
            Some(&preds),
            Some(&ev),
            &ReviewState::default(),
            &Uniformity::default(),
            FrameInfo::default(),
            &ViewOptions::default(),
        );
        assert_eq!(ids(&crops), vec!["#1", "#2", "#3", "P150"]);
        assert_eq!(crops[0].status, CropStatus::Matched);
        assert_eq!(crops[0].pred.unwrap().score, 0.9);
        assert_eq!(crops[1].status, CropStatus::Fn);
        assert_eq!(crops[2].status, CropStatus::Unpredicted);
        assert_eq!(crops[3].status, CropStatus::Fp);
        let f = crops[0].frame_box();
        assert_eq!((f.x, f.x2()), (0.0, 11.0));
    }

    #[test]
    fn without_predictions_only_gt() {
        let (ds, ..) = fixture();
        let crops = build(
            &ds,
            None,
            None,
            &ReviewState::default(),
            &Uniformity::default(),
            FrameInfo::default(),
            &ViewOptions::default(),
        );
        assert_eq!(ids(&crops), vec!["#1", "#2", "#3"]);
        assert!(crops.iter().all(|c| c.status == CropStatus::Unpredicted));
    }

    #[test]
    fn filters() {
        let (ds, preds, ev) = fixture();
        let run = |opts: ViewOptions, review: &ReviewState| {
            ids(&build(
                &ds,
                Some(&preds),
                Some(&ev),
                review,
                &Uniformity::default(),
                FrameInfo::default(),
                &opts,
            ))
        };
        let none = ReviewState::default();
        assert_eq!(run(ViewOptions { fp_only: true, ..Default::default() }, &none), vec!["P150"]);
        assert_eq!(run(ViewOptions { fn_only: true, ..Default::default() }, &none), vec!["#2"]);
        assert_eq!(
            run(ViewOptions { fp_only: true, fn_only: true, ..Default::default() }, &none),
            vec!["#2", "P150"]
        );
        assert_eq!(
            run(ViewOptions { split: SplitFilter::Only(Split::Val), ..Default::default() }, &none),
            vec!["#3"]
        );
        let frames = Some(Arc::new(BTreeSet::from([2])));
        assert_eq!(run(ViewOptions { frames, ..Default::default() }, &none), vec!["#3"]);
        // Pending split moves count: image 1 moved to val, image 2 to test.
        let splits = [Some(Split::Val), Some(Split::Test)];
        let info = FrameInfo { splits: Some(&splits), ..Default::default() };
        let opts = ViewOptions { split: SplitFilter::Only(Split::Val), ..Default::default() };
        let moved = build(&ds, None, None, &none, &Uniformity::default(), info, &opts);
        assert_eq!(ids(&moved), vec!["#1", "#2"]);
        let mut no_split = splits;
        no_split[1] = None;
        let info = FrameInfo { splits: Some(&no_split), ..Default::default() };
        let opts = ViewOptions { split: SplitFilter::NoSplit, ..Default::default() };
        let none_left = build(&ds, None, None, &none, &Uniformity::default(), info, &opts);
        assert_eq!(ids(&none_left), vec!["#3"]);

        let mut review = ReviewState::default();
        review.reviewed.insert(1);
        review.reject(&pred(1, BBox::new(150.0, 10.0, 8.0, 8.0), 0.4));
        assert_eq!(
            run(ViewOptions { hide_reviewed: true, ..Default::default() }, &review),
            vec!["#2", "#3"]
        );
    }

    #[test]
    fn split_filter_reads_the_old_session_format() {
        let parse = |j: &str| serde_json::from_str::<SplitFilter>(j).unwrap();
        assert_eq!(parse("null"), SplitFilter::All);
        assert_eq!(parse("\"Test\""), SplitFilter::Only(Split::Test));
        assert_eq!(parse("\"None\""), SplitFilter::NoSplit);
        for f in SplitFilter::choices() {
            assert_eq!(parse(&serde_json::to_string(&f).unwrap()), f);
        }
    }

    #[test]
    fn sorting_puts_missing_values_last_in_both_directions() {
        let (ds, preds, ev) = fixture();
        let run = |sort, descending| {
            ids(&build(
                &ds,
                Some(&preds),
                Some(&ev),
                &ReviewState::default(),
                &Uniformity::default(),
                FrameInfo::default(),
                &ViewOptions { sort, descending, ..Default::default() },
            ))
        };
        assert_eq!(run(SortKey::Width, false), vec!["P150", "#1", "#2", "#3"]);
        assert_eq!(run(SortKey::Height, true), vec!["#2", "#1", "P150", "#3"]);
        assert_eq!(run(SortKey::Iou, false), vec!["#2", "P150", "#1", "#3"]);
        assert_eq!(run(SortKey::Iou, true), vec!["#1", "#2", "P150", "#3"]);
        assert_eq!(run(SortKey::Confidence, false), vec!["P150", "#1", "#2", "#3"]);
        assert_eq!(run(SortKey::Id, true), vec!["#3", "#2", "#1", "P150"]);
        // #1 10x10 (1.0, 100), #2 20x30 (0.67, 600), #3 40x5 (8.0, 200), P150 8x8 (1.0, 64)
        assert_eq!(run(SortKey::AspectRatio, false), vec!["#2", "#1", "P150", "#3"]);
        assert_eq!(run(SortKey::Area, true), vec!["#2", "#3", "#1", "P150"]);
    }
}
