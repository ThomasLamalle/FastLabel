//! Model predictions, stored separately from annotations in `predictions.json`.

use std::hash::{Hash, Hasher};
use std::path::Path;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::coco::{CategoryId, ImageId};
use crate::dataset::Dataset;
use crate::error::{Error, Result};
use crate::fsutil;
use crate::geom::BBox;

pub const PREDICTIONS_FILE: &str = "predictions.json";
pub const FORMAT_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PredictionFile {
    pub version: u32,
    pub model: String,
    pub confidence: f64,
    /// Keyed by image id (as a string, JSON object keys).
    pub images: IndexMap<String, PredImage>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PredImage {
    pub file_name: String,
    pub width: u32,
    pub height: u32,
    pub boxes: Vec<PredBox>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PredBox {
    #[serde(default = "minus_one")]
    pub id: i64,
    pub category_id: CategoryId,
    pub bbox: BBox,
    pub score: f64,
}

fn minus_one() -> i64 {
    -1
}

impl PredictionFile {
    pub fn load(path: &Path) -> Result<Self> {
        let file: Self = fsutil::read_json(path)?;
        if file.version != FORMAT_VERSION {
            return Err(Error::invalid(
                path,
                format!("unsupported predictions version {}", file.version),
            ));
        }
        Ok(file)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        fsutil::write_json_atomic(path, self)
    }
}

/// Identity of a prediction: image, class and exact geometry. Self-contained,
/// so pending edits survive a reload even if `predictions.json` changed.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PredRef {
    pub image_id: ImageId,
    pub category_id: CategoryId,
    pub bbox: BBox,
    pub score: f64,
}

impl PartialEq for PredRef {
    fn eq(&self, other: &Self) -> bool {
        self.image_id == other.image_id
            && self.category_id == other.category_id
            && self.bbox.bits() == other.bbox.bits()
    }
}

impl Eq for PredRef {}

impl Hash for PredRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.image_id.hash(state);
        self.category_id.hash(state);
        self.bbox.bits().hash(state);
    }
}

/// Predictions validated against a dataset, indexed like its images.
#[derive(Clone, Debug, Default)]
pub struct Predictions {
    pub model: String,
    pub confidence: f64,
    /// `None` = image not evaluated ("unpredicted"), indexed by image index.
    per_image: Vec<Option<Vec<PredRef>>>,
    pub warnings: Vec<String>,
}

impl Predictions {
    pub fn empty(model: impl Into<String>, confidence: f64) -> Self {
        Self { model: model.into(), confidence, ..Self::default() }
    }

    /// Keeps only entries whose image id, file name and size match the
    /// dataset; anything else is reported in `warnings` and left unpredicted.
    pub fn from_file(file: &PredictionFile, ds: &Dataset) -> Self {
        let mut per_image = vec![None; ds.images().len()];
        let mut warnings = Vec::new();
        let mut mismatched = 0usize;
        let mut unknown_categories = 0usize;
        for (key, entry) in &file.images {
            let Some(idx) = key.parse::<ImageId>().ok().and_then(|id| ds.image_idx(id)) else {
                mismatched += 1;
                continue;
            };
            let img = &ds.images()[idx];
            if img.file_name != entry.file_name
                || img.width != entry.width
                || img.height != entry.height
            {
                mismatched += 1;
                continue;
            }
            let boxes = entry
                .boxes
                .iter()
                .filter(|b| {
                    let known = ds.category_name(b.category_id).is_some();
                    unknown_categories += usize::from(!known);
                    known && b.bbox.is_finite() && b.score.is_finite()
                })
                .map(|b| PredRef {
                    image_id: img.id,
                    category_id: b.category_id,
                    bbox: b.bbox,
                    score: b.score,
                })
                .collect();
            per_image[idx] = Some(boxes);
        }
        if mismatched > 0 {
            warnings.push(format!(
                "{mismatched} prediction entries do not match the dataset images and were ignored"
            ));
        }
        if unknown_categories > 0 {
            warnings.push(format!(
                "{unknown_categories} predictions have an unknown category and were ignored"
            ));
        }
        Self { model: file.model.clone(), confidence: file.confidence, per_image, warnings }
    }

    /// Predictions of the image at `image_idx`, `None` when not evaluated.
    pub fn of(&self, image_idx: usize) -> Option<&[PredRef]> {
        self.per_image.get(image_idx)?.as_deref()
    }

    pub fn is_empty(&self) -> bool {
        self.per_image.iter().all(Option::is_none)
    }

    pub fn evaluated_images(&self) -> usize {
        self.per_image.iter().filter(|p| p.is_some()).count()
    }

    pub fn len(&self) -> usize {
        self.per_image.iter().flatten().map(Vec::len).sum()
    }

    /// Replaces the predictions of one image (single-frame model run).
    pub fn set_image(&mut self, image_idx: usize, preds: Vec<PredRef>) {
        if self.per_image.len() <= image_idx {
            self.per_image.resize(image_idx + 1, None);
        }
        self.per_image[image_idx] = Some(preds);
    }

    /// Serialisable form, for writing `predictions.json`.
    pub fn to_file(&self, ds: &Dataset) -> PredictionFile {
        let images = ds
            .images()
            .iter()
            .enumerate()
            .filter_map(|(idx, img)| {
                let preds = self.of(idx)?;
                let boxes = preds
                    .iter()
                    .map(|p| PredBox {
                        id: -1,
                        category_id: p.category_id,
                        bbox: p.bbox,
                        score: p.score,
                    })
                    .collect();
                Some((
                    img.id.to_string(),
                    PredImage {
                        file_name: img.file_name.clone(),
                        width: img.width,
                        height: img.height,
                        boxes,
                    },
                ))
            })
            .collect();
        PredictionFile {
            version: FORMAT_VERSION,
            model: self.model.clone(),
            confidence: self.confidence,
            images,
            extra: Map::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::test_util::dataset;

    fn file() -> PredictionFile {
        serde_json::from_value(serde_json::json!({
            "version": 1, "model": "m.onnx", "confidence": 0.1,
            "images": {
                "1": {"file_name": "images/train/0001.jpg", "width": 100, "height": 100,
                      "boxes": [{"id": -1, "category_id": 1, "bbox": [1, 2, 3, 4], "score": 0.9},
                                {"id": -1, "category_id": 7, "bbox": [1, 2, 3, 4], "score": 0.9}]},
                "2": {"file_name": "images/train/other.jpg", "width": 100, "height": 100, "boxes": []},
                "99": {"file_name": "x.jpg", "width": 1, "height": 1, "boxes": []}
            }
        }))
        .unwrap()
    }

    #[test]
    fn validates_against_dataset() {
        let ds = dataset(3, &[]);
        let preds = Predictions::from_file(&file(), &ds);
        assert_eq!(preds.of(0).unwrap().len(), 1);
        assert!(preds.of(1).is_none(), "file name mismatch → unpredicted");
        assert!(preds.of(2).is_none());
        assert_eq!(preds.warnings.len(), 2);
        assert_eq!(preds.evaluated_images(), 1);
    }

    #[test]
    fn to_file_round_trips() {
        let ds = dataset(3, &[]);
        let preds = Predictions::from_file(&file(), &ds);
        let again = Predictions::from_file(&preds.to_file(&ds), &ds);
        assert_eq!(again.of(0), preds.of(0));
        assert!(again.warnings.is_empty());
    }

    #[test]
    fn pred_ref_identity_ignores_score() {
        let a = PredRef {
            image_id: 1,
            category_id: 1,
            bbox: BBox::new(1.0, 2.0, 3.0, 4.0),
            score: 0.5,
        };
        let b = PredRef { score: 0.7, ..a };
        assert_eq!(a, b);
        assert_ne!(a, PredRef { bbox: BBox::new(1.0, 2.0, 3.0, 4.5), ..a });
    }
}
