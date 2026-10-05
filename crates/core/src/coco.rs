//! COCO detection file model. Unknown fields are kept in `extra` so a
//! load/save round-trip never drops data written by other tools.

use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::error::{Error, Result};
use crate::fsutil;
use crate::geom::BBox;

pub type ImageId = i64;
pub type AnnId = i64;
pub type CategoryId = i64;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CocoFile {
    pub images: Vec<CocoImage>,
    pub annotations: Vec<CocoAnnotation>,
    pub categories: Vec<CocoCategory>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CocoImage {
    pub id: ImageId,
    pub file_name: String,
    pub width: u32,
    pub height: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CocoAnnotation {
    pub id: AnnId,
    pub image_id: ImageId,
    pub category_id: CategoryId,
    pub bbox: BBox,
    #[serde(default)]
    pub area: f64,
    #[serde(default)]
    pub iscrowd: u8,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CocoCategory {
    pub id: CategoryId,
    pub name: String,
    #[serde(default)]
    pub supercategory: String,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl CocoAnnotation {
    pub fn new(id: AnnId, image_id: ImageId, category_id: CategoryId, bbox: BBox) -> Self {
        Self { id, image_id, category_id, bbox, area: bbox.area(), iscrowd: 0, extra: Map::new() }
    }
}

/// Legacy deletion flag written by the previous Python tool.
pub(crate) const LEGACY_DELETE_KEY: &str = "to_delete";

pub(crate) fn take_legacy_flag(extra: &mut Map<String, Value>) -> bool {
    matches!(extra.remove(LEGACY_DELETE_KEY), Some(Value::Bool(true)))
}

impl CocoFile {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|e| Error::io(path, e))?;
        serde_json::from_slice(&bytes).map_err(|e| Error::json(path, e))
    }

    /// Atomically writes the file with 4-space indentation (same layout as
    /// the previous tool, so versions diff cleanly).
    pub fn save(&self, path: &Path) -> Result<()> {
        fsutil::write_json_atomic(path, self)
    }

    pub fn next_annotation_id(&self) -> AnnId {
        self.annotations.iter().map(|a| a.id).max().unwrap_or(0) + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
        "images": [{"id": 1, "file_name": "images/train/a.jpg", "width": 1920, "height": 1080, "to_delete": false, "custom": 3}],
        "annotations": [{"id": 7, "image_id": 1, "category_id": 1, "bbox": [1095.2171615999998, 270.5, 94.0, 49.1], "area": 4617.8, "iscrowd": 0, "to_delete": true}],
        "categories": [{"id": 1, "name": "bib", "supercategory": ""}],
        "info": {"year": 2026}
    }"#;

    #[test]
    fn round_trip_preserves_unknown_fields_and_floats() {
        let coco: CocoFile = serde_json::from_str(SAMPLE).unwrap();
        assert_eq!(coco.images[0].extra["custom"], 3);
        assert_eq!(coco.annotations[0].bbox.x, 1095.2171615999998);
        assert_eq!(coco.extra["info"]["year"], 2026);
        let text = serde_json::to_string(&coco).unwrap();
        assert!(text.contains("1095.2171615999998"));
        let back: CocoFile = serde_json::from_str(&text).unwrap();
        assert_eq!(back, coco);
    }

    #[test]
    fn legacy_flag_is_extracted() {
        let mut coco: CocoFile = serde_json::from_str(SAMPLE).unwrap();
        assert!(take_legacy_flag(&mut coco.annotations[0].extra));
        assert!(!coco.annotations[0].extra.contains_key(LEGACY_DELETE_KEY));
        assert!(!take_legacy_flag(&mut coco.images[0].extra));
    }
}
