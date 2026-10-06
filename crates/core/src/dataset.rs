//! In-memory dataset with id indices.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};

use crate::coco::{
    AnnId, CategoryId, CocoAnnotation, CocoCategory, CocoFile, CocoImage, ImageId, take_legacy_flag,
};
use crate::error::{Error, Result};

/// Image field holding the split (`"train"`, `"val"`, `"test"`). The split
/// is data, independent of where the file is; exports lay files out by split.
pub const SPLIT_FIELD: &str = "split";

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum Split {
    Train,
    Val,
    Test,
}

impl Split {
    pub const ALL: [Split; 3] = [Split::Train, Split::Val, Split::Test];

    /// `train`, `val` / `valid` / `validation` or `test`, any case.
    pub fn parse(name: &str) -> Option<Split> {
        match name.trim().to_ascii_lowercase().as_str() {
            "train" => Some(Split::Train),
            "val" | "valid" | "validation" => Some(Split::Val),
            "test" => Some(Split::Test),
            _ => None,
        }
    }

    /// Split from any path component named train / val / valid / test.
    pub fn from_path(file_name: &str) -> Option<Split> {
        Path::new(file_name).components().rev().skip(1).find_map(|c| match c {
            Component::Normal(s) => Split::parse(s.to_str()?),
            _ => None,
        })
    }

    /// Split of an image: its `split` field, else its path (older files).
    pub fn of_image(image: &CocoImage) -> Option<Split> {
        image
            .extra
            .get(SPLIT_FIELD)
            .and_then(|v| v.as_str())
            .and_then(Split::parse)
            .or_else(|| Split::from_path(&image.file_name))
    }

    pub fn name(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Val => "val",
            Split::Test => "test",
        }
    }
}

/// Deletion marks found in files written by the previous tool. They become
/// pending edits when the dataset is opened and are never written back.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LegacyMarks {
    pub deleted_frames: Vec<ImageId>,
    pub deleted_annotations: Vec<AnnId>,
}

#[derive(Clone, Debug)]
pub struct Dataset {
    /// Directory image `file_name`s are relative to.
    pub root: PathBuf,
    pub coco: CocoFile,
    image_index: HashMap<ImageId, usize>,
    ann_index: HashMap<AnnId, usize>,
    /// Annotation indices per image index.
    anns_by_image: Vec<Vec<usize>>,
    splits: Vec<Option<Split>>,
}

impl Dataset {
    /// Builds indices, rejecting duplicate ids and dangling image references.
    /// Returns legacy `to_delete` marks separately.
    pub fn new(root: PathBuf, mut coco: CocoFile, source: &Path) -> Result<(Self, LegacyMarks)> {
        let mut legacy = LegacyMarks::default();
        for img in &mut coco.images {
            if take_legacy_flag(&mut img.extra) {
                legacy.deleted_frames.push(img.id);
            }
        }
        for ann in &mut coco.annotations {
            if take_legacy_flag(&mut ann.extra) {
                legacy.deleted_annotations.push(ann.id);
            }
        }

        let mut image_index = HashMap::with_capacity(coco.images.len());
        for (i, img) in coco.images.iter().enumerate() {
            if image_index.insert(img.id, i).is_some() {
                return Err(Error::invalid(source, format!("duplicate image id {}", img.id)));
            }
        }
        let mut ann_index = HashMap::with_capacity(coco.annotations.len());
        let mut anns_by_image = vec![Vec::new(); coco.images.len()];
        for (i, ann) in coco.annotations.iter().enumerate() {
            if ann_index.insert(ann.id, i).is_some() {
                return Err(Error::invalid(source, format!("duplicate annotation id {}", ann.id)));
            }
            let Some(&img) = image_index.get(&ann.image_id) else {
                return Err(Error::invalid(
                    source,
                    format!("annotation {} refers to missing image {}", ann.id, ann.image_id),
                ));
            };
            anns_by_image[img].push(i);
        }
        let splits = coco.images.iter().map(Split::of_image).collect();
        Ok((Self { root, coco, image_index, ann_index, anns_by_image, splits }, legacy))
    }

    pub fn images(&self) -> &[CocoImage] {
        &self.coco.images
    }

    pub fn annotations(&self) -> &[CocoAnnotation] {
        &self.coco.annotations
    }

    pub fn categories(&self) -> &[CocoCategory] {
        &self.coco.categories
    }

    pub fn image_idx(&self, id: ImageId) -> Option<usize> {
        self.image_index.get(&id).copied()
    }

    pub fn image(&self, id: ImageId) -> Option<&CocoImage> {
        self.image_idx(id).map(|i| &self.coco.images[i])
    }

    pub fn ann_idx(&self, id: AnnId) -> Option<usize> {
        self.ann_index.get(&id).copied()
    }

    pub fn annotation(&self, id: AnnId) -> Option<&CocoAnnotation> {
        self.ann_idx(id).map(|i| &self.coco.annotations[i])
    }

    /// Annotations of the image at `image_idx`.
    pub fn annotations_of(&self, image_idx: usize) -> impl Iterator<Item = &CocoAnnotation> {
        self.anns_by_image[image_idx].iter().map(|&i| &self.coco.annotations[i])
    }

    pub fn split_of(&self, image_idx: usize) -> Option<Split> {
        self.splits[image_idx]
    }

    pub fn image_path(&self, image: &CocoImage) -> PathBuf {
        self.root.join(&image.file_name)
    }

    pub fn category_name(&self, id: CategoryId) -> Option<&str> {
        self.coco.categories.iter().find(|c| c.id == id).map(|c| c.name.as_str())
    }

    /// Category id for a class name, ignoring case and surrounding spaces.
    pub fn category_by_name(&self, name: &str) -> Option<CategoryId> {
        let name = name.trim();
        self.coco.categories.iter().find(|c| c.name.trim().eq_ignore_ascii_case(name)).map(|c| c.id)
    }
}

#[cfg(test)]
pub(crate) mod test_util {
    use super::*;
    use crate::geom::BBox;
    use serde_json::Map;

    /// Dataset with `n_images` 100x100 images; `anns` = (image_id, bbox).
    pub fn dataset(n_images: i64, anns: &[(ImageId, BBox)]) -> Dataset {
        let coco = CocoFile {
            images: (1..=n_images)
                .map(|id| CocoImage {
                    id,
                    file_name: format!("images/train/{id:04}.jpg"),
                    width: 100,
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
        Dataset::new(PathBuf::from("/data"), coco, Path::new("test.json")).unwrap().0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::BBox;

    #[test]
    fn split_from_path() {
        assert_eq!(Split::from_path("images/train/a.jpg"), Some(Split::Train));
        assert_eq!(Split::from_path("images/valid/a.jpg"), Some(Split::Val));
        assert_eq!(Split::from_path("test/x/a.jpg"), Some(Split::Test));
        assert_eq!(Split::from_path("images/a.jpg"), None);
        assert_eq!(Split::from_path("images/train.jpg"), None);
    }

    #[test]
    fn split_field_wins_over_the_path() {
        let mut coco = test_util::dataset(3, &[]).coco;
        coco.images[0].extra.insert(SPLIT_FIELD.into(), "valid".into());
        coco.images[1].extra.insert(SPLIT_FIELD.into(), "nonsense".into());
        coco.images[2].file_name = "images/c.jpg".into();
        let ds = Dataset::new(PathBuf::new(), coco, Path::new("x")).unwrap().0;
        assert_eq!(ds.split_of(0), Some(Split::Val));
        assert_eq!(ds.split_of(1), Some(Split::Train), "unknown value: the path decides");
        assert_eq!(ds.split_of(2), None);
    }

    #[test]
    fn indexes_and_groups_annotations() {
        let ds = test_util::dataset(
            3,
            &[
                (1, BBox::new(0.0, 0.0, 5.0, 5.0)),
                (3, BBox::new(1.0, 1.0, 5.0, 5.0)),
                (1, BBox::new(2.0, 2.0, 5.0, 5.0)),
            ],
        );
        assert_eq!(ds.annotations_of(0).count(), 2);
        assert_eq!(ds.annotations_of(1).count(), 0);
        assert_eq!(ds.annotations_of(2).next().unwrap().id, 2);
        assert_eq!(ds.annotation(3).unwrap().image_id, 1);
        assert_eq!(ds.split_of(0), Some(Split::Train));
        assert_eq!(ds.category_by_name(" BIB "), Some(1));
    }

    #[test]
    fn rejects_dangling_annotation_and_extracts_legacy_marks() {
        let mut coco = test_util::dataset(1, &[(1, BBox::new(0.0, 0.0, 5.0, 5.0))]).coco;
        coco.images[0].extra.insert("to_delete".into(), true.into());
        coco.annotations[0].extra.insert("to_delete".into(), true.into());
        let (_, legacy) = Dataset::new(PathBuf::new(), coco.clone(), Path::new("x")).unwrap();
        assert_eq!(legacy, LegacyMarks { deleted_frames: vec![1], deleted_annotations: vec![1] });

        coco.annotations[0].image_id = 99;
        assert!(Dataset::new(PathBuf::new(), coco, Path::new("x")).is_err());
    }
}
