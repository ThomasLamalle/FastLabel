//! Random and malformed inputs must give errors, never panics.

use std::path::{Path, PathBuf};

use fastlabel_core::coco::{CocoAnnotation, CocoCategory, CocoFile, CocoImage};
use fastlabel_core::commit::apply;
use fastlabel_core::matching::{Thresholds, evaluate};
use fastlabel_core::predictions::{PredBox, PredImage, PredictionFile};
use fastlabel_core::review::ReviewState;
use fastlabel_core::{
    AnnMark, BBox, Dataset, EditLog, Op, PredMark, PredRef, Predictions, journal, yolo,
};
use proptest::prelude::*;
use serde_json::Map;

fn arb_box() -> impl Strategy<Value = BBox> {
    (-1e6..1e6f64, -1e6..1e6f64, -10.0..1e6f64, -10.0..1e6f64)
        .prop_map(|(x, y, w, h)| BBox::new(x, y, w, h))
}

/// Mostly-valid datasets (unique ids, existing image references) with
/// extreme geometry, so the whole pipeline is exercised.
fn arb_coco() -> impl Strategy<Value = CocoFile> {
    let images = prop::collection::vec((0..3000u32, 0..3000u32), 1..20);
    let anns = prop::collection::vec((0..1000usize, 1..4i64, arb_box()), 0..60);
    (images, anns).prop_map(|(images, anns)| {
        let n = images.len();
        CocoFile {
            images: images
                .into_iter()
                .enumerate()
                .map(|(i, (w, h))| CocoImage {
                    id: i as i64 + 1,
                    file_name: format!("images/train/{i}.jpg"),
                    width: w,
                    height: h,
                    tags: vec![],
                    extra: Map::new(),
                })
                .collect(),
            annotations: anns
                .into_iter()
                .enumerate()
                .map(|(i, (img, cat, b))| {
                    CocoAnnotation::new(i as i64 + 1, (img % n) as i64 + 1, cat, b)
                })
                .collect(),
            categories: (1..=3)
                .map(|id| CocoCategory {
                    id,
                    name: format!("c{id}"),
                    supercategory: String::new(),
                    extra: Map::new(),
                })
                .collect(),
            extra: Map::new(),
        }
    })
}

fn write_tmp(bytes: &[u8]) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("f.json");
    std::fs::write(&p, bytes).unwrap();
    (dir, p)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    #[test]
    fn loaders_never_panic_on_random_bytes(bytes in prop::collection::vec(any::<u8>(), 0..400)) {
        let (_d, p) = write_tmp(&bytes);
        let _ = CocoFile::load(&p);
        let _ = PredictionFile::load(&p);
        let _ = journal::read(&p);
        let _ = yolo::read_class_names(&p);
    }

    #[test]
    fn loaders_never_panic_on_json_like_text(text in r#"[\[\]{}:,"a-z0-9 .\-]{0,300}"#) {
        let (_d, p) = write_tmp(text.as_bytes());
        let _ = CocoFile::load(&p);
        let _ = PredictionFile::load(&p);
        let _ = journal::read(&p);
    }

    #[test]
    fn whole_pipeline_never_panics(
        coco in arb_coco(),
        preds in prop::collection::vec((0..20usize, 0..4i64, arb_box(), 0.0..1.0f64), 0..40),
        ops in prop::collection::vec((0..8u8, 0..60i64, arb_box()), 0..40),
    ) {
        let (ds, _) = Dataset::new(PathBuf::from("/d"), coco, Path::new("x")).expect("generated dataset is valid");
        // Predictions through the file format (validation path).
        let mut file = PredictionFile { version: 1, model: "m".into(), confidence: 0.1, images: Default::default(), extra: Map::new() };
        for (idx, cat, bbox, score) in &preds {
            if let Some(img) = ds.images().get(*idx) {
                file.images.entry(img.id.to_string()).or_insert_with(|| PredImage {
                    file_name: img.file_name.clone(), width: img.width, height: img.height, boxes: vec![],
                }).boxes.push(PredBox { id: -1, category_id: *cat, bbox: *bbox, score: *score });
            }
        }
        let predictions = Predictions::from_file(&file, &ds);
        let ev = evaluate(&ds, &predictions, Thresholds::default());
        prop_assert_eq!(ev.gt.len(), ds.annotations().len());

        let mut log = EditLog::new(ds.coco.next_annotation_id());
        let image_id = ds.images().first().map_or(1, |i| i.id);
        for (kind, ann, bbox) in ops {
            let pred = PredRef { image_id, category_id: 1, bbox, score: 0.5 };
            log.push(match kind {
                0 => Op::AnnMark { ann, mark: AnnMark::Delete },
                1 => Op::AnnMark { ann, mark: AnnMark::Replace(pred) },
                2 => Op::PredMark { pred, mark: PredMark::Accept },
                3 => Op::PredMark { pred, mark: PredMark::Reject },
                4 => Op::FrameDelete { image: ann % 20 + 1 },
                5 => Op::SetBox { ann, bbox },
                6 => Op::AddBox { ann: 1000 + ann, image: image_id, category: 1, bbox },
                _ => Op::SetTags { image: ann % 20 + 1, tags: vec![" x ".into()] },
            });
        }
        let out = apply(&ds, log.state(), &ReviewState::default());
        // The result is always a valid dataset again, and every new box
        // (drawn or accepted) lies inside its image. Invalid edits of existing
        // boxes are skipped, keeping the original geometry.
        let (again, _) = Dataset::new(PathBuf::from("/d"), out.coco, Path::new("y")).expect("applied dataset is valid");
        for a in again.annotations() {
            let img = again.image(a.image_id).unwrap();
            if ds.annotation(a.id).is_none() {
                prop_assert!(a.bbox.x >= 0.0 && a.bbox.y >= 0.0, "{:?}", a.bbox);
                prop_assert!(a.bbox.x2() <= f64::from(img.width) + 1e-9 && a.bbox.y2() <= f64::from(img.height) + 1e-9);
                prop_assert!(a.bbox.w >= 1.0 && a.bbox.h >= 1.0);
            }
        }
    }
}
