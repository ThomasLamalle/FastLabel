//! Scale check on a synthetic 100k-image dataset:
//! `cargo run --release -p fastlabel-core --example synthetic`

use std::path::{Path, PathBuf};
use std::time::Instant;

use fastlabel_core::coco::{CocoAnnotation, CocoCategory, CocoFile, CocoImage};
use fastlabel_core::commit::apply;
use fastlabel_core::matching::evaluate;
use fastlabel_core::review::ReviewState;
use fastlabel_core::{AnnMark, BBox, Dataset, EditLog, Op, PredRef, Predictions};
use serde_json::Map;

fn main() {
    const IMAGES: i64 = 100_000;
    let mut coco = CocoFile {
        categories: vec![CocoCategory {
            id: 1,
            name: "bib".into(),
            supercategory: String::new(),
            extra: Map::new(),
        }],
        ..CocoFile::default()
    };
    let mut preds = Predictions::empty("synthetic", 0.1);
    let mut next = 1;
    for id in 1..=IMAGES {
        coco.images.push(CocoImage {
            id,
            file_name: format!("images/train/{id:07}.jpg"),
            width: 1920,
            height: 1080,
            tags: vec![],
            extra: Map::new(),
        });
        let mut p = Vec::new();
        for k in 0..(id % 5) {
            let b = BBox::new(100.0 + 300.0 * k as f64, 200.0, 90.0, 50.0);
            coco.annotations.push(CocoAnnotation::new(next, id, 1, b));
            next += 1;
            p.push(PredRef {
                image_id: id,
                category_id: 1,
                bbox: BBox::new(b.x + 3.0, b.y, b.w, b.h),
                score: 0.8,
            });
        }
        p.push(PredRef {
            image_id: id,
            category_id: 1,
            bbox: BBox::new(1500.0, 800.0, 60.0, 40.0),
            score: 0.2,
        });
        preds.set_image(id as usize - 1, p);
    }
    println!(
        "{} images, {} annotations, {} predictions",
        coco.images.len(),
        coco.annotations.len(),
        preds.len()
    );

    let t = Instant::now();
    let (ds, _) = Dataset::new(PathBuf::from("/data"), coco, Path::new("synthetic")).unwrap();
    println!("index        {:?}", t.elapsed());

    let t = Instant::now();
    let ev = evaluate(&ds, &preds);
    println!("evaluate     {:?} {:?}", t.elapsed(), ev.counts);

    let mut log = EditLog::new(ds.coco.next_annotation_id());
    for ann in (1..next).step_by(7) {
        log.push(Op::AnnMark { ann, mark: AnnMark::Delete });
    }
    for image in (1..IMAGES).step_by(50) {
        log.push(Op::FrameDelete { image });
    }
    let t = Instant::now();
    log.undo();
    println!("undo ({} ops) {:?}", log.len() + 1, t.elapsed());

    let t = Instant::now();
    let applied = apply(&ds, log.state(), &ReviewState::default());
    println!("apply        {:?} {:?}", t.elapsed(), applied.summary);

    let dir = tempfile::tempdir().unwrap();
    let t = Instant::now();
    applied.coco.save(&dir.path().join("annotations.v001.json")).unwrap();
    println!("save         {:?}", t.elapsed());
    let t = Instant::now();
    CocoFile::load(&dir.path().join("annotations.v001.json")).unwrap();
    println!("load         {:?}", t.elapsed());
}
