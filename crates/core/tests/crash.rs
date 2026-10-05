//! Kills a process (SIGKILL) at random moments while it records edits and
//! saves versions, then checks that the dataset always reopens cleanly:
//! every annotation file parses and the journal replays.

use std::path::Path;
use std::process::Command;
use std::time::Duration;

use fastlabel_core::coco::CocoFile;
use fastlabel_core::{BBox, Op, Project, versions};

const CHILD_ENV: &str = "FASTLABEL_CRASH_CHILD";

fn write_dataset(dir: &Path) {
    let mut coco: CocoFile = serde_json::from_value(serde_json::json!({
        "images": [], "annotations": [],
        "categories": [{"id": 1, "name": "bib", "supercategory": ""}]
    }))
    .unwrap();
    for id in 1..=2000i64 {
        coco.images.push(serde_json::from_value(serde_json::json!({
            "id": id, "file_name": format!("images/train/{id}.jpg"), "width": 1920, "height": 1080
        })).unwrap());
        coco.annotations.push(fastlabel_core::coco::CocoAnnotation::new(
            id,
            id,
            1,
            BBox::new(10.0, 10.0, 50.0, 30.0),
        ));
    }
    coco.save(&dir.join("annotations.v001.json")).unwrap();
}

/// Child mode: edit and save forever.
fn child(dir: &Path) -> ! {
    let mut i = 0i64;
    loop {
        let mut p = Project::open(dir).unwrap();
        for _ in 0..50 {
            i += 1;
            let ann = p.dataset.annotations()[(i as usize * 7) % p.dataset.annotations().len()].id;
            p.record(Op::SetBox {
                ann,
                bbox: BBox::new(f64::from((i % 500) as i32), 5.0, 40.0, 20.0),
            })
            .unwrap();
        }
        p.save().unwrap();
    }
}

#[test]
fn sigkill_during_edits_and_saves_never_corrupts() {
    if let Ok(dir) = std::env::var(CHILD_ENV) {
        child(Path::new(&dir));
    }
    let tmp = tempfile::tempdir().unwrap();
    write_dataset(tmp.path());
    let exe = std::env::current_exe().unwrap();
    for round in 0..8u64 {
        let mut c = Command::new(&exe)
            .args(["--exact", "sigkill_during_edits_and_saves_never_corrupts", "--nocapture"])
            .env(CHILD_ENV, tmp.path())
            .spawn()
            .unwrap();
        std::thread::sleep(Duration::from_millis(100 + (round * 37) % 200));
        c.kill().unwrap();
        c.wait().unwrap();

        for v in versions::discover(tmp.path()).unwrap() {
            CocoFile::load(&v.path)
                .unwrap_or_else(|e| panic!("round {round}: {} corrupt: {e}", v.file_name()));
        }
        let p = Project::open(tmp.path())
            .unwrap_or_else(|e| panic!("round {round}: reopen failed: {e}"));
        assert_eq!(p.dataset.annotations().len(), 2000);
    }
    let n = versions::discover(tmp.path()).unwrap().len();
    assert!(n > 2, "the child saved several versions ({n})");
}
