//! Headless UI tests: the real app driven through egui_kittest (offscreen
//! wgpu rendering, synthetic key presses).

use std::path::{Path, PathBuf};
use std::time::Duration;

use eframe::egui::{self, Key, Modifiers};
use egui_kittest::Harness;
use fastlabel_core::coco::{CocoAnnotation, CocoCategory, CocoFile, CocoImage};
use fastlabel_core::{BBox, PredRef, Predictions};
use fastlabel_media::decode::encode_jpeg;
use serde_json::Map;

use crate::app::FastLabelApp;
use crate::session::Session;
use crate::workspace::Mark;

/// 6 frames of 320x240, two GT boxes each; predictions match the first box
/// of every frame and add one false positive on frame 1.
fn synthetic_dataset(dir: &Path) {
    let mut coco = CocoFile {
        categories: vec![CocoCategory {
            id: 1,
            name: "bib".into(),
            supercategory: String::new(),
            extra: Map::new(),
        }],
        ..CocoFile::default()
    };
    let mut preds = Vec::new();
    for id in 1..=6i64 {
        let file_name = format!("images/train/{id:03}.jpg");
        let img = image::RgbImage::from_fn(320, 240, |x, y| {
            let on_box = (40..120).contains(&x) && (60..110).contains(&y);
            if on_box {
                image::Rgb([240, 240, 240])
            } else {
                image::Rgb([(x / 2) as u8, (y / 2) as u8, (id * 40) as u8])
            }
        });
        let path = dir.join(&file_name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, encode_jpeg(&img, 90).unwrap()).unwrap();
        coco.images.push(CocoImage {
            id,
            file_name,
            width: 320,
            height: 240,
            tags: vec![],
            extra: Map::new(),
        });
        let a = BBox::new(40.0, 60.0, 80.0, 50.0);
        let b = BBox::new(200.0, 120.0, 60.0, 40.0);
        let n = coco.annotations.len() as i64;
        coco.annotations.push(CocoAnnotation::new(n + 1, id, 1, a));
        coco.annotations.push(CocoAnnotation::new(n + 2, id, 1, b));
        preds.push(PredRef {
            image_id: id,
            category_id: 1,
            bbox: BBox::new(42.0, 61.0, 80.0, 50.0),
            score: 0.9,
        });
    }
    preds.push(PredRef {
        image_id: 1,
        category_id: 1,
        bbox: BBox::new(250.0, 20.0, 40.0, 30.0),
        score: 0.35,
    });
    coco.save(&dir.join("annotations.v001.json")).unwrap();
    let ds = fastlabel_core::Dataset::new(dir.to_path_buf(), coco, Path::new("t")).unwrap().0;
    let mut p = Predictions::empty("synthetic.onnx", 0.1);
    for (idx, img) in ds.images().iter().enumerate() {
        p.set_image(idx, preds.iter().filter(|r| r.image_id == img.id).copied().collect());
    }
    p.to_file(&ds).save(&dir.join("predictions.json")).unwrap();
}

fn harness(dataset: PathBuf, size: egui::Vec2) -> Harness<'static, FastLabelApp> {
    let session = Session { columns: 4, ..Session::default() };
    Harness::builder()
        .with_size(size)
        .build_eframe(move |cc| FastLabelApp::new(cc, session, None, Some(dataset)))
}

/// Steps until every thumbnail is loaded.
fn settle(h: &mut Harness<'static, FastLabelApp>) {
    for _ in 0..200 {
        h.step();
        if h.state().thumbs_busy() == 0 {
            h.run_steps(3);
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("thumbnails did not finish loading");
}

fn save_shot(h: &mut Harness<'static, FastLabelApp>, name: &str) {
    if let Ok(dir) = std::env::var("FASTLABEL_SHOT_DIR") {
        let img = h.render().expect("render");
        img.save(Path::new(&dir).join(name)).unwrap();
    }
}

#[test]
fn keyboard_review_flow_and_save() {
    let tmp = tempfile::tempdir().unwrap();
    synthetic_dataset(tmp.path());
    let mut h = harness(tmp.path().to_path_buf(), egui::vec2(1100.0, 700.0));
    settle(&mut h);
    // 12 GT crops + 1 FP crop.
    assert_eq!(h.state().ws().crops.len(), 13);
    save_shot(&mut h, "01_opened.png");

    h.key_press(Key::D); // delete #1, advance to #2
    h.key_press(Key::L); // → #3
    h.key_press(Key::F); // frame 2, advance past its crops to #5
    h.run_steps(2);
    let ws = h.state().ws();
    assert_eq!(ws.marks_of(&ws.crops[0]).mark, Some(Mark::Delete));
    assert!(ws.marks_of(&ws.crops[3]).frame_deleted);
    assert_eq!(ws.selected, 4);

    h.key_press_modifiers(Modifiers::SHIFT, Key::G); // confirm row 1 (#5..#8)
    h.key_press(Key::J);
    h.key_press(Key::End); // the FP crop
    h.key_press(Key::G); // accept it
    h.run_steps(2);
    let ws = h.state().ws();
    assert!((4..8).all(|i| ws.marks_of(&ws.crops[i]).mark == Some(Mark::Good)));
    assert_eq!(ws.marks_of(&ws.crops[12]).mark, Some(Mark::Good));
    save_shot(&mut h, "02_marked.png");

    h.key_press(Key::U); // undo the accept
    h.run_steps(2);
    assert_eq!(h.state().ws().marks_of(&h.state().ws().crops[12]).mark, None);

    h.key_press(Key::S);
    h.run_steps(2);
    assert!(tmp.path().join("annotations.v002.json").exists());
    assert!(!tmp.path().join("images/train/002.jpg").exists(), "frame 2 moved to trash");
    let ws = h.state().ws();
    assert_eq!(ws.project.dataset.annotations().len(), 12 - 1 - 2);
    assert!(!ws.project.log.is_dirty());
    settle(&mut h);
    save_shot(&mut h, "03_saved.png");
}

#[test]
fn text_fields_do_not_trigger_shortcuts() {
    let tmp = tempfile::tempdir().unwrap();
    synthetic_dataset(tmp.path());
    let mut h = harness(tmp.path().to_path_buf(), egui::vec2(1100.0, 700.0));
    settle(&mut h);
    // Focus a text input (the columns field) then type D.
    let ctx = h.ctx.clone();
    let id = egui::Id::new("focus-probe");
    ctx.memory_mut(|m| m.request_focus(id));
    h.state_mut().focus_probe = Some(id);
    h.run_steps(2);
    h.key_press(Key::D);
    h.run_steps(2);
    let ws = h.state().ws();
    assert_eq!(ws.marks_of(&ws.crops[0]).mark, None);
}

/// Opt-in visual check on a real dataset copy:
/// `FASTLABEL_SHOT_DATASET=<dir> FASTLABEL_SHOT_DIR=<out> cargo test -p fastlabel real_dataset_shot -- --ignored`
#[test]
#[ignore]
fn real_dataset_shot() {
    let dataset =
        PathBuf::from(std::env::var("FASTLABEL_SHOT_DATASET").expect("FASTLABEL_SHOT_DATASET"));
    let mut h = harness(dataset, egui::vec2(1500.0, 950.0));
    settle(&mut h);
    save_shot(&mut h, "real_01.png");
    for _ in 0..7 {
        h.key_press(Key::J);
    }
    h.key_press(Key::D);
    h.key_press(Key::G);
    h.key_press(Key::F);
    h.run_steps(2);
    settle(&mut h);
    save_shot(&mut h, "real_02.png");
}

/// Opt-in: runs a real model on the selected crop's frame through the app.
/// `FASTLABEL_SHOT_DATASET=<copy> FASTLABEL_MODEL=<model.onnx> cargo test -p fastlabel model_on_frame -- --ignored`
#[test]
#[ignore]
fn model_on_frame() {
    let dataset =
        PathBuf::from(std::env::var("FASTLABEL_SHOT_DATASET").expect("FASTLABEL_SHOT_DATASET"));
    let model = PathBuf::from(std::env::var("FASTLABEL_MODEL").expect("FASTLABEL_MODEL"));
    let mut h = harness(dataset, egui::vec2(1200.0, 800.0));
    settle(&mut h);
    let idx = h.state().ws().selected_crop().unwrap().image_idx;
    h.state_mut().session_mut().model_path = Some(model);
    let ctx = h.ctx.clone();
    h.state_mut().run_model(&ctx, crate::inference::Scope::Frame(idx));
    for _ in 0..600 {
        h.step();
        if !h.state().inference.is_running() {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    h.run_steps(2);
    let preds = h.state().ws().project.predictions.as_ref().unwrap();
    assert!(!preds.of(idx).unwrap().is_empty(), "the model found bibs on the frame");
}

#[test]
fn frame_mode_draw_edit_navigate_popup_and_peek() {
    use crate::app::Mode;
    use crate::frame::Sel;
    let tmp = tempfile::tempdir().unwrap();
    synthetic_dataset(tmp.path());
    let mut h = harness(tmp.path().to_path_buf(), egui::vec2(1100.0, 800.0));
    settle(&mut h);

    // Space held in the mosaic shows the peek overlay (read-only).
    h.key_down(Key::Space);
    h.run_steps(2);
    settle(&mut h);
    save_shot(&mut h, "10_peek.png");
    h.key_up(Key::Space);
    h.run_steps(2);

    // Tab → frame mode on the selected crop's frame, box selected.
    h.key_press(Key::Tab);
    h.run_steps(2);
    assert_eq!(h.state().mode, Mode::Frame);
    assert_eq!(h.state().ws().frame.sel, Some(Sel::Gt(1)));
    settle(&mut h);

    // Draw a new box by dragging on an empty area.
    h.drag_at(egui::pos2(700.0, 200.0));
    h.run_steps(2);
    h.hover_at(egui::pos2(760.0, 250.0));
    h.run_steps(2);
    h.drop_at(egui::pos2(800.0, 300.0));
    h.run_steps(3);
    let ws = h.state().ws();
    let added: Vec<_> = ws.frame_boxes(0).into_iter().filter(|b| b.added).collect();
    assert_eq!(added.len(), 1, "a box was drawn");
    assert_eq!(ws.frame.sel, Some(Sel::Gt(added[0].id)));
    save_shot(&mut h, "11_frame_drawn.png");

    // D marks the selection; J cycles; l goes to the next frame.
    h.key_press(Key::D);
    h.key_press(Key::L);
    h.run_steps(2);
    let ws = h.state().ws();
    assert_eq!(ws.frame.image_idx, 1);
    assert!(ws.project.log.state().ann_mark(added[0].id).is_some());

    // Back to the mosaic, Enter opens the popup, Esc closes it.
    h.key_press(Key::Tab);
    h.run_steps(2);
    assert_eq!(h.state().mode, Mode::Mosaic);
    h.key_press(Key::Enter);
    h.run_steps(2);
    assert!(h.state().popup);
    settle(&mut h);
    save_shot(&mut h, "12_popup.png");
    h.key_press(Key::Escape);
    h.run_steps(2);
    assert!(!h.state().popup);
}
