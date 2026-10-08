//! Headless UI tests: the real app driven through egui_kittest (offscreen
//! wgpu rendering, synthetic key presses).

use std::path::{Path, PathBuf};
use std::time::Duration;

use eframe::egui::{self, Key, Modifiers};
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
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
    h.get_by_label("To do (T)").click();
    h.run_steps(2);
    h.get_by_label("Show background").click();
    h.run_steps(2);
    settle(&mut h);
    save_shot(&mut h, "real_03.png");
}

/// Opt-in: runs a real model on the selected crop's frame through the app.
/// `FASTLABEL_SHOT_DATASET=<copy> FASTLABEL_MODEL=<model.onnx|best.pt> cargo test -p fastlabel model_on_frame -- --ignored`
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
    for _ in 0..6000 {
        h.step();
        if !h.state().inference.is_running() && h.state().convert_task.is_none() {
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

#[test]
fn uniformity_sort_computes_in_background() {
    use crate::crops::SortKey;
    let tmp = tempfile::tempdir().unwrap();
    synthetic_dataset(tmp.path());
    let mut h = harness(tmp.path().to_path_buf(), egui::vec2(1100.0, 700.0));
    settle(&mut h);
    let ctx = h.ctx.clone();
    assert!(h.state_mut().start_uniformity(&ctx));
    for _ in 0..500 {
        h.step();
        if h.state().ws().opts.sort == SortKey::Uniformity {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let ws = h.state().ws();
    assert_eq!(ws.opts.sort, SortKey::Uniformity);
    assert!(ws.crops.iter().all(|c| c.uniformity.is_some()));
    // The white boxes (first GT of every frame) are the most uniform.
    assert!(ws.crops[..6].iter().all(|c| c.ann_id().is_some_and(|id| id % 2 == 1)));
    assert!(tmp.path().join(".fastlabel/metrics.json").exists());
    assert!(ws.missing_uniformity().is_empty());
}

/// Opt-in timing of the uniformity computation on a real dataset copy.
#[test]
#[ignore]
fn real_uniformity_timing() {
    let dataset =
        PathBuf::from(std::env::var("FASTLABEL_SHOT_DATASET").expect("FASTLABEL_SHOT_DATASET"));
    let mut h = harness(dataset, egui::vec2(1200.0, 800.0));
    settle(&mut h);
    let ctx = h.ctx.clone();
    let t = std::time::Instant::now();
    assert!(h.state_mut().start_uniformity(&ctx));
    while h.state().ws().opts.sort != crate::crops::SortKey::Uniformity {
        h.step();
        std::thread::sleep(Duration::from_millis(20));
        assert!(t.elapsed() < Duration::from_secs(300));
    }
    println!("uniformity for {} crops in {:?}", h.state().ws().crops.len(), t.elapsed());
    settle(&mut h);
    save_shot(&mut h, "real_uniformity.png");
}

fn rss_mb() -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    status
        .lines()
        .find(|l| l.starts_with("VmRSS:"))
        .and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())
        .map_or(0.0, |kb| kb / 1024.0)
}

/// Opt-in scale test: 100k images (symlinks to 5 real 1080p JPEGs), 200k
/// boxes, 100k predictions. `cargo test --release -p fastlabel stress_100k -- --ignored --nocapture`
#[test]
#[ignore]
fn stress_100k() {
    const N: i64 = 100_000;
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let src: Vec<PathBuf> = (0..5)
        .map(|k| {
            let img = image::RgbImage::from_fn(1920, 1080, |x, y| {
                image::Rgb([(x / 8) as u8, (y / 5) as u8, (k * 50) as u8])
            });
            let p = dir.join(format!("src{k}.jpg"));
            std::fs::write(&p, encode_jpeg(&img, 85).unwrap()).unwrap();
            p
        })
        .collect();
    std::fs::create_dir_all(dir.join("images/train")).unwrap();
    let mut coco = CocoFile {
        categories: vec![CocoCategory {
            id: 1,
            name: "bib".into(),
            supercategory: String::new(),
            extra: Map::new(),
        }],
        ..CocoFile::default()
    };
    let mut preds = Predictions::empty("stress", 0.1);
    for id in 1..=N {
        let file_name = format!("images/train/{id:06}.jpg");
        std::os::unix::fs::symlink(&src[(id % 5) as usize], dir.join(&file_name)).unwrap();
        coco.images.push(CocoImage {
            id,
            file_name,
            width: 1920,
            height: 1080,
            tags: vec![],
            extra: Map::new(),
        });
        let n = coco.annotations.len() as i64;
        let a = BBox::new(100.0 + (id % 900) as f64, 300.0, 90.0, 50.0);
        coco.annotations.push(CocoAnnotation::new(n + 1, id, 1, a));
        coco.annotations.push(CocoAnnotation::new(
            n + 2,
            id,
            1,
            BBox::new(1200.0, 500.0, 70.0, 40.0),
        ));
        preds.set_image(
            id as usize - 1,
            vec![PredRef {
                image_id: id,
                category_id: 1,
                bbox: BBox::new(a.x + 4.0, a.y, a.w, a.h),
                score: 0.8,
            }],
        );
    }
    coco.save(&dir.join("annotations.v001.json")).unwrap();
    let ds = fastlabel_core::Dataset::new(dir.to_path_buf(), coco, Path::new("t")).unwrap().0;
    preds.to_file(&ds).save(&dir.join("predictions.json")).unwrap();
    drop(ds);
    let base = rss_mb();

    let t = std::time::Instant::now();
    let mut h = harness(dir.to_path_buf(), egui::vec2(1600.0, 1000.0));
    h.step();
    println!(
        "open + first frame: {:?}, {} crops, RSS +{:.0} MB",
        t.elapsed(),
        h.state().ws().crops.len(),
        rss_mb() - base
    );
    settle(&mut h);

    // Scroll through the dataset by pages and jumps; time each frame.
    let mut worst = Duration::ZERO;
    let t = std::time::Instant::now();
    for i in 0..200 {
        h.key_press(if i % 50 == 49 { Key::End } else { Key::PageDown });
        let f = std::time::Instant::now();
        h.step();
        worst = worst.max(f.elapsed());
    }
    println!("200 page jumps: {:?} total, worst UI frame {:?}", t.elapsed(), worst);
    settle(&mut h);

    // Mark a lot, then undo everything.
    let t = std::time::Instant::now();
    for _ in 0..2000 {
        h.key_press(Key::D);
    }
    h.step();
    for _ in 0..2000 {
        h.key_press(Key::U);
    }
    h.step();
    println!("2000 marks + 2000 undos: {:?}", t.elapsed());
    assert!(!h.state().ws().project.log.is_dirty());

    let t = std::time::Instant::now();
    h.state_mut().ws_mut().rebuild();
    println!("re-evaluate + rebuild 200k crops: {:?}", t.elapsed());
    println!("RSS after scrolling: +{:.0} MB", rss_mb() - base);
    assert!(rss_mb() - base < 1500.0, "memory stays bounded");
}

#[test]
fn empty_dataset_survives_every_key() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path().join("images")).unwrap();
    let mut h = harness(tmp.path().to_path_buf(), egui::vec2(900.0, 600.0));
    h.run_steps(2);
    assert_eq!(h.state().ws().crops.len(), 0);
    let keys = [
        Key::J,
        Key::K,
        Key::H,
        Key::L,
        Key::D,
        Key::G,
        Key::F,
        Key::R,
        Key::U,
        Key::Enter,
        Key::End,
        Key::PageDown,
        Key::Num1,
        Key::Escape,
        Key::Tab,
        Key::H,
        Key::L,
        Key::D,
        Key::J,
        Key::F,
        Key::Tab,
    ];
    for k in keys {
        h.key_press(k);
        h.key_press_modifiers(Modifiers::SHIFT, k);
        h.run_steps(1);
    }
    h.key_down(Key::Space);
    h.run_steps(2);
    h.key_up(Key::Space);
    h.key_press(Key::S); // first save of an imported folder writes v001
    h.run_steps(2);
    assert!(tmp.path().join("annotations.v001.json").exists());
}

/// Regression: after clicking a crop (which gives it egui focus), Enter,
/// Tab and Space must still reach our shortcuts.
#[test]
fn shortcuts_work_after_clicking_a_crop() {
    use crate::app::Mode;
    let tmp = tempfile::tempdir().unwrap();
    synthetic_dataset(tmp.path());
    let mut h = harness(tmp.path().to_path_buf(), egui::vec2(1100.0, 700.0));
    settle(&mut h);
    let click = |h: &mut Harness<'static, FastLabelApp>| {
        h.hover_at(egui::pos2(400.0, 150.0));
        h.run_steps(1);
        h.event(egui::Event::PointerButton {
            pos: egui::pos2(400.0, 150.0),
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: Modifiers::NONE,
        });
        h.run_steps(1);
        h.event(egui::Event::PointerButton {
            pos: egui::pos2(400.0, 150.0),
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: Modifiers::NONE,
        });
        h.run_steps(2);
    };
    click(&mut h);
    assert_eq!(h.state().ws().selected, 1, "click selected the second crop");
    h.key_press(Key::Enter);
    h.run_steps(2);
    assert!(h.state().popup, "Enter opens the popup after a click");
    h.key_press(Key::Escape);
    h.run_steps(2);
    assert!(!h.state().popup);

    click(&mut h);
    h.key_press(Key::Tab);
    h.run_steps(2);
    assert_eq!(h.state().mode, Mode::Frame, "Tab switches mode after a click");
    h.key_press(Key::M);
    h.run_steps(2);
    assert_eq!(h.state().mode, Mode::Mosaic, "M switches mode too");

    click(&mut h);
    h.key_press(Key::D);
    h.run_steps(2);
    let ws = h.state().ws();
    assert!(ws.marks_of(&ws.crops[1]).mark.is_some(), "D works after a click");
}

/// Regression: clicking a toolbar button must not steal the shortcuts.
#[test]
fn shortcuts_work_after_clicking_a_toolbar_button() {
    let tmp = tempfile::tempdir().unwrap();
    synthetic_dataset(tmp.path());
    let mut h = harness(tmp.path().to_path_buf(), egui::vec2(1100.0, 700.0));
    settle(&mut h);
    h.get_by_label("Hide reviewed (0)").click();
    h.run_steps(2);
    h.get_by_label("Hide reviewed (0)").click();
    h.run_steps(2);
    h.key_press(Key::D);
    h.run_steps(2);
    let ws = h.state().ws();
    assert!(ws.marks_of(&ws.crops[0]).mark.is_some(), "D works after clicking a toolbar button");
    h.key_press(Key::Enter);
    h.run_steps(2);
    assert!(h.state().popup, "Enter opens the popup after clicking a toolbar button");
}

/// Regression: a focused button (e.g. after a real mouse click) must not
/// swallow Enter / Space / Tab, nor be "pressed" by them.
#[test]
fn focused_button_does_not_swallow_shortcuts() {
    use crate::app::Mode;
    let tmp = tempfile::tempdir().unwrap();
    synthetic_dataset(tmp.path());
    let mut h = harness(tmp.path().to_path_buf(), egui::vec2(1100.0, 700.0));
    settle(&mut h);
    let focus = |h: &mut Harness<'static, FastLabelApp>| {
        h.get_by_label("Ascending").focus();
        h.run_steps(1);
        assert!(h.get_by_label("Ascending").is_focused(), "button focused");
    };
    focus(&mut h);
    h.key_press(Key::Enter);
    h.run_steps(2);
    assert!(h.state().popup, "Enter opens the popup");
    assert!(!h.state().ws().opts.descending, "the focused button was not pressed");
    h.key_press(Key::Escape);
    h.run_steps(2);
    focus(&mut h);
    h.key_press(Key::Tab);
    h.run_steps(2);
    assert_eq!(h.state().mode, Mode::Frame);
    h.key_press(Key::M);
    h.run_steps(2);
    assert_eq!(h.state().mode, Mode::Mosaic);
    h.key_press(Key::E);
    h.run_steps(2);
    assert!(h.state().popup, "E opens the popup");
}

#[test]
fn overlapping_gt_cleaning_workflow() {
    use crate::app::Mode;
    let tmp = tempfile::tempdir().unwrap();
    synthetic_dataset(tmp.path());
    // Add double annotations: a near-duplicate on frames 1 and 3, and a
    // chain of 11 overlapping boxes on frame 5.
    let path = tmp.path().join("annotations.v001.json");
    let mut coco = CocoFile::load(&path).unwrap();
    let mut next = coco.next_annotation_id();
    let mut add = |coco: &mut CocoFile, image: i64, b: BBox| {
        coco.annotations.push(CocoAnnotation::new(next, image, 1, b));
        next += 1;
    };
    add(&mut coco, 1, BBox::new(45.0, 62.0, 80.0, 50.0));
    add(&mut coco, 3, BBox::new(38.0, 58.0, 84.0, 54.0));
    for i in 0..11 {
        add(&mut coco, 5, BBox::new(130.0 + 5.0 * i as f64, 10.0, 30.0, 20.0));
    }
    coco.save(&path).unwrap();

    let mut h = harness(tmp.path().to_path_buf(), egui::vec2(1200.0, 800.0));
    settle(&mut h);
    h.get_by_label("Cleaning").click();
    h.run_steps(2);
    h.get_by_label("Overlapping GT boxes").click();
    h.run_steps(2);
    assert_eq!(h.state().mode, Mode::Clean);
    assert_eq!(h.state().clean.as_ref().unwrap().groups.len(), 3);
    settle(&mut h);
    save_shot(&mut h, "20_clean_group.png");

    // Group 1 (frame 1): keep box 2, the other is queued for deletion.
    let g0 = h.state().clean.as_ref().unwrap().groups[0].clone();
    h.key_press(Key::Num2);
    h.run_steps(2);
    let ws = h.state().ws();
    let marks: Vec<_> = ws
        .frame_boxes(g0.image_idx)
        .iter()
        .filter(|b| g0.boxes.contains(&b.id))
        .map(|b| (b.id, b.mark))
        .collect();
    assert_eq!(marks.iter().filter(|(_, m)| *m == Some(crate::workspace::Mark::Delete)).count(), 1);
    assert_eq!(
        marks.iter().find(|(id, _)| *id == g0.boxes[1]).unwrap().1,
        Some(crate::workspace::Mark::Good)
    );
    assert_eq!(h.state().clean.as_ref().unwrap().index, 1, "moved to the next group");

    // Progress is shown above the frame.
    let _ = h.get_by_label("Overlapping GT boxes: group 2 / 3");

    // Group 2: D deletes all its boxes; Left goes back, A keeps both instead.
    let g1 = h.state().clean.as_ref().unwrap().groups[1].clone();
    h.key_press(Key::D);
    h.run_steps(2);
    let ws = h.state().ws();
    assert!(
        ws.frame_boxes(g1.image_idx)
            .iter()
            .filter(|b| g1.boxes.contains(&b.id))
            .all(|b| b.mark == Some(crate::workspace::Mark::Delete))
    );
    h.key_press(Key::ArrowLeft);
    h.run_steps(2);
    assert_eq!(h.state().clean.as_ref().unwrap().index, 1);

    // Group 2: keep both. Group 3 (11 boxes): type 1, 1, Enter.
    h.key_press(Key::A);
    h.run_steps(2);
    let g2 = h.state().clean.as_ref().unwrap().groups[2].clone();
    assert_eq!(g2.boxes.len(), 11);
    settle(&mut h);
    save_shot(&mut h, "21_clean_eleven.png");
    h.key_press(Key::Num1);
    h.key_press(Key::Num1);
    h.run_steps(1);
    assert_eq!(h.state().clean.as_ref().unwrap().digits, "11");
    h.key_press(Key::Enter);
    h.run_steps(2);
    let ws = h.state().ws();
    let kept: Vec<_> = ws
        .frame_boxes(g2.image_idx)
        .iter()
        .filter(|b| g2.boxes.contains(&b.id) && b.mark == Some(crate::workspace::Mark::Good))
        .map(|b| b.id)
        .collect();
    assert_eq!(kept, vec![g2.boxes[10]]);
    assert_eq!(h.state().clean.as_ref().unwrap().index, 3, "done screen");

    // Save: decisions applied, nothing left to review; Esc back to the mosaic.
    h.key_press(Key::S);
    h.run_steps(2);
    assert!(tmp.path().join("annotations.v002.json").exists());
    assert!(h.state().clean.as_ref().unwrap().groups.is_empty());
    assert_eq!(h.state().ws().project.dataset.annotations().len(), 12 + 13 - 1 - 10);

    h.key_press(Key::Escape);
    h.run_steps(2);
    assert_eq!(h.state().mode, Mode::Mosaic);
}

/// Opt-in: the overlapping-GT cleaning view on a real dataset copy.
#[test]
#[ignore]
fn real_cleaning_shot() {
    let dataset =
        PathBuf::from(std::env::var("FASTLABEL_SHOT_DATASET").expect("FASTLABEL_SHOT_DATASET"));
    let mut h = harness(dataset, egui::vec2(1500.0, 950.0));
    settle(&mut h);
    h.state_mut().start_overlap_cleaning();
    h.run_steps(2);
    println!("groups: {}", h.state().clean.as_ref().map_or(0, |c| c.groups.len()));
    settle(&mut h);
    save_shot(&mut h, "real_clean_1.png");
    h.key_press(Key::ArrowRight);
    h.run_steps(2);
    settle(&mut h);
    save_shot(&mut h, "real_clean_2.png");
}

#[test]
fn cluster_mode_lasso_move_split_and_show_in_mosaic() {
    use crate::app::Mode;
    use crate::embeddings::{self, SavedLayout};
    use fastlabel_core::Split;
    let tmp = tempfile::tempdir().unwrap();
    synthetic_dataset(tmp.path());
    // Precomputed layout: frames 1-3 on the left, 4-6 on the right.
    let layout = SavedLayout {
        model: "test".into(),
        min_cluster_size: 2,
        keys: (1..=6).map(|id| embeddings::key(id, &format!("images/train/{id:03}.jpg"))).collect(),
        points: (0..6).map(|i| [if i < 3 { 0.0 } else { 10.0 }, (i % 3) as f32]).collect(),
        labels: (0..6).map(|i| i / 3).collect(),
    };
    layout.save(&tmp.path().join(".fastlabel").join(embeddings::LAYOUT_FILE)).unwrap();
    let mut h = harness(tmp.path().to_path_buf(), egui::vec2(1100.0, 700.0));
    settle(&mut h);

    h.key_press(Key::C);
    h.run_steps(3);
    assert_eq!(h.state().mode, Mode::Cluster);
    settle(&mut h);
    save_shot(&mut h, "20_cluster.png");

    // Lasso around the left half.
    h.drag_at(egui::pos2(20.0, 150.0));
    h.run_steps(1);
    for p in [(500.0, 150.0), (500.0, 650.0), (20.0, 650.0), (20.0, 160.0)] {
        h.hover_at(egui::pos2(p.0, p.1));
        h.run_steps(1);
    }
    h.drop_at(egui::pos2(20.0, 160.0));
    h.run_steps(3);
    assert_eq!(h.state().cluster.selection.iter().copied().collect::<Vec<_>>(), vec![1, 2, 3]);
    settle(&mut h);
    save_shot(&mut h, "21_cluster_selected.png");

    h.key_press(Key::Num2); // → val
    h.run_steps(2);
    let ws = h.state().ws();
    assert_eq!((0..6).map(|i| ws.split_of(i)).filter(|s| *s == Some(Split::Val)).count(), 3);
    h.key_press(Key::F); // delete them all
    h.run_steps(2);
    assert!((0..3).all(|i| h.state().ws().is_frame_deleted_idx(i)));
    h.key_press(Key::U); // and undo, in one step
    h.run_steps(2);
    assert!((0..3).all(|i| !h.state().ws().is_frame_deleted_idx(i)));

    h.key_press(Key::Enter); // show the selection in the mosaic
    h.run_steps(3);
    assert_eq!(h.state().mode, Mode::Mosaic);
    let ws = h.state().ws();
    assert!(ws.crops.iter().all(|c| c.image_id <= 3));
    assert_eq!(ws.crops.len(), 7, "3 frames x 2 GT + the FP of frame 1");

    h.key_press(Key::S);
    h.run_steps(2);
    // The split is saved as a field; files stay where they are.
    let saved = std::fs::read_to_string(tmp.path().join("annotations.v002.json")).unwrap();
    assert_eq!(saved.matches(r#""split": "val""#).count(), 3);
    assert!(tmp.path().join("images/train/001.jpg").exists());
    let ws = h.state().ws();
    assert_eq!(ws.project.dataset.split_of(0), Some(Split::Val));
    assert!(!ws.project.log.is_dirty());

    // The layout still places the moved frames.
    h.key_press(Key::C);
    h.run_steps(3);
    assert_eq!(h.state().cluster.unplaced, 0);
    assert_eq!(h.state().cluster.selection.len(), 3);
    // Auto-split: whole clusters (3 frames each) go to one split.
    use crate::cluster::AutoSplit;
    h.state_mut().session_mut().auto_split =
        AutoSplit { val: 0.5, test: 0.0, max_group_share: 1.0, seed: 3 };
    let ctx = h.ctx.clone();
    h.state_mut().cluster_event(&ctx, crate::cluster_view::ClusterEvent::AutoSplit);
    h.run_steps(2);
    let splits: Vec<_> = (0..6).map(|i| h.state().ws().split_of(i)).collect();
    assert_eq!(splits.iter().filter(|s| **s == Some(Split::Val)).count(), 3, "{splits:?}");
    assert_eq!(splits[0], splits[1]);
    assert_eq!(splits[1], splits[2]);
    assert_eq!(splits[3], splits[4]);
    h.key_press(Key::U);
    h.run_steps(2);
    assert!((3..6).all(|i| h.state().ws().split_of(i) == Some(Split::Train)), "one undo step");

    h.key_press(Key::Escape);
    h.run_steps(2);
    assert_eq!(h.state().mode, Mode::Mosaic);
}

/// Opt-in: embeddings + layout on a real dataset copy, with timings and shots.
/// `FASTLABEL_SHOT_DATASET=<copy> FASTLABEL_EMBED_MODEL=<dinov2.onnx> FASTLABEL_SHOT_DIR=<out> cargo test --release -p fastlabel real_clusters -- --ignored --nocapture`
#[test]
#[ignore]
fn real_clusters() {
    use crate::cluster_view::{ClusterEvent, ColorBy};
    let dataset =
        PathBuf::from(std::env::var("FASTLABEL_SHOT_DATASET").expect("FASTLABEL_SHOT_DATASET"));
    let model =
        PathBuf::from(std::env::var("FASTLABEL_EMBED_MODEL").expect("FASTLABEL_EMBED_MODEL"));
    let mut h = harness(dataset, egui::vec2(1500.0, 950.0));
    settle(&mut h);
    h.state_mut().session_mut().embed_model_path = Some(model);
    h.key_press(Key::C);
    h.run_steps(2);
    let ctx = h.ctx.clone();
    let start = std::time::Instant::now();
    h.state_mut().cluster_event(&ctx, ClusterEvent::ComputeEmbeddings);
    let wait = |h: &mut Harness<'static, FastLabelApp>| {
        while h.state().cluster_busy() {
            h.step();
            std::thread::sleep(Duration::from_millis(200));
        }
    };
    wait(&mut h);
    eprintln!("embeddings + layout: {:.1?}, RSS {:.0} MB", start.elapsed(), rss_mb());
    let t = std::time::Instant::now();
    h.state_mut().cluster_event(&ctx, ClusterEvent::Relayout);
    wait(&mut h);
    eprintln!("layout alone: {:.1?}", t.elapsed());
    h.run_steps(3);
    settle(&mut h);
    save_shot(&mut h, "real_clusters_split.png");
    h.state_mut().cluster.color_by = ColorBy::Cluster;
    h.run_steps(3);
    save_shot(&mut h, "real_clusters_clusters.png");
}

#[test]
fn todo_tab_lists_added_frames_and_confirms_background() {
    use crate::app::Mode;
    let tmp = tempfile::tempdir().unwrap();
    synthetic_dataset(tmp.path());
    let mut h = harness(tmp.path().to_path_buf(), egui::vec2(1100.0, 700.0));
    settle(&mut h);
    // Two new frames, no boxes.
    let incoming = tmp.path().join("incoming");
    std::fs::create_dir_all(&incoming).unwrap();
    for (i, name) in ["a.jpg", "b.jpg"].iter().enumerate() {
        let img = image::RgbImage::from_pixel(320, 240, image::Rgb([50 * i as u8, 90, 140]));
        std::fs::write(incoming.join(name), encode_jpeg(&img, 90).unwrap()).unwrap();
    }
    let r = fastlabel_core::add::copy_sources(
        tmp.path(),
        &fastlabel_core::add::collect(std::slice::from_ref(&incoming)),
        |_, _| true,
    );
    h.state_mut().ws_mut().register_added(&r.copied, None).unwrap();

    h.key_press(Key::T);
    h.run_steps(2);
    assert_eq!(h.state().mode, Mode::Todo);
    assert_eq!(h.state().todo.frames.len(), 2);
    settle(&mut h);
    save_shot(&mut h, "30_todo.png");

    let (first, second) = (h.state().todo.frames[0], h.state().todo.frames[1]);
    h.key_press(Key::G); // background from the grid, advance
    h.run_steps(2);
    assert!(h.state().ws().todo_tile(first).confirmed);
    assert_eq!(h.state().todo.selected, 1);
    h.key_press(Key::U); // undo it
    h.key_press(Key::Home);
    h.run_steps(2);
    assert!(!h.state().ws().todo_tile(first).confirmed);
    h.run_steps(2);

    // Enter: frame mode through the to-do frames only.
    h.key_press(Key::Enter);
    h.run_steps(2);
    assert_eq!(h.state().mode, Mode::Frame);
    assert_eq!(h.state().ws().frame.image_idx, first);
    h.key_press(Key::L);
    h.run_steps(2);
    assert_eq!(h.state().ws().frame.image_idx, second, "next to-do frame, not the next image");
    assert!(h.state().ws().todo_tile(first).confirmed, "skipped untouched: background");
    h.key_press(Key::H);
    h.run_steps(2);
    assert!(!h.state().ws().todo_tile(second).confirmed, "going back marks nothing");
    h.key_press(Key::G); // G on a confirmed frame takes the mark back
    h.run_steps(2);
    assert!(!h.state().ws().todo_tile(first).confirmed);
    assert_eq!(h.state().ws().frame.image_idx, first, "unmarking does not move");
    // Right click: next frame, skipping this one as background again.
    for pressed in [true, false] {
        h.event(egui::Event::PointerButton {
            pos: egui::pos2(500.0, 400.0),
            button: egui::PointerButton::Secondary,
            pressed,
            modifiers: Modifiers::NONE,
        });
        h.run_steps(1);
    }
    h.run_steps(1);
    assert_eq!(h.state().ws().frame.image_idx, second, "right click: next frame");
    assert!(h.state().ws().todo_tile(first).confirmed);
    h.key_press(Key::L); // last to-do frame: no move, no mark
    h.run_steps(2);
    assert!(!h.state().ws().todo_tile(second).confirmed);
    settle(&mut h);
    save_shot(&mut h, "31_todo_frame.png");
    h.key_press(Key::Tab); // back to the grid, on the frame reached
    h.run_steps(2);
    assert_eq!(h.state().mode, Mode::Todo);
    assert_eq!(h.state().todo.selected, 1);
    assert!(h.state().ws().frame_order.is_none());

    h.key_press(Key::S);
    h.run_steps(2);
    assert_eq!(h.state().todo.frames.len(), 1, "the background frame is done");
    let id = h.state().ws().project.dataset.images()[first].id;
    assert!(h.state().ws().project.review.empty_frames.contains(&id));
    h.key_press(Key::Escape);
    h.run_steps(2);
    assert_eq!(h.state().mode, Mode::Mosaic);
}
