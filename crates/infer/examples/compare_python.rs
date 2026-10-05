//! Compares Rust inference with an existing `predictions.json` (read-only):
//! `cargo run --release -p fastlabel-infer --example compare_python -- <dataset> <model.onnx> [n] [predictions.json]`

use std::path::PathBuf;
use std::time::Instant;

use fastlabel_core::Project;
use fastlabel_infer::Detector;
use fastlabel_media::decode::decode_rgb;

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().expect("dataset dir"));
    let model = PathBuf::from(args.next().expect("model"));
    let n: usize = args.next().map_or(50, |s| s.parse().unwrap());
    let project = Project::open(&dir).unwrap();
    let preds = match args.next() {
        Some(p) => fastlabel_core::Predictions::from_file(
            &fastlabel_core::predictions::PredictionFile::load(std::path::Path::new(&p)).unwrap(),
            &project.dataset,
        ),
        None => project.predictions.clone().expect("predictions.json"),
    };
    let preds = &preds;
    let conf = preds.confidence as f32;

    let t = Instant::now();
    let threads = std::env::var("FASTLABEL_THREADS").ok().and_then(|t| t.parse().ok()).unwrap_or(0);
    let mut det = Detector::load(&model, threads).unwrap();
    println!(
        "loaded {:?} {}x{} names {:?} in {:?}",
        det.arch,
        det.input_w,
        det.input_h,
        det.class_names(),
        t.elapsed()
    );

    let (mut matched, mut only_rust, mut only_py) = (0, 0, 0);
    let (mut iou_sum, mut max_score_diff, mut infer_time) = (0.0, 0f64, std::time::Duration::ZERO);
    let step = (project.dataset.images().len() / n).max(1);
    for idx in (0..project.dataset.images().len()).step_by(step).take(n) {
        let img = &project.dataset.images()[idx];
        let rgb = decode_rgb(&project.dataset.image_path(img)).unwrap();
        let t = Instant::now();
        let ours = det.detect(&rgb, conf).unwrap();
        infer_time += t.elapsed();
        let theirs = preds.of(idx).unwrap_or(&[]);
        let mut used = vec![false; theirs.len()];
        for d in &ours {
            let best = theirs
                .iter()
                .enumerate()
                .filter(|(i, _)| !used[*i])
                .map(|(i, p)| (i, p.bbox.iou(&d.bbox), p.score))
                .max_by(|a, b| a.1.total_cmp(&b.1));
            match best {
                Some((i, iou, score)) if iou > 0.5 => {
                    used[i] = true;
                    matched += 1;
                    iou_sum += iou;
                    max_score_diff = max_score_diff.max((score - f64::from(d.score)).abs());
                }
                _ => only_rust += 1,
            }
        }
        only_py += used.iter().filter(|u| !**u).count();
    }
    println!(
        "{n} images: {matched} matched (mean IoU {:.4}, max score diff {:.4}), {only_rust} only in Rust, {only_py} only in Python; {:?}/image",
        iou_sum / matched.max(1) as f64,
        max_score_diff,
        infer_time / n as u32
    );
}
