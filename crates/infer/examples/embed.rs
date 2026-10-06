//! Embeds a few frames and prints timing and nearest neighbours:
//! `cargo run --release -p fastlabel-infer --example embed -- model.onnx img1.jpg img2.jpg ...`

use std::path::PathBuf;
use std::time::Instant;

use fastlabel_infer::{BatchItem, Embedder, Progress, run_embed};

fn main() {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    let (model, images) = args.split_first().expect("usage: embed model.onnx images...");
    let t = Instant::now();
    let mut emb = Embedder::load(model, 0).expect("load model");
    println!("loaded in {:.2?} (input {}x{})", t.elapsed(), emb.input_w, emb.input_h);
    let items =
        images.iter().enumerate().map(|(key, p)| BatchItem { key, path: p.clone() }).collect();
    let t = Instant::now();
    let r = run_embed(&mut emb, items, &Progress::default());
    if let Some((_, e)) = r.failed.first() {
        println!("first failure: {e}");
    }
    let n = r.vectors.len();
    println!(
        "{n} vectors in {:.2?} ({:.1} ms/frame), {} failed",
        t.elapsed(),
        t.elapsed().as_secs_f64() * 1e3 / n.max(1) as f64,
        r.failed.len()
    );
    if let Ok(out) = std::env::var("EMBED_DUMP") {
        let lines: Vec<String> = r
            .vectors
            .iter()
            .map(|(_, v)| v.iter().map(f32::to_string).collect::<Vec<_>>().join(" "))
            .collect();
        std::fs::write(out, lines.join("\n")).expect("dump");
    }
    for (k, v) in &r.vectors {
        let best = r
            .vectors
            .iter()
            .filter(|(o, _)| o != k)
            .map(|(o, w)| (*o, v.iter().zip(w).map(|(a, b)| a * b).sum::<f32>()))
            .max_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((o, cos)) = best {
            println!("{} -> {} (cos {cos:.3})", images[*k].display(), images[o].display());
        }
    }
}
