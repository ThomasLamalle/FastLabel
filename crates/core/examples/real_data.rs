//! Read-only checks against a real dataset folder:
//! `cargo run --release -p fastlabel-core --example real_data -- <dataset dir>`

use std::path::PathBuf;
use std::time::Instant;

use fastlabel_core::coco::CocoFile;
use fastlabel_core::matching::{Thresholds, evaluate};
use fastlabel_core::{Project, versions};

fn main() {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: real_data <dataset dir>"),
    );

    // Lossless round-trip of every version file (JSON value equality).
    for v in versions::discover(&dir).unwrap() {
        let original: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&v.path).unwrap()).unwrap();
        let coco = CocoFile::load(&v.path).unwrap();
        let back = serde_json::to_value(&coco).unwrap();
        let same = numeric(&back) == numeric(&original);
        println!(
            "round-trip {:<28} {}",
            v.file_name(),
            if same { "ok" } else { "DIFFERENT" }
        );
    }

    let t = Instant::now();
    let project = Project::open(&dir).unwrap();
    println!(
        "open {}: {} images, {} annotations in {:?}",
        project.base_name(),
        project.dataset.images().len(),
        project.dataset.annotations().len(),
        t.elapsed()
    );
    for w in &project.warnings {
        println!("  warning: {w}");
    }
    if let Some(preds) = &project.predictions {
        let t = Instant::now();
        let ev = evaluate(&project.dataset, preds, Thresholds::default());
        println!(
            "evaluate {} predictions in {:?}: {:?}",
            preds.len(),
            t.elapsed(),
            ev.counts
        );
    }
}

/// Compares numbers by value (`0` and `0.0` are the same coordinate).
fn numeric(v: &serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match v {
        Value::Number(n) => n.as_f64().map_or(Value::Null, Value::from),
        Value::Array(a) => Value::Array(a.iter().map(numeric).collect()),
        Value::Object(o) => Value::Object(o.iter().map(|(k, v)| (k.clone(), numeric(v))).collect()),
        other => other.clone(),
    }
}
