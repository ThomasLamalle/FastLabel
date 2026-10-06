//! FastLabel: fast bounding-box review.

mod app;
mod clean;
mod cluster;
mod cluster_view;
mod crops;
mod editor;
mod embeddings;
mod frame;
mod inference;
mod metrics;
mod mosaic;
mod report;
mod session;
mod tasks;
mod thumbs;
#[cfg(test)]
mod ui_tests;
mod workspace;

use std::path::PathBuf;

use anyhow::anyhow;
use clap::Parser;
use eframe::egui;

use crate::session::Session;

#[derive(Parser)]
#[command(version, about = "Fast bounding-box annotation triage")]
struct Cli {
    /// Dataset folder (COCO annotations, YOLO layout or plain images).
    /// Defaults to the last opened one.
    dataset: Option<PathBuf>,
    /// Write an HTML report on the dataset to this file and exit (no window).
    #[arg(long, value_name = "OUT.html")]
    report: Option<PathBuf>,
}

/// `--report`: the same report as the Report button, without the GUI.
fn write_report(dataset: &std::path::Path, out: &std::path::Path) -> anyhow::Result<()> {
    let ws = workspace::Workspace::open(dataset, crops::ViewOptions::default(), 5)?;
    let state = ws.project.state_dir();
    let layout = embeddings::SavedLayout::load(&state.join(embeddings::LAYOUT_FILE));
    let emb = embeddings::Embeddings::load(&state.join(embeddings::EMBEDDINGS_FILE));
    let input = ws.report_input("All frames".into(), None, layout.as_ref());
    let neighbours = if emb.len() > 0 {
        report::cross_split_neighbours(&input.frames, &|k| emb.get(k).map(<[f32]>::to_vec))
    } else {
        Vec::new()
    };
    if let Some(dir) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(out, report::render(&input, &neighbours))?;
    println!("{}", out.display());
    Ok(())
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .init();
    let cli = Cli::parse();
    let dataset = cli.dataset.map(|p| p.canonicalize().unwrap_or(p));
    if let Some(out) = &cli.report {
        let session = Session::default_path().map(|p| Session::load(&p)).unwrap_or_default();
        let dir = dataset.or(session.last_dataset).ok_or_else(|| anyhow!("no dataset given"))?;
        return write_report(&dir, out);
    }
    let session_path = Session::default_path();
    let session = session_path.as_deref().map(Session::load).unwrap_or_default();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("FastLabel")
            .with_app_id("fastlabel")
            .with_inner_size([1400.0, 900.0]),
        ..Default::default()
    };
    eframe::run_native(
        "FastLabel",
        options,
        Box::new(move |cc| {
            Ok(Box::new(app::FastLabelApp::new(cc, session, session_path, dataset)))
        }),
    )
    .map_err(|e| anyhow!("{e}"))
}
