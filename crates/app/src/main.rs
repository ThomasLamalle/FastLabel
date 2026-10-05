//! FastLabel: fast bounding-box review.

mod app;
mod crops;
mod inference;
mod mosaic;
mod session;
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
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .init();
    let cli = Cli::parse();
    let dataset = cli.dataset.map(|p| p.canonicalize().unwrap_or(p));
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
