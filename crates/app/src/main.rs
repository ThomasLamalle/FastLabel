//! FastLabel: fast annotation triage (boxes, text).

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
mod persons;
mod report;
mod session;
mod tasks;
mod text_view;
mod thumbs;
mod todo;
#[cfg(test)]
mod ui_tests;
mod workspace;

use std::path::PathBuf;

use anyhow::anyhow;
use clap::Parser;
use eframe::egui;

use crate::session::Session;

#[derive(Parser)]
#[command(version, about = "Fast annotation triage")]
struct Cli {
    /// Dataset folder (COCO annotations, YOLO layout or plain images).
    /// Defaults to the last opened one.
    dataset: Option<PathBuf>,
    /// Write an HTML report on the dataset to this file and exit (no window).
    #[arg(long, value_name = "OUT.html")]
    report: Option<PathBuf>,
    /// Export the saved dataset and exit: a `.zip` path, else a new folder.
    #[arg(long, value_name = "OUT")]
    export: Option<PathBuf>,
    /// Export format: yolo, coco (per split), coco1 (single file), ndjson or
    /// crops (the first class's boxes + labels.csv, a text project for OCR).
    #[arg(long, default_value = "yolo", requires = "export")]
    format: String,
    /// Downscale exported images to this long side (labels follow).
    #[arg(long, requires = "export")]
    imgsz: Option<u32>,
    /// Add yolo26x person boxes (cached per frame), as the Export dialog does.
    #[arg(long, requires = "export")]
    persons: bool,
    /// Leave out boxes less tall than this (source pixels), in exports and
    /// reports. Default: 10, 26 for crops.
    #[arg(long)]
    min_height: Option<f64>,
}

/// `--export`: same as the Export dialog (all splits), without the GUI.
fn write_export(
    dataset: &std::path::Path,
    out: &std::path::Path,
    format: &str,
    imgsz: Option<u32>,
    min_box_height: f64,
    persons: bool,
) -> anyhow::Result<()> {
    use fastlabel_core::export::{self, ExportFormat, ExportOptions, Target};
    let format = ExportFormat::ALL
        .into_iter()
        .find(|f| f.slug() == format)
        .ok_or_else(|| anyhow!("unknown format {format} (yolo, coco, coco1, ndjson, crops)"))?;
    let kind = fastlabel_core::config::ProjectConfig::load(dataset)?.kind;
    if kind != fastlabel_core::config::Kind::Boxes {
        anyhow::bail!("only box projects export (a text project's data is its labels.csv)");
    }
    let project = fastlabel_core::Project::open(dataset)?;
    if project.has_unsaved_changes() {
        anyhow::bail!("the dataset has unsaved edits: save them in the app first");
    }
    let name = dataset.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let resize = imgsz
        .map(|max_side| export::Resize { max_side, resize: fastlabel_media::decode::resize_batch });
    let splits = fastlabel_core::Split::ALL.to_vec();
    let crops = (format == ExportFormat::Crops).then(|| export::CropOptions {
        category: project.dataset.categories().first().map_or(1, |c| c.id),
        margin: session::CROP_MARGIN,
        crop: fastlabel_media::decode::crop_batch,
    });
    let opts = ExportOptions { format, splits, name, resize, min_box_height, crops };
    let target = if out.extension().is_some_and(|e| e == "zip") {
        Target::Zip(out.to_path_buf())
    } else {
        Target::Folder(out.to_path_buf())
    };
    let mut coco = std::borrow::Cow::Borrowed(&project.dataset.coco);
    if persons {
        let cache = project.state_dir().join("cache").join("persons");
        let models = persons::models_dir();
        let progress = fastlabel_infer::Progress::default();
        let with = persons::add(&project.dir, &coco, &opts.splits, &cache, &models, &progress);
        coco = std::borrow::Cow::Owned(with.map_err(anyhow::Error::msg)?.unwrap_or_default());
    }
    let r = export::export(&project.dir, &coco, &opts, &target, |_, _| true)?;
    println!(
        "{}: {} images (train {}, val {}, test {}), {} without split, {} missing, {} moved to test by test_sources.txt",
        out.display(),
        r.images(),
        r.per_split[0],
        r.per_split[1],
        r.per_split[2],
        r.no_split,
        r.missing.len(),
        r.forced_test
    );
    Ok(())
}

/// `--report`: the same report as the Report button, without the GUI.
fn write_report(
    dataset: &std::path::Path,
    out: &std::path::Path,
    min_box_height: f64,
) -> anyhow::Result<()> {
    if fastlabel_core::config::ProjectConfig::load(dataset)?.kind
        != fastlabel_core::config::Kind::Boxes
    {
        anyhow::bail!("reports are for box projects");
    }
    let ws = workspace::Workspace::open(dataset, crops::ViewOptions::default(), 5)?;
    let state = ws.project.state_dir();
    let layout = embeddings::SavedLayout::load(&state.join(embeddings::LAYOUT_FILE));
    let emb = embeddings::Embeddings::load(&state.join(embeddings::EMBEDDINGS_FILE));
    let input = ws.report_input("All frames".into(), None, layout.as_ref(), min_box_height);
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
        return write_report(&dir, out, cli.min_height.unwrap_or(10.0));
    }
    if let Some(out) = &cli.export {
        let session = Session::default_path().map(|p| Session::load(&p)).unwrap_or_default();
        let dir = dataset.or(session.last_dataset).ok_or_else(|| anyhow!("no dataset given"))?;
        let crops = cli.format == fastlabel_core::export::ExportFormat::Crops.slug();
        let min_height =
            cli.min_height.unwrap_or(if crops { session::CROP_MIN_HEIGHT } else { 10.0 });
        return write_export(&dir, out, &cli.format, cli.imgsz, min_height, cli.persons);
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
