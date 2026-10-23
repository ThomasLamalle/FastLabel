//! Top-level window: toolbar, status line, mosaic and keyboard shortcuts.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use eframe::egui::{self, Color32, Key, Modifiers, RichText, ViewportCommand};
use fastlabel_core::Split;
use fastlabel_media::Job;

use crate::cluster_view::{ClusterEvent, ClusterView, Setup};
use crate::crops::SortKey;
use crate::editor::{EditorAction, EditorView, FrameData};
use crate::embeddings::{self, Embeddings, SavedLayout};
use crate::frame::Sel;
use crate::inference::{Inference, Scope};
use crate::mosaic::{self, MosaicEvent, MosaicState};
use crate::session::Session;
use crate::tasks::Task;
use crate::thumbs::Thumbs;
use crate::workspace::{Mark, Workspace};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Mosaic,
    Frame,
    /// Cleaning workflow: overlapping GT boxes, one group at a time.
    Clean,
    /// Frames as a point cloud of image embeddings, to manage splits.
    Cluster,
    /// Frames without any box (e.g. just added), to annotate.
    Todo,
    /// The dataset's labeling rules, editable.
    Rules,
}

/// Finished embedding run: model id, (image key, vector), failures, cancelled.
struct EmbedOutcome {
    model: String,
    vectors: Vec<(u64, Vec<f32>)>,
    failed: Vec<String>,
    cancelled: bool,
    error: Option<String>,
}

/// Progress through the overlapping-GT cleaning workflow.
pub(crate) struct CleanState {
    pub groups: Vec<crate::clean::OverlapGroup>,
    pub index: usize,
    /// Typed digits when a group has more than 9 boxes (Enter confirms).
    pub digits: String,
}

/// Full frames are loaded at most this large.
const FRAME_MAX_SIDE: u32 = 4096;

struct Status {
    text: String,
    error: bool,
    at: Instant,
}

/// What a file dialog picks.
enum Pick {
    Open,
    ExportZip,
    ExportFolder,
    Add,
    Model,
    EmbedModel,
}

fn pick_onnx() -> Vec<PathBuf> {
    rfd::FileDialog::new().add_filter("ONNX model", &["onnx"]).pick_file().into_iter().collect()
}

fn pick_detector() -> Vec<PathBuf> {
    let dialog = rfd::FileDialog::new().add_filter("ONNX or ultralytics model", &["onnx", "pt"]);
    dialog.pick_file().into_iter().collect()
}

pub struct FastLabelApp {
    session: Session,
    session_path: Option<PathBuf>,
    ws: Option<Workspace>,
    thumbs: Option<Thumbs>,
    mosaic: MosaicState,
    pub(crate) mode: Mode,
    /// Editor popup over the mosaic (Enter / double click).
    pub(crate) popup: bool,
    pub(crate) clean: Option<CleanState>,
    editor: EditorView,
    peek: EditorView,
    /// Tags text being edited: (image index, text).
    tags_buf: Option<(usize, String)>,
    pub(crate) inference: Inference,
    /// Drop keyboard focus at the end of this frame (after Tab).
    drop_focus: bool,
    /// Color-uniformity computation; sorts by it when done.
    uniformity_task: Option<Task<std::collections::HashMap<u64, f32>>>,
    export_task: Option<Task<Result<(fastlabel_core::export::ExportReport, PathBuf), String>>>,
    /// Person detection run before an export to this target; `None` result
    /// when cancelled.
    persons_task: Option<(fastlabel_core::export::Target, Task<crate::persons::Outcome>)>,
    pub(crate) cluster: ClusterView,
    pub(crate) todo: crate::todo::TodoState,
    /// Frame mode was opened from the To do tab (Tab / Esc go back to it).
    frame_from_todo: bool,
    /// Loaded when cluster mode is first opened.
    embeddings: Option<Embeddings>,
    embed_task: Option<Task<EmbedOutcome>>,
    layout_task: Option<Task<Option<SavedLayout>>>,
    report_task: Option<Task<Result<PathBuf, String>>>,
    /// `.pt` → ONNX export, then the run it was started for.
    pub(crate) convert_task: Option<(Scope, Task<Result<PathBuf, String>>)>,
    /// Export dialog open.
    export_dialog: bool,
    /// The report's suggested input size, computed when the dialog opens.
    export_suggested_imgsz: u32,
    add_task: Option<Task<fastlabel_core::add::AddReport>>,
    /// Open file dialog and what its result is for.
    pick: Option<(Pick, Task<Vec<PathBuf>>)>,
    status: Option<Status>,
    /// Set once the exit save succeeded, so the close request goes through.
    exit_ok: bool,
    /// Test hook: shows a text field with this id (keyboard focus tests).
    #[cfg(test)]
    pub focus_probe: Option<egui::Id>,
}

impl FastLabelApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        session: Session,
        session_path: Option<PathBuf>,
        dataset: Option<PathBuf>,
    ) -> Self {
        cc.egui_ctx.set_theme(egui::Theme::Dark);
        let min_cluster_size = session.min_cluster_size;
        let mut app = Self {
            session,
            session_path,
            ws: None,
            thumbs: None,
            mosaic: MosaicState::default(),
            mode: Mode::Mosaic,
            popup: false,
            clean: None,
            editor: EditorView::default(),
            peek: EditorView::default(),
            tags_buf: None,
            inference: Inference::default(),
            drop_focus: false,
            uniformity_task: None,
            export_task: None,
            persons_task: None,
            cluster: ClusterView::default(),
            todo: Default::default(),
            frame_from_todo: false,
            embeddings: None,
            embed_task: None,
            layout_task: None,
            convert_task: None,
            report_task: None,
            export_dialog: false,
            export_suggested_imgsz: 1280,
            add_task: None,
            pick: None,
            status: None,
            exit_ok: false,
            #[cfg(test)]
            focus_probe: None,
        };
        app.cluster.min_cluster_size = min_cluster_size;
        let restore = dataset.is_none();
        if let Some(dir) = dataset.or_else(|| app.session.last_dataset.clone()) {
            app.open(&cc.egui_ctx, &dir, restore);
        }
        app
    }

    #[cfg(test)]
    pub fn ws(&self) -> &Workspace {
        self.ws.as_ref().expect("a dataset is open")
    }

    #[cfg(test)]
    pub fn ws_mut(&mut self) -> &mut Workspace {
        self.ws.as_mut().expect("a dataset is open")
    }

    #[cfg(test)]
    pub fn session_mut(&mut self) -> &mut Session {
        &mut self.session
    }

    #[cfg(test)]
    pub fn cluster_busy(&self) -> bool {
        self.embed_task.is_some() || self.layout_task.is_some()
    }

    pub fn thumbs_busy(&self) -> usize {
        self.thumbs.as_ref().map_or(0, Thumbs::busy)
    }

    fn info(&mut self, text: impl Into<String>) {
        self.status = Some(Status { text: text.into(), error: false, at: Instant::now() });
    }

    fn error(&mut self, text: impl Into<String>) {
        self.status = Some(Status { text: text.into(), error: true, at: Instant::now() });
    }

    fn open(&mut self, ctx: &egui::Context, dir: &Path, restore_selection: bool) {
        let s = &self.session;
        match Workspace::open(dir, s.view.clone(), s.columns) {
            Ok(mut ws) => {
                if restore_selection && s.last_dataset.as_deref() == Some(dir) {
                    ws.select(s.selected);
                }
                let budget = s.texture_budget_mb * 1024 * 1024;
                self.thumbs = Some(Thumbs::new(
                    ctx,
                    ws.project.state_dir().join("cache").join("thumbs"),
                    budget,
                ));
                let warnings = ws.project.warnings.join(" · ");
                let summary = format!(
                    "Opened {} ({} images, {} annotations)",
                    ws.project.base_name(),
                    ws.project.dataset.images().len(),
                    ws.project.dataset.annotations().len()
                );
                self.session.last_dataset = Some(dir.to_path_buf());
                if let Some(old) = &mut self.ws
                    && let Err(e) = old.save_rules()
                {
                    tracing::warn!("rules not saved: {e:#}");
                }
                self.ws = Some(ws);
                self.cluster = ClusterView::default();
                self.cluster.min_cluster_size = self.session.min_cluster_size;
                self.embeddings = None;
                self.embed_task = None;
                self.layout_task = None;
                self.todo = Default::default();
                if matches!(self.mode, Mode::Cluster | Mode::Todo | Mode::Rules) {
                    self.mode = Mode::Mosaic;
                }
                self.mosaic.ensure_visible = true;
                if warnings.is_empty() {
                    self.info(summary)
                } else {
                    self.info(format!("{summary} — {warnings}"))
                }
            }
            Err(e) => self.error(format!("Cannot open {}: {e:#}", dir.display())),
        }
    }

    fn pick_and_open(&mut self, ctx: &egui::Context) {
        if self.inference.is_running() || self.embed_task.is_some() || self.layout_task.is_some() {
            self.error(
                "Wait for the background run to finish (or cancel it) before switching datasets",
            );
            return;
        }
        let start = self.session.last_dataset.clone().unwrap_or_default();
        self.pick(ctx, Pick::Open, move || {
            rfd::FileDialog::new().set_directory(start).pick_folder().into_iter().collect()
        });
    }

    /// Shows a file dialog on its own thread: the blocking dialog would stop
    /// the event loop and the compositor would report "not responding".
    fn pick(
        &mut self,
        ctx: &egui::Context,
        what: Pick,
        dialog: impl FnOnce() -> Vec<PathBuf> + Send + 'static,
    ) {
        if self.pick.is_none() {
            self.pick = Some((what, Task::spawn("file dialog", ctx, move |_| dialog())));
        }
    }

    fn poll_pick(&mut self, ctx: &egui::Context) {
        let Some(paths) = self.pick.as_mut().and_then(|(_, t)| t.poll()) else { return };
        let Some((what, _)) = self.pick.take() else { return };
        let Some(first) = paths.first().cloned() else { return };
        match what {
            // Pending edits of the current dataset stay in its journal.
            Pick::Open => self.open(ctx, &first, false),
            Pick::ExportZip => self.finish_export(ctx, fastlabel_core::export::Target::Zip(first)),
            Pick::ExportFolder => {
                let out = first.join(self.export_name().1);
                if out.exists() {
                    self.error(format!("{} already exists", out.display()));
                } else {
                    self.finish_export(ctx, fastlabel_core::export::Target::Folder(out));
                }
            }
            Pick::Add => self.finish_add(ctx, paths),
            Pick::Model => self.session.model_path = Some(first),
            Pick::EmbedModel => self.session.embed_model_path = Some(first),
        }
    }

    /// Saves a new version. Returns false when saving failed.
    fn save(&mut self) -> bool {
        if self.inference.is_running() {
            self.error("Wait for the model run to finish (or cancel it) before saving");
            return false;
        }
        let Some(ws) = &mut self.ws else { return true };
        match ws.save() {
            Ok(Some(report)) => {
                let s = &report.summary;
                let name = report
                    .path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let mut parts = vec![format!("Saved {name}")];
                for (n, what) in [
                    (s.deleted_annotations, "boxes deleted"),
                    (s.accepted, "predictions accepted"),
                    (s.rejected, "rejected"),
                    (s.replaced, "replaced"),
                    (s.confirmed, "confirmed"),
                    (s.added + s.edited, "edited"),
                    (s.deleted_frames, "frames moved to trash"),
                    (s.moved_frames, "frames moved to another split"),
                ] {
                    if n > 0 {
                        parts.push(format!("{n} {what}"));
                    }
                }
                if let Some(t) = &mut self.thumbs {
                    t.retry_failed();
                }
                self.cluster.dirty = true;
                self.refresh_todo();
                if report.trash.failed.is_empty() {
                    self.info(parts.join(" · "));
                } else {
                    let files: Vec<String> = report
                        .trash
                        .failed
                        .iter()
                        .map(|(p, why)| format!("{} ({why})", p.display()))
                        .collect();
                    self.error(format!(
                        "{} — could not move: {}",
                        parts.join(" · "),
                        files.join(", ")
                    ));
                }
                self.mosaic.ensure_visible = true;
                if self.mode == Mode::Clean
                    && let Some(ws) = &self.ws
                {
                    // Saved decisions are resolved: continue with what is left.
                    let groups = ws.overlap_groups();
                    self.clean = Some(CleanState { groups, index: 0, digits: String::new() });
                    self.goto_group(0);
                }
                true
            }
            Ok(None) => {
                self.info("Nothing to save");
                true
            }
            Err(e) => {
                self.error(format!("Save failed, nothing was lost (edits are journaled): {e:#}"));
                false
            }
        }
    }

    pub(crate) fn run_model(&mut self, ctx: &egui::Context, scope: Scope) {
        if self.ws.is_none() {
            return;
        }
        let Some(model) = self.session.model_path.clone().filter(|p| p.is_file()) else {
            self.error("Choose an ONNX or .pt model file first (Model menu)");
            return;
        };
        let Some(model) = crate::inference::onnx_for(&model) else {
            if self.convert_task.is_none() {
                self.info(format!("Exporting {} to ONNX…", model.display()));
                let task = Task::spawn("ONNX export", ctx, move |_| {
                    crate::inference::export_pt(&model, &[])
                });
                self.convert_task = Some((scope, task));
            }
            return;
        };
        let Some(ws) = &self.ws else { return };
        let items = match scope {
            Scope::All => ws.inference_items(),
            Scope::Unpredicted => ws.unpredicted_items(),
            Scope::Frame(idx) => vec![ws.frame_item(idx)],
        };
        if items.is_empty() {
            self.error("No frames to run the model on");
            return;
        }
        self.inference.start(ctx, &model, self.session.confidence, scope, items);
    }

    fn poll_convert(&mut self, ctx: &egui::Context) {
        let Some(result) = self.convert_task.as_mut().and_then(|(_, t)| t.poll()) else { return };
        let Some((scope, _)) = self.convert_task.take() else { return };
        match result {
            Ok(_) => self.run_model(ctx, scope),
            Err(e) => self.error(e),
        }
    }

    fn poll_inference(&mut self) {
        let Some(outcome) = self.inference.poll() else { return };
        let Some(ws) = &mut self.ws else { return };
        let model = outcome.model.display().to_string();
        let applied = outcome.result.and_then(|r| {
            ws.apply_detections(
                outcome.scope == Scope::All,
                &model,
                outcome.confidence,
                &outcome.class_names,
                r,
            )
        });
        match applied {
            Ok(msg) => {
                self.mosaic.ensure_visible = true;
                self.info(format!("Model run done: {msg}"));
            }
            Err(e) => self.error(e),
        }
    }

    /// Starts computing missing uniformity values; returns false if none are missing.
    pub(crate) fn start_uniformity(&mut self, ctx: &egui::Context) -> bool {
        if self.uniformity_task.is_some() {
            return true;
        }
        let Some(ws) = &self.ws else { return false };
        let frames = ws.missing_uniformity();
        if frames.is_empty() {
            return false;
        }
        let threads =
            std::thread::available_parallelism().map_or(2, |n| n.get().saturating_sub(1).max(1));
        self.uniformity_task = Some(Task::spawn("color uniformity", ctx, move |p| {
            crate::metrics::compute(frames, threads, p)
        }));
        true
    }

    /// Saves pending edits, asks where to write, then exports in the
    /// background with the dialog's settings.
    fn start_export(&mut self, ctx: &egui::Context) {
        if self.export_task.is_some()
            || self.persons_task.is_some()
            || self.pick.is_some()
            || !self.save()
        {
            return;
        }
        let Some(ws) = self.ws.as_ref().filter(|ws| ws.project.source.is_some()) else { return };
        if !self.session.export.splits.contains(&true) {
            self.error("Choose at least one split to export");
            return;
        }
        let name = self.export_name().1;
        let start_dir = ws.project.dir.parent().unwrap_or(&ws.project.dir).to_path_buf();
        if self.session.export.zip {
            self.pick(ctx, Pick::ExportZip, move || {
                rfd::FileDialog::new()
                    .set_directory(&start_dir)
                    .set_file_name(format!("{name}.zip"))
                    .add_filter("zip", &["zip"])
                    .save_file()
                    .into_iter()
                    .collect()
            });
        } else {
            self.pick(ctx, Pick::ExportFolder, move || {
                rfd::FileDialog::new()
                    .set_title("Folder to create the export in")
                    .set_directory(&start_dir)
                    .pick_folder()
                    .into_iter()
                    .collect()
            });
        }
    }

    /// (dataset folder name, export name).
    fn export_name(&self) -> (String, String) {
        let Some(ws) = &self.ws else { return Default::default() };
        let p = &ws.project;
        let dir_name =
            p.dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let stem = p.source.as_ref().map(|s| s.file_name()).unwrap_or_default();
        let stem = stem.trim_start_matches("annotations.").trim_end_matches(".json");
        let mut name = format!("{dir_name}_{stem}_{}", self.session.export.format.slug());
        if let Some(sz) = self.export_imgsz() {
            name.push_str(&format!("_{sz}"));
        }
        (dir_name, name)
    }

    /// Long side the export downscales images to, if any.
    fn export_imgsz(&self) -> Option<u32> {
        let s = &self.session.export;
        s.resize.then(|| s.imgsz.unwrap_or(self.export_suggested_imgsz))
    }

    fn export_splits(&self) -> Vec<Split> {
        Split::ALL.into_iter().filter(|s| self.session.export.splits[*s as usize]).collect()
    }

    /// Runs person detection first when the dialog asks for it.
    fn finish_export(&mut self, ctx: &egui::Context, target: fastlabel_core::export::Target) {
        let Some(ws) = &self.ws else { return };
        let p = &ws.project;
        if !self.session.export.persons {
            let coco = p.dataset.coco.clone();
            return self.run_export(ctx, target, coco);
        }
        let (root, coco, splits) = (p.dir.clone(), p.dataset.coco.clone(), self.export_splits());
        let cache = p.state_dir().join("cache").join("persons");
        let task = Task::spawn("person detection", ctx, move |progress| {
            crate::persons::add(
                &root,
                &coco,
                &splits,
                &cache,
                &crate::persons::models_dir(),
                progress,
            )
        });
        self.persons_task = Some((target, task));
    }

    fn run_export(
        &mut self,
        ctx: &egui::Context,
        target: fastlabel_core::export::Target,
        coco: fastlabel_core::coco::CocoFile,
    ) {
        use fastlabel_core::export::{ExportOptions, Target};
        let Some(ws) = &self.ws else { return };
        let p = &ws.project;
        let splits = self.export_splits();
        let resize = self.export_imgsz().map(|max_side| fastlabel_core::export::Resize {
            max_side,
            resize: fastlabel_media::decode::resize_batch,
        });
        let opts = ExportOptions {
            format: self.session.export.format,
            splits,
            name: self.export_name().0,
            resize,
            min_box_height: self.session.min_box_height,
        };
        let root = p.dir.clone();
        self.export_task = Some(Task::spawn("export", ctx, move |progress| {
            use std::sync::atomic::Ordering;
            let (Target::Zip(out) | Target::Folder(out)) = &target;
            fastlabel_core::export::export(&root, &coco, &opts, &target, |done, total| {
                progress.done.store(done, Ordering::Relaxed);
                progress.total.store(total, Ordering::Relaxed);
                !progress.cancel.load(Ordering::Relaxed)
            })
            .map(|mut r| {
                r.missing.truncate(5);
                (r, out.clone())
            })
            .map_err(|e| format!("{e} ({})", out.display()))
        }));
    }

    /// Copies picked images (files or whole folders) into `images/` under
    /// content-hash names in the background; registered when done.
    fn start_add(&mut self, ctx: &egui::Context, folders: bool) {
        let Some(ws) = &self.ws else { return };
        if self.add_task.is_some() {
            return;
        }
        let dialog = rfd::FileDialog::new()
            .set_directory(ws.project.dir.parent().unwrap_or(&ws.project.dir));
        self.pick(ctx, Pick::Add, move || {
            let picked = if folders {
                dialog.set_title("Folders of images to add").pick_folders()
            } else {
                dialog
                    .set_title("Images to add")
                    .add_filter("images", &fastlabel_core::yolo::IMAGE_EXTENSIONS)
                    .pick_files()
            };
            picked.unwrap_or_default()
        });
    }

    fn finish_add(&mut self, ctx: &egui::Context, paths: Vec<PathBuf>) {
        let Some(ws) = &self.ws else { return };
        if self.add_task.is_some() {
            return;
        }
        let dir = ws.project.dir.clone();
        self.add_task = Some(Task::spawn("adding images", ctx, move |progress| {
            use std::sync::atomic::Ordering;
            let sources = fastlabel_core::add::collect(&paths);
            fastlabel_core::add::copy_sources(&dir, &sources, |done, total| {
                progress.done.store(done, Ordering::Relaxed);
                progress.total.store(total, Ordering::Relaxed);
                !progress.cancel.load(Ordering::Relaxed)
            })
        }));
    }

    fn poll_add(&mut self) {
        let Some(r) = self.add_task.as_mut().and_then(Task::poll) else { return };
        self.add_task = None;
        let Some(ws) = &mut self.ws else { return };
        let split = self.session.add_split;
        match ws.register_added(&r.copied, split) {
            Ok(n) => {
                self.cluster.dirty = true;
                self.mosaic.ensure_visible = true;
                self.refresh_todo();
                let mut msg = format!("Added {n} images");
                if let Some(s) = split {
                    msg.push_str(&format!(" to {}", s.name()));
                }
                if !r.duplicates.is_empty() {
                    msg.push_str(&format!(", {} already in the dataset", r.duplicates.len()));
                }
                if r.cancelled {
                    msg.push_str(" (cancelled)");
                }
                msg.push_str(
                    " — saved with the next version; Model → Run on frames without predictions",
                );
                if r.failed.is_empty() {
                    self.info(msg);
                } else {
                    let (name, why) = &r.failed[0];
                    self.error(format!("{msg}; {} failed ({name}: {why})", r.failed.len()));
                }
            }
            Err(e) => self.error(format!("Images copied into images/ but not registered: {e}")),
        }
    }

    /// Format, splits and destination; Export saves, then asks where.
    fn export_dialog(&mut self, ctx: &egui::Context) {
        use fastlabel_core::export::ExportFormat;
        let Some(ws) = &self.ws else { return };
        let totals = ws.split_totals();
        let mut export = false;
        let modal = egui::Modal::new(egui::Id::new("export-dialog")).show(ctx, |ui| {
            ui.set_width(440.0);
            ui.heading("Export");
            let s = &mut self.session.export;
            ui.label("Format");
            for f in ExportFormat::ALL {
                ui.radio_value(&mut s.format, f, f.label());
            }
            ui.add_space(6.0);
            ui.label("Splits");
            ui.horizontal(|ui| {
                for split in Split::ALL {
                    let k = split as usize;
                    ui.checkbox(&mut s.splits[k], format!("{} ({})", split.name(), totals[k]));
                }
            });
            if totals[3] > 0 {
                ui.label(
                    RichText::new(format!("{} frames without a split are left out", totals[3]))
                        .color(Color32::from_rgb(245, 150, 30)),
                );
            }
            ui.add_space(6.0);
            ui.label("Destination");
            ui.horizontal(|ui| {
                ui.radio_value(&mut s.zip, true, "Zip file");
                ui.radio_value(&mut s.zip, false, "Folder (images hardlinked)");
            });
            ui.add_space(6.0);
            let suggested = format!("{} (report suggestion)", self.export_suggested_imgsz);
            ui.horizontal(|ui| {
                ui.checkbox(&mut s.resize, "Downscale images to a long side of");
                ui.add_enabled_ui(s.resize, |ui| {
                    let text = s.imgsz.map_or(suggested.clone(), |v| v.to_string());
                    egui::ComboBox::from_id_salt("export-imgsz").selected_text(text).show_ui(
                        ui,
                        |ui| {
                            ui.selectable_value(&mut s.imgsz, None, &suggested);
                            for sz in crate::report::IMGSZ {
                                ui.selectable_value(&mut s.imgsz, Some(sz), sz.to_string());
                            }
                        },
                    );
                });
            });
            if s.resize {
                ui.label(
                    RichText::new("Smaller images are kept as is; labels follow. Training then skips decoding full frames every epoch.")
                        .weak(),
                );
            }
            if self.session.min_box_height > 0.0 {
                ui.label(
                    RichText::new(format!(
                        "Boxes under {} px high are left out (Settings).",
                        self.session.min_box_height
                    ))
                    .weak(),
                );
            }
            ui.add_space(6.0);
            ui.checkbox(&mut s.persons, "Add person boxes (yolo26x)").on_hover_text(format!(
                "Detects people (score ≥ {}) on the exported frames and adds them as a \"{}\" class \
                 in the export only. Results are cached per frame; the first run on a frame takes \
                 a few seconds on the CPU.",
                crate::persons::CONF,
                crate::persons::CATEGORY
            ));
            ui.add_space(6.0);
            ui.label(
                RichText::new("Pending edits are saved first (new version); frames marked F are not exported.")
                    .weak(),
            );
            ui.horizontal(|ui| {
                if ui.button("Export…").clicked() {
                    export = true;
                }
                if ui.button("Cancel").clicked() {
                    self.export_dialog = false;
                }
            });
        });
        if modal.should_close() {
            self.export_dialog = false;
        }
        if export {
            self.export_dialog = false;
            self.start_export(ctx);
        }
    }

    fn poll_tasks(&mut self, ctx: &egui::Context) {
        if let Some(values) = self.uniformity_task.as_mut().and_then(Task::poll) {
            self.uniformity_task = None;
            if let Some(ws) = &mut self.ws {
                let n = values.len();
                ws.opts.sort = SortKey::Uniformity;
                let result = ws.add_uniformity(values);
                ws.select(0);
                match result {
                    Ok(()) => self.info(format!("Color uniformity computed for {n} boxes")),
                    Err(e) => self.error(format!("Cannot store color uniformity: {e:#}")),
                }
                self.mosaic.ensure_visible = true;
            }
        }
        if let Some(result) = self.persons_task.as_mut().and_then(|(_, t)| t.poll())
            && let Some((target, _)) = self.persons_task.take()
        {
            match result {
                Ok(Some(coco)) => self.run_export(ctx, target, coco),
                Ok(None) => self.info("Export cancelled"),
                Err(e) => self.error(format!("Person detection failed: {e}")),
            }
        }
        if let Some(result) = self.export_task.as_mut().and_then(Task::poll) {
            self.export_task = None;
            match result {
                Ok((r, _)) if r.cancelled => self.info("Export cancelled"),
                Ok((r, out)) => {
                    let [t, v, te] = r.per_split;
                    let mut msg = format!(
                        "Exported {} images (train {t}, val {v}, test {te}) to {}",
                        r.images(),
                        out.display()
                    );
                    if r.forced_test > 0 {
                        msg.push_str(&format!(
                            "; {} moved to test by test_sources.txt",
                            r.forced_test
                        ));
                    }
                    if r.no_split > 0 {
                        msg.push_str(&format!("; {} without a split left out", r.no_split));
                    }
                    if r.missing.is_empty() {
                        self.info(msg);
                    } else {
                        self.error(format!("{msg}; missing: {}", r.missing.join(", ")));
                    }
                }
                Err(e) => self.error(format!("Export failed: {e}")),
            }
        }
        let _ = ctx;
    }

    fn settings_menu(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("Settings", |ui| {
            let s = &mut self.session;
            ui.horizontal(|ui| {
                ui.label("Min box height (px)");
                ui.add(egui::DragValue::new(&mut s.min_box_height).range(0.0..=1000.0).speed(0.5));
            })
            .response
            .on_hover_text("Smaller GT boxes and predictions are tagged <N px and left out of exports; 0 keeps all");
            ui.horizontal(|ui| {
                ui.label("Thumbnail GPU memory (MB)");
                ui.add(egui::DragValue::new(&mut s.texture_budget_mb).range(32..=4096).speed(8));
            })
            .response
            .on_hover_text("Applies when a dataset is opened");
        });
    }

    fn model_menu(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        ui.menu_button("Model", |ui| {
            ui.set_min_width(380.0);
            let mut path = self
                .session
                .model_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default();
            ui.horizontal(|ui| {
                if ui
                    .add(
                        egui::TextEdit::singleline(&mut path)
                            .hint_text("model.onnx or best.pt")
                            .desired_width(280.0),
                    )
                    .changed()
                {
                    self.session.model_path = (!path.is_empty()).then(|| PathBuf::from(&path));
                }
                if ui.button("Browse…").clicked() {
                    self.pick(&ctx, Pick::Model, pick_detector);
                }
            });
            ui.add(
                egui::Slider::new(&mut self.session.confidence, 0.01..=1.0)
                    .step_by(0.01)
                    .text("confidence"),
            );
            let idle = !self.inference.is_running() && self.ws.is_some();
            if ui.add_enabled(idle, egui::Button::new("Run on all frames")).clicked() {
                self.run_model(&ctx, Scope::All);
                ui.close();
            }
            let missing = self.ws.as_ref().map_or(0, |ws| ws.unpredicted_items().len());
            if ui
                .add_enabled(
                    idle && missing > 0,
                    egui::Button::new(format!("Run on frames without predictions ({missing})")),
                )
                .on_hover_text(
                    "e.g. images newly copied into images/; existing predictions are kept",
                )
                .clicked()
            {
                self.run_model(&ctx, Scope::Unpredicted);
                ui.close();
            }
            let frame = self.ws.as_ref().and_then(|ws| ws.selected_crop()).map(|c| c.image_idx);
            if ui
                .add_enabled(
                    idle && frame.is_some(),
                    egui::Button::new("Run on the selected crop's frame"),
                )
                .clicked()
                && let Some(idx) = frame
            {
                self.run_model(&ctx, Scope::Frame(idx));
                ui.close();
            }
        });
    }

    fn save_session(&mut self) {
        if let Some(ws) = &self.ws {
            self.session.selected = ws.selected;
            self.session.view = ws.opts.clone();
            self.session.columns = ws.columns;
        }
        self.session.min_cluster_size = self.cluster.min_cluster_size;
        if let Some(path) = &self.session_path
            && let Err(e) = self.session.save(path)
        {
            tracing::warn!("cannot save session: {e}");
        }
    }

    fn handle_keys(&mut self, ctx: &egui::Context) {
        let events = ctx.input(|i| i.events.clone());
        for e in &events {
            if matches!(e, egui::Event::Key { .. } | egui::Event::Text(_) | egui::Event::Ime(_)) {
                tracing::debug!(
                    event = ?e,
                    focused = ?ctx.memory(|m| m.focused()),
                    text_edit = ctx.text_edit_focused(),
                    "input"
                );
            }
        }
        // Only a text field being edited takes the keyboard. A focused
        // button (e.g. after a mouse click) must not swallow shortcuts.
        if ctx.text_edit_focused() {
            return;
        }
        for event in events {
            let egui::Event::Key { key, pressed: true, repeat, modifiers, .. } = event else {
                continue;
            };
            self.handle_key(ctx, key, repeat, modifiers);
        }
        // Keep egui from using these keys too: Enter / Space would "press" a
        // focused button. Tab already moved egui's focus this frame (focus
        // traversal), so that focus is dropped at the end of the frame.
        ctx.input_mut(|i| {
            for key in [Key::Tab, Key::Enter, Key::Space] {
                for m in [Modifiers::NONE, Modifiers::SHIFT] {
                    if i.consume_key(m, key) && key == Key::Tab {
                        self.drop_focus = true;
                    }
                }
            }
        });
    }

    fn set_mode(&mut self, mode: Mode) {
        let Some(ws) = &mut self.ws else { return };
        if self.mode == Mode::Rules
            && mode != Mode::Rules
            && let Err(e) = ws.save_rules()
        {
            self.error(format!("Rules not saved: {e:#}"));
            return;
        }
        if mode != Mode::Frame {
            ws.frame_order = None;
            self.frame_from_todo = false;
        }
        match mode {
            Mode::Frame => ws.focus_selected_crop(),
            Mode::Todo => self.todo.refresh(ws),
            Mode::Mosaic => {
                if matches!(self.mode, Mode::Cluster | Mode::Todo) {
                    // Split moves change what the split filter shows.
                    ws.rebuild();
                }
                ws.select_crop_of_frame();
                self.mosaic.ensure_visible = true;
            }
            Mode::Clean | Mode::Cluster | Mode::Rules => {}
        }
        self.mode = mode;
    }

    fn handle_key(&mut self, ctx: &egui::Context, key: Key, repeat: bool, m: Modifiers) {
        if self.export_dialog {
            return;
        }
        if self.ws.is_none() {
            if key == Key::O && m.command {
                self.pick_and_open(ctx);
            }
            return;
        }
        let plain = !m.command && !m.alt;
        // Shortcuts valid everywhere.
        let result = match key {
            Key::S if plain && !repeat => {
                if self.save() {
                    self.popup = false;
                }
                return;
            }
            Key::Q if plain && !repeat => {
                if self.popup {
                    self.popup = false;
                } else if self.mode == Mode::Clean {
                    self.exit_clean();
                } else if matches!(self.mode, Mode::Cluster | Mode::Todo | Mode::Rules) {
                    self.set_mode(Mode::Mosaic);
                } else {
                    self.inference.shutdown();
                    if self.save() {
                        self.exit_ok = true;
                        ctx.send_viewport_cmd(ViewportCommand::Close);
                    }
                }
                return;
            }
            Key::O if m.command => {
                self.pick_and_open(ctx);
                return;
            }
            Key::Tab | Key::M if plain && !m.shift && !self.popup && self.mode == Mode::Clean => {
                self.exit_clean();
                return;
            }
            Key::Tab | Key::M | Key::C
                if plain && !m.shift && !self.popup && self.mode == Mode::Cluster =>
            {
                self.set_mode(Mode::Mosaic);
                return;
            }
            Key::C if plain && !m.shift && !repeat && !self.popup && self.mode == Mode::Mosaic => {
                self.start_cluster_mode(ctx);
                return;
            }
            Key::Tab | Key::M | Key::T | Key::Escape
                if plain && !m.shift && !self.popup && self.mode == Mode::Todo =>
            {
                self.set_mode(Mode::Mosaic);
                return;
            }
            Key::T if plain && !m.shift && !repeat && !self.popup && self.mode == Mode::Mosaic => {
                self.set_mode(Mode::Todo);
                return;
            }
            Key::Tab | Key::M | Key::T
                if plain
                    && !m.shift
                    && !self.popup
                    && self.mode == Mode::Frame
                    && self.frame_from_todo =>
            {
                self.back_to_todo();
                return;
            }
            Key::Escape
                if !self.popup
                    && self.mode == Mode::Frame
                    && self.frame_from_todo
                    && self.ws.as_ref().is_some_and(|ws| ws.frame.sel.is_none()) =>
            {
                self.back_to_todo();
                return;
            }
            Key::Tab | Key::M if plain && !m.shift && !self.popup => {
                let next = if self.mode == Mode::Mosaic { Mode::Frame } else { Mode::Mosaic };
                self.set_mode(next);
                return;
            }
            Key::U if plain => self.ws.as_mut().map(Workspace::undo),
            Key::Z if m.command && m.shift => self.ws.as_mut().map(Workspace::redo),
            Key::Z if m.command => self.ws.as_mut().map(Workspace::undo),
            Key::Y if m.command => self.ws.as_mut().map(Workspace::redo),
            _ => None,
        };
        if let Some(r) = result {
            self.mosaic.ensure_visible = true;
            if let Err(e) = r {
                self.error(e);
            }
            return;
        }
        if self.mode == Mode::Clean
            && !self.popup
            && let Some(r) = self.clean_key(key, repeat)
        {
            if let Err(e) = r {
                self.error(e);
            }
            return;
        }
        if self.mode == Mode::Cluster && !self.popup {
            if let Some(event) = self.cluster_key(key, repeat, m) {
                self.cluster_event(ctx, event);
            }
            return;
        }
        if self.mode == Mode::Rules && !self.popup {
            if key == Key::Escape {
                self.set_mode(Mode::Mosaic);
            }
            return;
        }
        if self.mode == Mode::Todo && !self.popup {
            if let Err(e) = self.todo_key(key, repeat, m) {
                self.error(e);
            }
            return;
        }
        let r = if self.popup || self.mode != Mode::Mosaic {
            self.frame_key(key, repeat, m)
        } else {
            self.mosaic_key(key, repeat, m)
        };
        if let Err(e) = r {
            self.error(e);
        }
    }

    fn mosaic_key(&mut self, key: Key, repeat: bool, m: Modifiers) -> Result<(), String> {
        let page = self.mosaic.rows_per_screen.max(1) as isize;
        let Some(ws) = &mut self.ws else { return Ok(()) };
        let mut result = Ok(());
        match key {
            Key::ArrowLeft | Key::H if !m.command => ws.move_selection(-1, 0),
            Key::ArrowRight | Key::L if !m.command => ws.move_selection(1, 0),
            Key::ArrowUp | Key::K => ws.move_selection(0, -1),
            Key::ArrowDown | Key::J => ws.move_selection(0, 1),
            Key::PageUp => ws.move_selection(0, -page),
            Key::PageDown => ws.move_selection(0, page),
            Key::Home => ws.select(0),
            Key::End => ws.select(usize::MAX),
            // Marks ignore keyboard auto-repeat: holding a key must not flip-flop.
            Key::D | Key::G | Key::F | Key::R if !repeat && !m.command => {
                let mark = match key {
                    Key::D => Mark::Delete,
                    Key::G => Mark::Good,
                    Key::F => Mark::Frame,
                    _ => Mark::Replace,
                };
                result = if m.shift && mark != Mark::Replace {
                    ws.mark_row(mark)
                } else {
                    ws.mark_selected(mark)
                };
            }
            Key::Enter | Key::E if !repeat && !m.command && ws.selected_crop().is_some() => {
                ws.focus_selected_crop();
                self.popup = true;
                return Ok(());
            }
            _ => return Ok(()),
        }
        self.mosaic.ensure_visible = true;
        result
    }

    fn frame_key(&mut self, key: Key, repeat: bool, m: Modifiers) -> Result<(), String> {
        // The cleaning view behaves like the popup: no frame navigation.
        let popup = self.popup || self.mode == Mode::Clean;
        let Some(ws) = &mut self.ws else { return Ok(()) };
        if ws.project.dataset.images().is_empty() {
            return Ok(());
        }
        const DIGITS: [Key; 9] = [
            Key::Num1,
            Key::Num2,
            Key::Num3,
            Key::Num4,
            Key::Num5,
            Key::Num6,
            Key::Num7,
            Key::Num8,
            Key::Num9,
        ];
        match key {
            Key::ArrowLeft | Key::H | Key::ArrowRight | Key::L if !popup => {
                let dir = if matches!(key, Key::ArrowLeft | Key::H) { -1 } else { 1 };
                ws.next_frame(dir, m.command)
            }
            Key::Home | Key::End if !popup => {
                if let Some(i) = ws.edge_frame(key == Key::End) {
                    ws.go_to_frame(i);
                }
                Ok(())
            }
            Key::J | Key::K | Key::ArrowDown | Key::ArrowUp => {
                cycle_selection(ws, if matches!(key, Key::K | Key::ArrowUp) { -1 } else { 1 });
                Ok(())
            }
            Key::D if !repeat => ws.mark_sel(Mark::Delete),
            Key::G
                if !repeat && ws.frame.sel.is_none() && ws.frame_is_empty(ws.frame.image_idx) =>
            {
                // No box at all: background, nothing to annotate; then move on.
                let idx = ws.frame.image_idx;
                ws.toggle_confirm_frame(idx)?;
                if !popup
                    && ws.todo_tile(idx).confirmed
                    && let Some(i) = ws.step_frame(1, false)
                {
                    ws.go_to_frame(i);
                }
                Ok(())
            }
            Key::G if !repeat => ws.mark_sel(Mark::Good),
            Key::F if !repeat => {
                ws.toggle_frame_delete()?;
                // Like the mosaic, F moves on (except in the popup).
                if !popup
                    && ws.is_frame_deleted(ws.frame.image_idx)
                    && let Some(i) = ws.step_frame(1, false)
                {
                    ws.go_to_frame(i);
                }
                Ok(())
            }
            Key::Escape if popup => {
                self.popup = false;
                Ok(())
            }
            Key::Enter | Key::E if popup && !repeat => {
                self.popup = false;
                Ok(())
            }
            Key::Escape => {
                ws.frame.sel = None;
                Ok(())
            }
            k => match DIGITS.iter().position(|d| *d == k) {
                Some(n) if !repeat => ws.set_category_index(n),
                _ => Ok(()),
            },
        }
    }

    /// Opens cluster mode, loading the cached embeddings and layout; the
    /// layout is computed right away when every frame has an embedding.
    pub(crate) fn start_cluster_mode(&mut self, ctx: &egui::Context) {
        let Some(ws) = &self.ws else { return };
        let state = ws.project.state_dir();
        if self.embeddings.is_none() {
            self.embeddings = Some(Embeddings::load(&state.join(embeddings::EMBEDDINGS_FILE)));
        }
        if self.cluster.layout.is_none()
            && let Some(l) = SavedLayout::load(&state.join(embeddings::LAYOUT_FILE))
        {
            self.cluster.set_layout(l);
        }
        self.cluster.dirty = true;
        self.popup = false;
        self.mode = Mode::Cluster;
        let has_vectors = self.embeddings.as_ref().is_some_and(|e| e.len() > 0);
        if self.cluster.layout.is_none() && has_vectors && self.embedding_items().is_empty() {
            self.start_layout(ctx);
        }
    }

    /// Model id of the configured embedder, if the file exists.
    fn embed_model(&self) -> Option<(PathBuf, String)> {
        let path = self.session.embed_model_path.clone().filter(|p| p.is_file())?;
        let id = embeddings::model_id(&path);
        Some((path, id))
    }

    /// Frames without an embedding from the configured model: (image index, key).
    fn embedding_items(&self) -> Vec<(usize, u64)> {
        let (Some(ws), Some(emb)) = (&self.ws, &self.embeddings) else { return Vec::new() };
        let other_model =
            self.embed_model().is_some_and(|(_, id)| emb.len() > 0 && id != emb.model);
        ws.project
            .dataset
            .images()
            .iter()
            .enumerate()
            .map(|(i, img)| (i, embeddings::key(img.id, &img.file_name)))
            .filter(|(_, k)| other_model || emb.get(*k).is_none())
            .collect()
    }

    fn start_embeddings(&mut self, ctx: &egui::Context) {
        let Some((model_path, model)) = self.embed_model() else {
            self.error("Choose an ONNX embedding model first (e.g. from scripts/export_dinov2.py)");
            return;
        };
        let Some(ws) = &self.ws else { return };
        let ds = &ws.project.dataset;
        let todo = self.embedding_items();
        let keys: Vec<u64> = todo.iter().map(|(_, k)| *k).collect();
        let items: Vec<fastlabel_infer::BatchItem> = todo
            .iter()
            .enumerate()
            .map(|(n, (i, _))| fastlabel_infer::BatchItem {
                key: n,
                path: ds.image_path(&ds.images()[*i]),
            })
            .collect();
        if items.is_empty() {
            return;
        }
        self.embed_task = Some(Task::spawn("embeddings", ctx, move |progress| {
            let mut out = EmbedOutcome {
                model,
                vectors: Vec::new(),
                failed: Vec::new(),
                cancelled: false,
                error: None,
            };
            match fastlabel_infer::Embedder::load(&model_path, 0) {
                Ok(mut emb) => {
                    let r = fastlabel_infer::run_embed(&mut emb, items, progress);
                    out.vectors = r.vectors.into_iter().map(|(n, v)| (keys[n], v)).collect();
                    out.failed = r.failed.into_iter().map(|(_, e)| e).collect();
                    out.cancelled = r.cancelled;
                }
                Err(e) => out.error = Some(format!("Cannot load {}: {e}", model_path.display())),
            }
            out
        }));
    }

    /// Lays out every frame that has an embedding.
    fn start_layout(&mut self, ctx: &egui::Context) {
        let (Some(ws), Some(emb)) = (&self.ws, &self.embeddings) else { return };
        let mut keys = Vec::new();
        let mut vectors = Vec::new();
        for img in ws.project.dataset.images() {
            let k = embeddings::key(img.id, &img.file_name);
            if let Some(v) = emb.get(k) {
                keys.push(k);
                vectors.push(v.to_vec());
            }
        }
        if keys.is_empty() {
            return;
        }
        let (model, mcs) = (emb.model.clone(), self.cluster.min_cluster_size);
        self.layout_task = Some(Task::spawn("clustering", ctx, move |progress| {
            let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();
            let l = crate::cluster::layout(&refs, mcs, 42, progress)?;
            Some(SavedLayout {
                model,
                min_cluster_size: mcs,
                keys,
                points: l.points,
                labels: l.labels,
            })
        }));
    }

    /// New cluster labels for the current layout (minimum size changed).
    fn start_recluster(&mut self, ctx: &egui::Context) {
        let Some(layout) = self.cluster.layout.clone() else { return };
        let mcs = self.cluster.min_cluster_size;
        self.layout_task = Some(Task::spawn("clustering", ctx, move |progress| {
            use std::sync::atomic::Ordering;
            progress.total.store(1, Ordering::Relaxed);
            let labels = crate::cluster::hdbscan(&layout.points, mcs);
            progress.done.store(1, Ordering::Relaxed);
            Some(SavedLayout { min_cluster_size: mcs, labels, ..layout })
        }));
    }

    /// Writes an HTML report in the background and opens it in the browser.
    /// `scope`: image ids, `None` for all frames.
    pub(crate) fn start_report(
        &mut self,
        ctx: &egui::Context,
        title: String,
        slug: String,
        scope: Option<std::collections::BTreeSet<fastlabel_core::ImageId>>,
    ) {
        if self.report_task.is_some() {
            return;
        }
        let Some(ws) = &self.ws else { return };
        let state = ws.project.state_dir();
        let saved_layout;
        let layout = match &self.cluster.layout {
            Some(l) => Some(l),
            None => {
                saved_layout = SavedLayout::load(&state.join(embeddings::LAYOUT_FILE));
                saved_layout.as_ref()
            }
        };
        let input = ws.report_input(title, scope.as_ref(), layout, self.session.min_box_height);
        self.report_task = Some(Task::spawn("report", ctx, move |_| {
            let emb = Embeddings::load(&state.join(embeddings::EMBEDDINGS_FILE));
            crate::report::write(&input, &emb, &state.join("reports"), &slug)
                .map_err(|e| e.to_string())
        }));
    }

    fn poll_cluster_tasks(&mut self, ctx: &egui::Context) {
        if let Some(result) = self.report_task.as_mut().and_then(Task::poll) {
            self.report_task = None;
            match result {
                Ok(path) => {
                    let opened = std::process::Command::new("xdg-open").arg(&path).spawn().is_ok();
                    let how = if opened { "opened in the browser" } else { "open it in a browser" };
                    self.info(format!("Report written to {} ({how})", path.display()));
                }
                Err(e) => self.error(format!("Cannot write the report: {e}")),
            }
        }
        if let Some(out) = self.embed_task.as_mut().and_then(Task::poll) {
            self.embed_task = None;
            let (Some(ws), Some(emb)) = (&self.ws, &mut self.embeddings) else { return };
            if let Some(e) = out.error {
                self.error(e);
                return;
            }
            let n = out.vectors.len();
            emb.use_model(&out.model);
            emb.extend(out.vectors);
            let saved = emb.save(&ws.project.state_dir().join(embeddings::EMBEDDINGS_FILE));
            self.cluster.dirty = true;
            if let Err(e) = saved {
                self.error(format!("Cannot store embeddings: {e}"));
            } else if !out.failed.is_empty() {
                self.error(format!(
                    "{n} frames embedded, {} failed ({})",
                    out.failed.len(),
                    out.failed[0]
                ));
            } else if out.cancelled {
                self.info(format!("Embeddings cancelled ({n} frames done and kept)"));
            } else {
                self.info(format!("{n} frames embedded; computing the layout…"));
                self.start_layout(ctx);
            }
        }
        if let Some(result) = self.layout_task.as_mut().and_then(Task::poll) {
            self.layout_task = None;
            let Some(layout) = result else {
                self.info("Clustering cancelled");
                return;
            };
            let clusters = layout.labels.iter().copied().max().map_or(0, |m| m + 1);
            let noise = layout.labels.iter().filter(|l| **l < 0).count();
            if let Some(ws) = &self.ws
                && let Err(e) = layout.save(&ws.project.state_dir().join(embeddings::LAYOUT_FILE))
            {
                tracing::warn!("cannot store the cluster layout: {e}");
            }
            self.info(format!(
                "{} frames in {clusters} clusters ({noise} unclustered)",
                layout.keys.len()
            ));
            self.cluster.set_layout(layout);
        }
    }

    /// Re-lists the to-do frames (after a save or an add); frame mode opened
    /// from To do keeps navigating the new list.
    fn refresh_todo(&mut self) {
        let Some(ws) = &mut self.ws else { return };
        if self.mode == Mode::Todo || self.frame_from_todo {
            self.todo.refresh(ws);
        }
        if ws.frame_order.is_some() {
            ws.frame_order = Some(self.todo.frames.clone());
        }
    }

    /// Frame mode on a to-do frame (k-th tile), navigating the to-do list.
    fn open_todo_frame(&mut self, k: usize) {
        let Some(ws) = &mut self.ws else { return };
        let Some(&idx) = self.todo.frames.get(k) else { return };
        self.todo.selected = k;
        ws.frame_order = Some(self.todo.frames.clone());
        ws.go_to_frame(idx);
        ws.frame.sel = None;
        self.popup = false;
        self.frame_from_todo = true;
        self.mode = Mode::Frame;
    }

    /// From frame mode back to the To do grid, on the frame reached.
    fn back_to_todo(&mut self) {
        let Some(ws) = &mut self.ws else { return };
        if let Some(k) = self.todo.frames.iter().position(|&f| f == ws.frame.image_idx) {
            self.todo.selected = k;
        }
        ws.frame_order = None;
        self.frame_from_todo = false;
        self.set_mode(Mode::Todo);
    }

    /// Keys of the To do tab.
    fn todo_key(&mut self, key: Key, repeat: bool, m: Modifiers) -> Result<(), String> {
        let Some(ws) = &mut self.ws else { return Ok(()) };
        let cols = ws.columns.max(1) as isize;
        let t = &mut self.todo;
        match key {
            Key::ArrowLeft | Key::H => t.move_selection(-1),
            Key::ArrowRight | Key::L => t.move_selection(1),
            Key::ArrowUp | Key::K => t.move_selection(-cols),
            Key::ArrowDown | Key::J => t.move_selection(cols),
            Key::PageUp => t.move_selection(-cols * 4),
            Key::PageDown => t.move_selection(cols * 4),
            Key::Home => t.move_selection(isize::MIN / 2),
            Key::End => t.move_selection(isize::MAX / 2),
            Key::Enter | Key::E if !repeat && !m.command => {
                let k = t.selected;
                self.open_todo_frame(k);
            }
            Key::G | Key::F if !repeat && !m.command => {
                let Some(idx) = t.selected_frame() else { return Ok(()) };
                let r = if key == Key::G {
                    ws.toggle_confirm_frame(idx)
                } else {
                    ws.toggle_frame_delete_idx(idx)
                };
                let t = ws.todo_tile(idx);
                if t.confirmed || t.deleted {
                    self.todo.advance(ws);
                }
                return r;
            }
            _ => {}
        }
        Ok(())
    }

    /// Keys of cluster mode; `None` when the key does nothing there.
    fn cluster_key(&mut self, key: Key, repeat: bool, m: Modifiers) -> Option<ClusterEvent> {
        if repeat {
            return None;
        }
        match key {
            Key::Escape => Some(ClusterEvent::Exit),
            Key::Num1 => Some(ClusterEvent::Move(Split::Train)),
            Key::Num2 => Some(ClusterEvent::Move(Split::Val)),
            Key::Num3 => Some(ClusterEvent::Move(Split::Test)),
            Key::F if !m.command => Some(ClusterEvent::ToggleDelete),
            Key::Enter => Some(ClusterEvent::ShowInMosaic),
            Key::A if m.command => {
                if let Some(ws) = &self.ws {
                    self.cluster.select_all_visible(ws);
                }
                None
            }
            _ => None,
        }
    }

    pub(crate) fn cluster_event(&mut self, ctx: &egui::Context, event: ClusterEvent) {
        let frames =
            self.ws.as_ref().map(|ws| self.cluster.selected_frames(ws)).unwrap_or_default();
        let needs_selection = matches!(
            event,
            ClusterEvent::Move(_)
                | ClusterEvent::ShowInMosaic
                | ClusterEvent::ToggleDelete
                | ClusterEvent::ReportSelection
        );
        if needs_selection && frames.is_empty() {
            self.error("Select frames first: drag around points");
            return;
        }
        let result = match event {
            ClusterEvent::ComputeEmbeddings => {
                self.start_embeddings(ctx);
                Ok(())
            }
            ClusterEvent::Relayout => {
                self.start_layout(ctx);
                Ok(())
            }
            ClusterEvent::Recluster => {
                self.start_recluster(ctx);
                Ok(())
            }
            ClusterEvent::Move(split) => {
                let Some(ws) = &mut self.ws else { return };
                let r = ws.move_to_split(&frames, split);
                if r.is_ok() {
                    self.info(format!(
                        "{} frames → {} (files are moved on save)",
                        frames.len(),
                        split.name()
                    ));
                }
                r
            }
            ClusterEvent::ToggleDelete => {
                self.ws.as_mut().map_or(Ok(()), |ws| ws.toggle_frames_delete(&frames))
            }
            ClusterEvent::ShowInMosaic => {
                let Some(ws) = &mut self.ws else { return };
                ws.opts.frames = Some(std::sync::Arc::new(self.cluster.selection.clone()));
                self.set_mode(Mode::Mosaic);
                if let Some(ws) = &mut self.ws {
                    ws.rebuild();
                    ws.select(0);
                }
                Ok(())
            }
            ClusterEvent::Open(idx) => {
                if let Some(ws) = &mut self.ws {
                    ws.go_to_frame(idx);
                    self.popup = true;
                }
                Ok(())
            }
            ClusterEvent::BrowseModel => {
                self.pick(ctx, Pick::EmbedModel, pick_onnx);
                Ok(())
            }
            ClusterEvent::Exit => {
                self.set_mode(Mode::Mosaic);
                Ok(())
            }
            ClusterEvent::AutoSplit => {
                let Some(ws) = &mut self.ws else { return };
                let groups = self.cluster.split_groups(ws);
                let sizes: Vec<usize> = groups.iter().map(Vec::len).collect();
                let splits = crate::cluster::auto_split(&sizes, &self.session.auto_split);
                let assign: Vec<(usize, Split)> = groups
                    .iter()
                    .zip(&splits)
                    .flat_map(|(g, s)| g.iter().map(move |&i| (i, *s)))
                    .collect();
                match ws.assign_splits(&assign) {
                    Ok(n) => {
                        let shares = crate::cluster_view::split_shares(ws.split_totals());
                        self.info(format!(
                            "Auto-split: {n} frames changed split — {shares} (pending; U undoes)"
                        ));
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            }
            ClusterEvent::ReportSelection => {
                let sel = self.cluster.selection.clone();
                let title = format!("Selection ({} frames)", sel.len());
                self.start_report(ctx, title, "selection".into(), Some(sel));
                Ok(())
            }
            ClusterEvent::ReportCluster(label) => {
                let Some(ws) = &self.ws else { return };
                let frames = self.cluster.cluster_frames(ws, label);
                let title = format!("Cluster #{label} ({} frames)", frames.len());
                self.start_report(ctx, title, format!("cluster{label}"), Some(frames));
                Ok(())
            }
        };
        if let Err(e) = result {
            self.error(e);
        }
    }

    fn cluster_panel(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let missing = self.embedding_items().len();
        let busy = self.embed_task.is_some() || self.layout_task.is_some();
        let (Some(ws), Some(thumbs)) = (&self.ws, &mut self.thumbs) else { return };
        if self.cluster.dirty {
            let emb = self.embeddings.as_ref();
            self.cluster.sync(ws, |k| emb.is_some_and(|e| e.get(k).is_some()));
        }
        let setup = Setup {
            model_path: &mut self.session.embed_model_path,
            missing,
            busy,
            auto_split: &mut self.session.auto_split,
        };
        let mut events = self.cluster.setup_bar(ui, setup);
        ui.separator();
        events.extend(self.cluster.show(ui, ws, thumbs));
        for e in events {
            self.cluster_event(&ctx, e);
        }
    }

    /// Starts the overlapping-GT cleaning workflow.
    pub(crate) fn start_overlap_cleaning(&mut self) {
        let Some(ws) = &self.ws else { return };
        let groups = ws.overlap_groups();
        if groups.is_empty() {
            self.info("No overlapping GT boxes left to review");
            return;
        }
        let n = groups.len();
        self.clean = Some(CleanState { groups, index: 0, digits: String::new() });
        self.popup = false;
        self.mode = Mode::Clean;
        self.goto_group(0);
        self.info(format!("{n} groups of overlapping GT boxes to review"));
    }

    fn exit_clean(&mut self) {
        self.clean = None;
        self.set_mode(Mode::Mosaic);
    }

    fn goto_group(&mut self, index: usize) {
        let (Some(cs), Some(ws)) = (&mut self.clean, &mut self.ws) else { return };
        cs.index = index.min(cs.groups.len());
        cs.digits.clear();
        if let Some(g) = cs.groups.get(cs.index) {
            ws.go_to_frame(g.image_idx);
            ws.frame.sel = None;
            if let Some(extent) = ws.group_extent(g) {
                self.editor.focus_on(extent);
            }
        }
    }

    /// Keys of the cleaning view; `None` when the key is not one of them
    /// (box editing keys then apply as in the popup).
    fn clean_key(&mut self, key: Key, repeat: bool) -> Option<Result<(), String>> {
        let cs = self.clean.as_mut()?;
        let index = cs.index;
        if key == Key::Escape {
            self.exit_clean();
            return Some(Ok(()));
        }
        let group = cs.groups.get(index)?.clone();
        let n = group.boxes.len();
        let digit = [
            Key::Num0,
            Key::Num1,
            Key::Num2,
            Key::Num3,
            Key::Num4,
            Key::Num5,
            Key::Num6,
            Key::Num7,
            Key::Num8,
            Key::Num9,
        ]
        .iter()
        .position(|k| *k == key);
        let keep = match (digit, key) {
            (Some(d), _) if !repeat => {
                if n <= 9 {
                    Some(d)
                } else {
                    cs.digits.push(char::from(b'0' + d as u8));
                    return Some(Ok(()));
                }
            }
            (_, Key::Enter) if !cs.digits.is_empty() => {
                let d = cs.digits.parse().unwrap_or(0);
                cs.digits.clear();
                Some(d)
            }
            (_, Key::Backspace) => {
                cs.digits.pop();
                return Some(Ok(()));
            }
            _ => None,
        };
        let ws = self.ws.as_mut()?;
        let result = match (keep, key) {
            (Some(d), _) => {
                if d == 0 || d > n {
                    return Some(Err(format!("No box {d} in this group (1-{n})")));
                }
                ws.keep_only(&group, group.boxes[d - 1])
            }
            (None, Key::A) if !repeat => ws.keep_all(&group),
            (None, Key::D) if !repeat => ws.delete_all(&group),
            (None, Key::F) if !repeat => ws.toggle_frame_delete(),
            (None, Key::ArrowRight | Key::L | Key::N) => Ok(()),
            (None, Key::ArrowLeft | Key::H | Key::P) => {
                self.goto_group(index.saturating_sub(1));
                return Some(Ok(()));
            }
            _ => return None,
        };
        if result.is_ok() {
            self.goto_group(index + 1);
        }
        Some(result)
    }

    /// The cleaning view: progress, keys and the editor zoomed on the group.
    fn clean_panel(&mut self, ui: &mut egui::Ui) {
        let Some(cs) = &self.clean else { return };
        let total = cs.groups.len();
        let Some(group) = cs.groups.get(cs.index).cloned() else {
            ui.vertical_centered(|ui| {
                ui.add_space(80.0);
                ui.heading(format!("All {total} groups of overlapping GT boxes reviewed"));
                ui.label("S saves the decisions · Esc goes back to the mosaic · Left reviews the last group again");
            });
            return;
        };
        let digits = cs.digits.clone();
        let index = cs.index;
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!("Overlapping GT boxes: group {} / {total}", index + 1))
                    .strong()
                    .color(Color32::from_rgb(255, 215, 0)),
            );
            ui.separator();
            let keys = if group.boxes.len() <= 9 {
                format!("1-{}: keep that box, delete the others", group.boxes.len())
            } else {
                "type the box number + Enter: keep it, delete the others".to_string()
            };
            ui.label(
                RichText::new(format!(
                    "{keys} · A: keep all · D: delete all · F: delete frame · Right/Left: next/previous group · Esc: exit"
                ))
                .weak(),
            );
            if !digits.is_empty() {
                ui.label(
                    RichText::new(format!("keep #{digits}_")).color(Color32::from_rgb(255, 215, 0)),
                );
            }
        });
        if let (Some(ws), Some(thumbs)) = (&self.ws, &mut self.thumbs)
            && let Some(next) = self.clean.as_ref().and_then(|c| c.groups.get(c.index + 1))
        {
            let img = &ws.project.dataset.images()[next.image_idx];
            thumbs.want(vec![Job::Frame {
                path: ws.project.dataset.image_path(img),
                max_side: FRAME_MAX_SIDE,
            }]);
        }
        let numbers =
            group.boxes.iter().enumerate().map(|(i, id)| (*id, (i + 1).to_string())).collect();
        self.frame_panel(ui, true, numbers);
    }

    /// Frame header (name, tags, frame deletion, model) and the editor.
    fn frame_panel(
        &mut self,
        ui: &mut egui::Ui,
        popup: bool,
        numbers: Vec<(fastlabel_core::AnnId, String)>,
    ) {
        let ctx = ui.ctx().clone();
        let (Some(ws), Some(thumbs)) = (&mut self.ws, &mut self.thumbs) else { return };
        let n = ws.project.dataset.images().len();
        if n == 0 {
            ui.centered_and_justified(|ui| ui.label("No frames."));
            return;
        }
        let idx = ws.frame.image_idx.min(n - 1);
        ws.frame.image_idx = idx;
        let img = ws.project.dataset.images()[idx].clone();
        let job =
            Job::Frame { path: ws.project.dataset.image_path(&img), max_side: FRAME_MAX_SIDE };
        let texture = thumbs.texture(job.id());
        thumbs.want_urgent(job);
        if !popup {
            for dir in [1, -1] {
                if let Some(i) = ws.step_frame(dir, false) {
                    let path = ws.project.dataset.image_path(&ws.project.dataset.images()[i]);
                    thumbs.want(vec![Job::Frame { path, max_side: FRAME_MAX_SIDE }]);
                }
            }
        }

        let mut run_here = false;
        let mut error = None;
        let deleted = ws.is_frame_deleted(idx);
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("{} / {n}", idx + 1)).strong());
            ui.label(&img.file_name);
            if let Some(src) = img.extra.get(fastlabel_core::add::SOURCE_FIELD).and_then(|v| v.as_str()) {
                ui.label(RichText::new(src).weak()).on_hover_text("Original path (source_name)");
            }
            let saved = ws.project.dataset.split_of(idx);
            match ws.split_of(idx) {
                Some(split) if Some(split) != saved => {
                    ui.label(
                        RichText::new(format!("→ {} (on save)", split.name()))
                            .color(Color32::from_rgb(245, 150, 30)),
                    );
                }
                Some(split) => {
                    ui.label(RichText::new(split.name()).weak());
                }
                None => {}
            }
            ui.separator();
            ui.label("Tags");
            let buf = match &mut self.tags_buf {
                Some((i, text)) if *i == idx => text,
                other => {
                    *other = Some((idx, ws.tags_of(idx).join(", ")));
                    &mut other.as_mut().expect("just set").1
                }
            };
            let resp = ui.add(
                egui::TextEdit::singleline(buf).hint_text("blurry, occluded").desired_width(240.0),
            );
            if resp.lost_focus()
                && let Err(e) = ws.set_tags_text(idx, buf)
            {
                error = Some(e);
            }
            ui.separator();
            if ui.selectable_label(deleted, "F  delete frame").clicked()
                && let Err(e) = ws.toggle_frame_delete()
            {
                error = Some(e);
            }
            if ws.frame_is_empty(idx) {
                let confirmed = ws.todo_tile(idx).confirmed;
                if ui
                    .selectable_label(confirmed, "G  background")
                    .on_hover_text("Nothing to annotate on this frame (G with no box selected)")
                    .clicked()
                    && let Err(e) = ws.toggle_confirm_frame(idx)
                {
                    error = Some(e);
                }
            }
            if !popup && self.frame_from_todo {
                ui.separator();
                let (pos, total) = match &ws.frame_order {
                    Some(order) => (order.iter().position(|&i| i == idx).map_or(0, |p| p + 1), order.len()),
                    None => (0, self.todo.frames.len()),
                };
                let mut only = ws.frame_order.is_some();
                if ui
                    .checkbox(&mut only, format!("To do only ({pos} / {total})"))
                    .on_hover_text("h / l and arrows move through the to-do frames, newest first; Tab or Esc: back to the grid")
                    .changed()
                {
                    ws.frame_order = only.then(|| self.todo.frames.clone());
                }
            }
            if ui.button("Run model on this frame").clicked() {
                run_here = true;
            }
            if popup && numbers.is_empty() {
                ui.label(RichText::new("Esc / Enter / E / Q to close").weak());
            }
        });

        let boxes = ws.frame_boxes(idx);
        let preds = if numbers.is_empty() { ws.frame_preds(idx) } else { Vec::new() };
        let categories: Vec<_> =
            ws.project.dataset.categories().iter().map(|c| (c.id, c.name.clone())).collect();
        let data = FrameData {
            image_idx: idx,
            texture: texture.as_ref(),
            img_w: f64::from(img.width),
            img_h: f64::from(img.height),
            boxes: &boxes,
            preds: &preds,
            sel: ws.frame.sel,
            frame_deleted: deleted,
            interactive: true,
            categories: &categories,
            numbers: &numbers,
            min_box_height: self.session.min_box_height,
        };
        for action in self.editor.show(ui, &data) {
            let r = match action {
                EditorAction::Select(sel) => {
                    ws.frame.sel = sel;
                    Ok(())
                }
                EditorAction::SetBox(id, b) => ws.set_box(id, b),
                EditorAction::AddBox(b) => ws.add_box(b).map(|_| ()),
                // Frame mode only (the popup and the cleaning view stay put).
                EditorAction::Next if !popup => ws.next_frame(1, false),
                EditorAction::Next => Ok(()),
            };
            if let Err(e) = r {
                error = Some(e);
            }
        }
        if let Some(e) = error {
            self.error(e);
        }
        if run_here {
            self.run_model(&ctx, Scope::Frame(idx));
        }
    }

    /// Hold Space in the mosaic: the selected crop's whole frame, read-only.
    fn peek_overlay(&mut self, ctx: &egui::Context, area: egui::Rect) {
        let (Some(ws), Some(thumbs)) = (&mut self.ws, &mut self.thumbs) else { return };
        let Some(crop) = ws.selected_crop().copied() else { return };
        let img = ws.project.dataset.images()[crop.image_idx].clone();
        let job =
            Job::Frame { path: ws.project.dataset.image_path(&img), max_side: FRAME_MAX_SIDE };
        let texture = thumbs.texture(job.id());
        thumbs.want_urgent(job);
        let boxes = ws.frame_boxes(crop.image_idx);
        let preds = ws.frame_preds(crop.image_idx);
        let sel = Some(match crop.id {
            crate::crops::CropId::Gt(a) => Sel::Gt(a),
            crate::crops::CropId::Pred(p) => Sel::Pred(p),
        });
        let data = FrameData {
            image_idx: crop.image_idx,
            texture: texture.as_ref(),
            img_w: f64::from(img.width),
            img_h: f64::from(img.height),
            boxes: &boxes,
            preds: &preds,
            sel,
            frame_deleted: ws.is_frame_deleted(crop.image_idx),
            interactive: false,
            categories: &[],
            numbers: &[],
            min_box_height: self.session.min_box_height,
        };
        let peek = &mut self.peek;
        egui::Area::new(egui::Id::new("peek"))
            .order(egui::Order::Foreground)
            .fixed_pos(area.min)
            .show(ctx, |ui| {
                ui.set_min_size(area.size());
                ui.set_max_size(area.size());
                peek.show(ui, &data);
            });
    }

    /// Top row: dataset actions, the tabs, tools.
    fn toolbar(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        ui.horizontal_wrapped(|ui| {
            if ui.button("📂 Open…").on_hover_text("Open a dataset folder (Ctrl+O)").clicked() {
                self.pick_and_open(&ctx);
            }
            let has_ws = self.ws.is_some();
            let (pending, breakdown) = self.ws.as_ref().map_or((0, String::new()), pending_summary);
            let save = if pending == 0 {
                egui::Button::new("💾 Save (S)")
            } else {
                egui::Button::new(RichText::new(format!("💾 Save (S): {pending} unsaved")).color(PENDING_COLOR))
            };
            let resp = ui.add_enabled(has_ws, save);
            let resp = if pending == 0 { resp } else { resp.on_hover_text(breakdown) };
            if resp.clicked() {
                self.save();
            }
            if let Some(ws) = &mut self.ws {
                if ui
                    .add_enabled(ws.project.log.can_undo(), egui::Button::new("Undo"))
                    .on_hover_text("Undo (U, Ctrl+Z)")
                    .clicked()
                {
                    let _ = ws.undo();
                    self.mosaic.ensure_visible = true;
                }
                if ui
                    .add_enabled(ws.project.log.can_redo(), egui::Button::new("Redo"))
                    .on_hover_text("Redo (Ctrl+Shift+Z)")
                    .clicked()
                {
                    let _ = ws.redo();
                    self.mosaic.ensure_visible = true;
                }
            }
            ui.separator();
            ui.add_enabled_ui(has_ws, |ui| {
                let tabs = [
                    (Mode::Mosaic, "Mosaic", "Crops of the GT boxes and predictions (M or Tab: Frame)"),
                    (Mode::Frame, "Frame", "Frame by frame annotation (M or Tab: Mosaic)"),
                    (
                        Mode::Todo,
                        "To do (T)",
                        "Frames without any box (e.g. just added), newest first: draw boxes, or G for background",
                    ),
                    (
                        Mode::Cluster,
                        "Clusters (C)",
                        "Frames as a point cloud of image embeddings: select clusters, move them to another split",
                    ),
                    (Mode::Rules, "Rules", "The dataset's labeling rules, editable"),
                ];
                for (mode, label, hover) in tabs {
                    if ui.selectable_label(self.mode == mode, label).on_hover_text(hover).clicked()
                        && self.mode != mode
                    {
                        self.goto_tab(&ctx, mode);
                    }
                    if mode == Mode::Mosaic {
                        // One key toggles the two views.
                        ui.label(RichText::new("↔ M").weak())
                            .on_hover_text("M or Tab switches between Mosaic and Frame");
                    }
                }
            });
            ui.separator();
            ui.add_enabled_ui(has_ws && self.add_task.is_none(), |ui| {
                ui.menu_button("Add", |ui| {
                    ui.label(
                        RichText::new("Copied into images/ with unique content-hash names; duplicates are skipped")
                            .weak(),
                    );
                    if ui.button("Images…").clicked() {
                        self.start_add(&ctx, false);
                        ui.close();
                    }
                    if ui.button("Folders…").on_hover_text("All images inside, subfolders included").clicked() {
                        self.start_add(&ctx, true);
                        ui.close();
                    }
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label("Split of the added frames");
                        egui::ComboBox::from_id_salt("add-split")
                            .selected_text(self.session.add_split.map_or("none", |s| s.name()))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut self.session.add_split, None, "none");
                                for s in Split::ALL {
                                    ui.selectable_value(&mut self.session.add_split, Some(s), s.name());
                                }
                            });
                    });
                });
            });
            if ui
                .add_enabled(has_ws && self.export_task.is_none() && self.persons_task.is_none(), egui::Button::new("Export…"))
                .on_hover_text("YOLO, COCO or Ultralytics NDJSON, laid out by split, to a zip or a folder")
                .clicked()
            {
                self.export_dialog = true;
                if let Some(ws) = &self.ws {
                    let input = ws.report_input(String::new(), None, None, self.session.min_box_height);
                    let frames: Vec<_> = input.frames.iter().collect();
                    self.export_suggested_imgsz =
                        crate::report::suggested_imgsz(&frames).unwrap_or(1280);
                }
            }
            if ui
                .add_enabled(has_ws && self.report_task.is_none(), egui::Button::new("Report"))
                .on_hover_text("HTML report on all frames: split balance, box statistics, input size guide, similarity between splits (opens in the browser)")
                .clicked()
            {
                self.start_report(&ctx, "All frames".into(), "all".into(), None);
            }
            self.model_menu(ui);
            self.settings_menu(ui);
            ui.add_enabled_ui(has_ws, |ui| {
                ui.menu_button("Cleaning", |ui| {
                    if ui
                        .button("Overlapping GT boxes")
                        .on_hover_text("Review GT boxes that overlap each other (double annotations): keep one with 1-9")
                        .clicked()
                    {
                        self.start_overlap_cleaning();
                        ui.close();
                    }
                });
            });
        });
    }

    /// Second row: what the current tab shows and its filters. The split
    /// filter also scopes To do and Clusters, "No boxes" also Clusters.
    fn view_bar(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let mode = self.mode;
        let Some(ws) = &mut self.ws else { return };
        let mut want_uniformity = false;
        let mut opts = ws.opts.clone();
        let has_preds = ws.has_predictions();
        let k = ws.counts;
        ui.horizontal_wrapped(|ui| {
            if mode == Mode::Mosaic {
                let shown = ws.crops.len();
                let text = if shown == k.total {
                    format!("{shown} crops")
                } else {
                    format!("{shown} of {} crops", k.total)
                };
                ui.label(RichText::new(text).strong())
                    .on_hover_text("Crops shown / crops in the split, cluster selection and No boxes scope");
                ui.separator();
            }
            egui::ComboBox::from_id_salt("split")
                .selected_text(opts.split.label())
                .show_ui(ui, |ui| {
                    for f in crate::crops::SplitFilter::choices() {
                        ui.selectable_value(&mut opts.split, f, f.label());
                    }
                });
            if mode != Mode::Todo {
                ui.toggle_value(&mut opts.no_boxes, format!("No boxes ({})", k.no_box_frames)).on_hover_text(
                    "Frames without GT boxes: their predictions in the mosaic (whole frames: To do tab), their points in the cluster view",
                );
            }
            if let Some(frames) = &opts.frames
                && mode == Mode::Mosaic
                && ui
                    .button(
                        RichText::new(format!("Cluster selection: {} frames ✕", frames.len()))
                            .color(Color32::from_rgb(255, 215, 0)),
                    )
                    .on_hover_text("The mosaic shows only these frames; click to show all")
                    .clicked()
            {
                opts.frames = None;
            }
            if mode == Mode::Mosaic {
                ui.separator();
                ui.toggle_value(&mut opts.hide_reviewed, format!("Hide reviewed ({})", k.reviewed))
                    .on_hover_text("Crops already marked good or deleted in a saved version");
                ui.add_enabled_ui(has_preds, |ui| {
                    let (fp, fn_) = if has_preds {
                        (format!("FP only ({})", k.fp), format!("FN only ({})", k.fn_))
                    } else {
                        ("FP only".into(), "FN only".into())
                    };
                    ui.toggle_value(&mut opts.fp_only, fp)
                        .on_hover_text("FP: prediction overlapping no GT box");
                    ui.toggle_value(&mut opts.fn_only, fn_)
                        .on_hover_text("FN: GT box overlapping no prediction");
                });
                ui.separator();
                ui.label("Sort");
                egui::ComboBox::from_id_salt("sort").selected_text(opts.sort.label()).show_ui(
                    ui,
                    |ui| {
                        for key in SortKey::ALL {
                            let enabled = has_preds || !key.needs_predictions();
                            ui.add_enabled_ui(enabled, |ui| {
                                ui.selectable_value(&mut opts.sort, key, key.label())
                            });
                        }
                    },
                );
                if opts.sort == SortKey::ImageDate && ws.opts.sort != SortKey::ImageDate {
                    opts.descending = true; // newest first
                }
                let dir = if opts.descending { "Descending" } else { "Ascending" };
                if ui.button(dir).clicked() {
                    opts.descending = !opts.descending;
                }
            }
            if matches!(mode, Mode::Mosaic | Mode::Todo) {
                ui.separator();
                ui.label("Columns");
                if ui.add(egui::DragValue::new(&mut ws.columns).range(1..=20)).changed() {
                    self.mosaic.ensure_visible = true;
                    self.todo.ensure_visible = true;
                }
            }
            if mode == Mode::Mosaic
                && let Some(crop) = ws.selected_crop()
            {
                ui.separator();
                let img = &ws.project.dataset.images()[crop.image_idx];
                let b = crop.primary();
                ui.label(format!("#{} / {}", ws.selected + 1, ws.crops.len()));
                ui.label(
                    RichText::new(format!("{}  {:.0}×{:.0} px", img.file_name, b.w, b.h)).weak(),
                );
            }
        });
        if opts.sort == SortKey::Uniformity
            && ws.opts.sort != SortKey::Uniformity
            && !ws.missing_uniformity().is_empty()
        {
            // Sorted once the values are computed.
            opts.sort = ws.opts.sort;
            want_uniformity = true;
        }
        if opts != ws.opts {
            let scope_changed = opts.split != ws.opts.split || opts.no_boxes != ws.opts.no_boxes;
            ws.opts = opts;
            ws.rebuild();
            ws.select(0);
            self.mosaic.ensure_visible = true;
            if scope_changed && mode == Mode::Todo {
                self.todo.refresh(ws);
            }
        }
        if want_uniformity {
            self.start_uniformity(&ctx);
        }
    }

    /// Toolbar tab click: leaves the cleaning view first.
    fn goto_tab(&mut self, ctx: &egui::Context, mode: Mode) {
        self.popup = false;
        if self.mode == Mode::Clean {
            self.exit_clean();
        }
        if mode == Mode::Cluster {
            self.start_cluster_mode(ctx);
        } else {
            self.set_mode(mode);
        }
    }

    /// Rules tab: the dataset's labeling rules, saved when the text field
    /// loses focus, on leaving the tab and on exit.
    fn rules_view(&mut self, ui: &mut egui::Ui) {
        let Some(ws) = &mut self.ws else { return };
        ui.heading("Labeling rules");
        ui.label(RichText::new("Saved in .fastlabel/rules.md. Esc: back to the mosaic.").weak());
        let mut lost_focus = false;
        egui::ScrollArea::vertical().show(ui, |ui| {
            let edit = egui::TextEdit::multiline(&mut ws.rules)
                .hint_text("Write the labeling rules here")
                .desired_width(f32::INFINITY)
                .desired_rows(20);
            lost_focus = ui.add(edit).lost_focus();
        });
        if lost_focus && let Err(e) = ws.save_rules() {
            self.error(format!("Rules not saved: {e:#}"));
        }
    }

    fn status_line(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if let Some(ws) = &self.ws {
                let totals = ws.split_totals();
                let frames: usize = totals.iter().sum();
                ui.label(format!("{frames} frames"));
                ui.label(RichText::new(crate::cluster_view::split_shares(totals)).weak())
                    .on_hover_text("Frames per split, unsaved split moves included");
                if let Some(e) = &ws.eval {
                    ui.separator();
                    let k = e.counts;
                    ui.label(
                        RichText::new(format!(
                            "model vs GT: paired {}  FP {}  FN {}  extra {}",
                            k.matched, k.fp, k.fn_, k.extra
                        ))
                        .weak(),
                    )
                    .on_hover_text(
                        "paired: GT box with its best-overlapping prediction (IoU shown on the crop)\n\
                         FP: prediction overlapping no GT box\n\
                         FN: GT box overlapping no prediction\n\
                         extra: unpaired box overlapping an already paired one (duplicate)",
                    );
                }
                if self.thumbs_busy() > 0 {
                    ui.separator();
                    ui.spinner();
                }
            }
            if let Some((scope, done, total)) = self.inference.progress() {
                ui.separator();
                let label = match scope {
                    Scope::All | Scope::Unpredicted => format!("model {done}/{total}"),
                    Scope::Frame(_) => "model on frame…".into(),
                };
                let frac = if total == 0 { 0.0 } else { done as f32 / total as f32 };
                ui.add(egui::ProgressBar::new(frac).text(label).desired_width(180.0));
                if ui.button("Cancel").clicked() {
                    self.inference.cancel();
                }
            }
            let mut cancel = [false; 6];
            for (i, task) in [
                self.add_task.as_ref().map(|t| (t.label, t.progress())),
                self.uniformity_task.as_ref().map(|t| (t.label, t.progress())),
                self.export_task.as_ref().map(|t| (t.label, t.progress())),
                self.embed_task.as_ref().map(|t| (t.label, t.progress())),
                self.layout_task.as_ref().map(|t| (t.label, t.progress())),
                self.persons_task.as_ref().map(|(_, t)| (t.label, t.progress())),
            ]
            .into_iter()
            .enumerate()
            {
                let Some((label, (done, total))) = task else { continue };
                ui.separator();
                let frac = if total == 0 { 0.0 } else { done as f32 / total as f32 };
                ui.add(
                    egui::ProgressBar::new(frac)
                        .text(format!("{label} {done}/{total}"))
                        .desired_width(200.0),
                );
                cancel[i] = ui.button("Cancel").clicked();
            }
            if cancel[0]
                && let Some(t) = &self.add_task
            {
                t.cancel();
            }
            if cancel[1]
                && let Some(t) = &self.uniformity_task
            {
                t.cancel();
            }
            if cancel[2]
                && let Some(t) = &self.export_task
            {
                t.cancel();
            }
            if cancel[3]
                && let Some(t) = &self.embed_task
            {
                t.cancel();
            }
            if cancel[4]
                && let Some(t) = &self.layout_task
            {
                t.cancel();
            }
            if cancel[5]
                && let Some((_, t)) = &self.persons_task
            {
                t.cancel();
            }
            if let Some(s) = &self.status {
                ui.separator();
                let color =
                    if s.error { Color32::from_rgb(240, 90, 80) } else { Color32::from_gray(190) };
                ui.label(RichText::new(&s.text).color(color));
            }
        });
    }
}

impl eframe::App for FastLabelApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        if ctx.input(|i| i.viewport().close_requested()) && !self.exit_ok {
            self.inference.shutdown();
            if self.save() {
                self.exit_ok = true;
            } else {
                ctx.send_viewport_cmd(ViewportCommand::CancelClose);
            }
        }
        if self.exit_ok {
            self.save_session();
        }
        if let Some(s) = &self.status
            && !s.error
            && s.at.elapsed() > Duration::from_secs(8)
        {
            self.status = None;
        }

        if let Some(t) = &mut self.thumbs {
            t.begin_frame(&ctx);
        }
        self.poll_inference();
        self.poll_convert(&ctx);
        self.poll_tasks(&ctx);
        self.poll_cluster_tasks(&ctx);
        self.poll_add();
        self.poll_pick(&ctx);
        self.handle_keys(&ctx);

        egui::Panel::top("toolbar").show(ui, |ui| {
            self.toolbar(ui);
            #[cfg(test)]
            if let Some(id) = self.focus_probe {
                let mut text = String::new();
                ui.add(egui::TextEdit::singleline(&mut text).id(id));
            }
        });
        if self.ws.is_some() && matches!(self.mode, Mode::Mosaic | Mode::Todo | Mode::Cluster) {
            egui::Panel::top("view-bar").show(ui, |ui| self.view_bar(ui));
        }
        egui::Panel::bottom("status").show(ui, |ui| self.status_line(ui));
        let mut central = egui::Rect::NOTHING;
        egui::CentralPanel::default().show(ui, |ui| {
            central = ui.max_rect();
            if self.ws.is_none() {
                ui.centered_and_justified(|ui| {
                    if ui.button("Open a dataset folder (Ctrl+O)").clicked() {
                        self.pick_and_open(&ctx);
                    }
                });
                return;
            }
            if self.mode == Mode::Frame {
                self.frame_panel(ui, false, Vec::new());
                return;
            }
            if self.mode == Mode::Clean {
                self.clean_panel(ui);
                return;
            }
            if self.mode == Mode::Cluster {
                self.cluster_panel(ui);
                return;
            }
            if self.mode == Mode::Rules {
                self.rules_view(ui);
                return;
            }
            if self.mode == Mode::Todo {
                let (Some(ws), Some(thumbs)) = (&mut self.ws, &mut self.thumbs) else { return };
                let mut open = None;
                for ev in crate::todo::show(ui, ws, thumbs, &mut self.todo) {
                    match ev {
                        crate::todo::TodoEvent::Select(k) => self.todo.selected = k,
                        crate::todo::TodoEvent::Open(k) => open = Some(k),
                        crate::todo::TodoEvent::Refresh => self.todo.refresh(ws),
                    }
                }
                if let Some(k) = open {
                    self.open_todo_frame(k);
                }
                return;
            }
            let (Some(ws), Some(thumbs)) = (&mut self.ws, &mut self.thumbs) else { return };
            for ev in mosaic::show(ui, ws, thumbs, &mut self.mosaic, self.session.min_box_height) {
                match ev {
                    MosaicEvent::Select(i) => ws.select(i),
                    MosaicEvent::Open(i) => {
                        ws.select(i);
                        ws.focus_selected_crop();
                        self.popup = true;
                    }
                    MosaicEvent::Accept(i) => {
                        ws.select(i);
                        let keep = ws.selected;
                        if let Err(e) = ws.mark_selected(Mark::Good) {
                            self.status = Some(Status { text: e, error: true, at: Instant::now() });
                        }
                        ws.select(keep);
                    }
                    MosaicEvent::QuickDelete(i) => {
                        ws.select(i);
                        let keep = ws.selected;
                        if let Err(e) = ws.mark_selected(Mark::Delete) {
                            self.status = Some(Status { text: e, error: true, at: Instant::now() });
                        }
                        ws.select(keep);
                    }
                }
            }
        });

        if self.export_dialog {
            self.export_dialog(&ctx);
        }
        if self.popup && self.ws.is_some() {
            let size = ctx.content_rect().size() * 0.92;
            let modal = egui::Modal::new(egui::Id::new("frame-editor")).show(&ctx, |ui| {
                ui.set_min_size(size);
                ui.set_max_size(size);
                self.frame_panel(ui, true, Vec::new());
            });
            if modal.should_close() {
                self.popup = false;
            }
        } else if self.mode == Mode::Mosaic
            && !ctx.egui_wants_keyboard_input()
            && ctx.input(|i| i.key_down(Key::Space))
        {
            self.peek_overlay(&ctx, central);
        }

        if let Some(t) = &mut self.thumbs {
            t.flush();
        }
        if std::mem::take(&mut self.drop_focus)
            && let Some(id) = ctx.memory(|m| m.focused())
        {
            ctx.memory_mut(|m| m.surrender_focus(id));
        }
    }

    fn on_exit(&mut self) {
        if let Some(ws) = &mut self.ws
            && let Err(e) = ws.save_rules()
        {
            tracing::warn!("rules not saved: {e:#}");
        }
        self.inference.shutdown();
        self.save_session();
    }
}

const PENDING_COLOR: Color32 = Color32::from_rgb(245, 150, 30);

/// Unsaved changes: their number and one line per kind.
fn pending_summary(ws: &Workspace) -> (usize, String) {
    let c = ws.pending_counts();
    let kinds = [
        (c.delete, "boxes to delete (D)"),
        (c.good, "boxes or frames confirmed (G)"),
        (c.replace, "boxes replaced by their prediction (R)"),
        (c.frames, "frames to trash (F)"),
        (c.moves, "frames moved to another split"),
        (c.other, "other edits"),
    ];
    let total = kinds.iter().map(|(n, _)| n).sum();
    let lines: Vec<_> =
        kinds.iter().filter(|(n, _)| *n > 0).map(|(n, l)| format!("{n} {l}")).collect();
    (total, lines.join("\n"))
}

/// J / K in the editor: next / previous box of the frame (GT, then predictions).
fn cycle_selection(ws: &mut Workspace, dir: isize) {
    let idx = ws.frame.image_idx;
    let items: Vec<Sel> = ws
        .frame_boxes(idx)
        .iter()
        .map(|b| Sel::Gt(b.id))
        .chain(ws.frame_preds(idx).iter().map(|(p, _)| Sel::Pred(*p)))
        .collect();
    if items.is_empty() {
        return;
    }
    let n = items.len() as isize;
    let next = match ws.frame.sel.and_then(|s| items.iter().position(|x| *x == s)) {
        Some(i) => (i as isize + dir).rem_euclid(n),
        None if dir > 0 => 0,
        None => n - 1,
    };
    ws.frame.sel = Some(items[next as usize]);
}
