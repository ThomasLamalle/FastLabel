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
use crate::mosaic::{self, Margins, MosaicEvent, MosaicState};
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
    export_task: Option<Task<Result<fastlabel_core::export::ExportReport, String>>>,
    pub(crate) cluster: ClusterView,
    /// Loaded when cluster mode is first opened.
    embeddings: Option<Embeddings>,
    embed_task: Option<Task<EmbedOutcome>>,
    layout_task: Option<Task<Option<SavedLayout>>>,
    report_task: Option<Task<Result<PathBuf, String>>>,
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
            cluster: ClusterView::default(),
            embeddings: None,
            embed_task: None,
            layout_task: None,
            report_task: None,
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
                self.ws = Some(ws);
                self.cluster = ClusterView::default();
                self.cluster.min_cluster_size = self.session.min_cluster_size;
                self.embeddings = None;
                self.embed_task = None;
                self.layout_task = None;
                if self.mode == Mode::Cluster {
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
        if let Some(dir) = rfd::FileDialog::new().set_directory(start).pick_folder() {
            // Pending edits of the current dataset stay in its journal.
            self.open(ctx, &dir, false);
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
                    (s.moves_skipped, "split moves skipped (no split folder or file exists)"),
                ] {
                    if n > 0 {
                        parts.push(format!("{n} {what}"));
                    }
                }
                if let Some(t) = &mut self.thumbs {
                    t.retry_failed();
                }
                self.cluster.dirty = true;
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
        let Some(ws) = &self.ws else { return };
        let Some(model) = self.session.model_path.clone().filter(|p| p.is_file()) else {
            self.error("Choose an ONNX model file first (Model menu)");
            return;
        };
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

    fn start_export(&mut self, ctx: &egui::Context) {
        if self.export_task.is_some() || !self.save() {
            return;
        }
        let Some(ws) = &self.ws else { return };
        let p = &ws.project;
        let Some(source) = &p.source else { return };
        let name = source.file_name();
        let dir_name =
            p.dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let stem = name.trim_start_matches("annotations.").trim_end_matches(".json");
        let Some(out) = rfd::FileDialog::new()
            .set_directory(p.dir.parent().unwrap_or(&p.dir))
            .set_file_name(format!("{dir_name}_{stem}.zip"))
            .add_filter("zip", &["zip"])
            .save_file()
        else {
            return;
        };
        let (root, coco) = (p.dir.clone(), p.dataset.coco.clone());
        self.export_task = Some(Task::spawn("export", ctx, move |progress| {
            use std::sync::atomic::Ordering;
            fastlabel_core::export::export_zip(&root, &coco, &name, &out, |done, total| {
                progress.done.store(done, Ordering::Relaxed);
                progress.total.store(total, Ordering::Relaxed);
                !progress.cancel.load(Ordering::Relaxed)
            })
            .map(|mut r| {
                r.missing.truncate(5);
                r
            })
            .map_err(|e| format!("{e} ({})", out.display()))
        }));
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
        if let Some(result) = self.export_task.as_mut().and_then(Task::poll) {
            self.export_task = None;
            match result {
                Ok(r) if r.cancelled => self.info("Export cancelled"),
                Ok(r) if r.missing.is_empty() => self.info(format!("Exported {} images", r.images)),
                Ok(r) => self.error(format!(
                    "Exported {} images; missing: {}",
                    r.images,
                    r.missing.join(", ")
                )),
                Err(e) => self.error(format!("Export failed: {e}")),
            }
        }
        let _ = ctx;
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
                            .hint_text("model.onnx")
                            .desired_width(280.0),
                    )
                    .changed()
                {
                    self.session.model_path = (!path.is_empty()).then(|| PathBuf::from(&path));
                }
                if ui.button("Browse…").clicked()
                    && let Some(p) =
                        rfd::FileDialog::new().add_filter("ONNX model", &["onnx"]).pick_file()
                {
                    self.session.model_path = Some(p);
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
        match mode {
            Mode::Frame => ws.focus_selected_crop(),
            Mode::Mosaic => {
                if self.mode == Mode::Cluster {
                    // Split moves change what the split filter shows.
                    ws.rebuild();
                }
                ws.select_crop_of_frame();
                self.mosaic.ensure_visible = true;
            }
            Mode::Clean | Mode::Cluster => {}
        }
        self.mode = mode;
    }

    fn handle_key(&mut self, ctx: &egui::Context, key: Key, repeat: bool, m: Modifiers) {
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
                } else if self.mode == Mode::Cluster {
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
                if let Some(i) = ws.step_frame(dir, m.command) {
                    ws.go_to_frame(i);
                }
                Ok(())
            }
            Key::Home | Key::End if !popup => {
                let last = ws.project.dataset.images().len() - 1;
                ws.go_to_frame(if key == Key::Home { 0 } else { last });
                Ok(())
            }
            Key::J | Key::K | Key::ArrowDown | Key::ArrowUp => {
                cycle_selection(ws, if matches!(key, Key::K | Key::ArrowUp) { -1 } else { 1 });
                Ok(())
            }
            Key::D if !repeat => ws.mark_sel(Mark::Delete),
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
        let input = ws.report_input(title, scope.as_ref(), layout);
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
            ClusterEvent::Exit => {
                self.set_mode(Mode::Mosaic);
                Ok(())
            }
            ClusterEvent::ReportAll => {
                self.start_report(ctx, "All frames".into(), "all".into(), None);
                Ok(())
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
        let setup = Setup { model_path: &mut self.session.embed_model_path, missing, busy };
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
        ui.horizontal(|ui| {
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
        };
        for action in self.editor.show(ui, &data) {
            let r = match action {
                EditorAction::Select(sel) => {
                    ws.frame.sel = sel;
                    Ok(())
                }
                EditorAction::SetBox(id, b) => ws.set_box(id, b),
                EditorAction::AddBox(b) => ws.add_box(b).map(|_| ()),
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

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let mut want_uniformity = false;
        ui.horizontal_wrapped(|ui| {
            if ui.button("📂 Open…").on_hover_text("Open a dataset folder (Ctrl+O)").clicked() {
                self.pick_and_open(&ctx);
            }
            let has_ws = self.ws.is_some();
            if ui.add_enabled(has_ws, egui::Button::new("💾 Save (S)")).clicked() {
                self.save();
            }
            if ui
                .add_enabled(has_ws && self.export_task.is_none(), egui::Button::new("Export zip…"))
                .on_hover_text("Saves pending edits, then zips the annotation file and images")
                .clicked()
            {
                self.start_export(&ctx);
            }
            self.model_menu(ui);
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
                if ui
                    .selectable_label(self.mode == Mode::Cluster, "Clusters (C)")
                    .on_hover_text("Frames as a point cloud of image embeddings: select clusters, move them to another split")
                    .clicked()
                {
                    if self.mode == Mode::Cluster {
                        self.set_mode(Mode::Mosaic);
                    } else {
                        self.start_cluster_mode(&ctx);
                    }
                }
                if ui
                    .add_enabled(self.report_task.is_none(), egui::Button::new("Report"))
                    .on_hover_text("HTML report on all frames: split balance, box statistics, input size guide, similarity between splits (opens in the browser)")
                    .clicked()
                {
                    self.start_report(&ctx, "All frames".into(), "all".into(), None);
                }
            });
            let Some(ws) = &mut self.ws else { return };
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
            ui.separator();

            ui.label("Columns");
            if ui.add(egui::DragValue::new(&mut ws.columns).range(1..=20)).changed() {
                self.mosaic.ensure_visible = true;
            }
            ui.label("Margin W");
            ui.add(
                egui::DragValue::new(&mut self.session.margin_x)
                    .range(0.0..=4096.0)
                    .speed(1.0)
                    .suffix(" px"),
            );
            ui.label("H");
            ui.add(
                egui::DragValue::new(&mut self.session.margin_y)
                    .range(0.0..=4096.0)
                    .speed(1.0)
                    .suffix(" px"),
            );
            ui.separator();

            let mut opts = ws.opts.clone();
            let has_preds = ws.has_predictions();
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
            if let Some(frames) = &opts.frames
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
            egui::ComboBox::from_id_salt("split")
                .selected_text(opts.split.map_or("All splits", |s| s.name()))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut opts.split, None, "All splits");
                    for s in Split::ALL {
                        ui.selectable_value(&mut opts.split, Some(s), s.name());
                    }
                });
            ui.toggle_value(&mut opts.hide_reviewed, "Hide reviewed");
            ui.add_enabled_ui(has_preds, |ui| {
                ui.horizontal(|ui| {
                    ui.toggle_value(&mut opts.fp_only, "FP only");
                    ui.toggle_value(&mut opts.fn_only, "FN only")
                        .on_hover_text("FP: prediction overlapping no GT box. FN: GT box overlapping no prediction.");
                })
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
                ws.opts = opts;
                ws.rebuild();
                ws.select(0);
                self.mosaic.ensure_visible = true;
            }
        });
        if want_uniformity {
            self.start_uniformity(&ctx);
        }
    }

    fn status_line(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if self.mode == Mode::Clean
                && let Some(cs) = &self.clean
            {
                let total = cs.groups.len();
                ui.label(
                    RichText::new(format!("Overlapping GT boxes {} / {total}", (cs.index + 1).min(total)))
                        .strong()
                        .color(Color32::from_rgb(255, 215, 0)),
                );
                ui.separator();
            }
            if let Some(ws) = &self.ws {
                let n = ws.crops.len();
                let clean = matches!(self.mode, Mode::Clean | Mode::Cluster);
                if !clean {
                ui.label(format!("{} / {n}", if n == 0 { 0 } else { ws.selected + 1 }));
                if let Some(crop) = ws.selected_crop() {
                    let img = &ws.project.dataset.images()[crop.image_idx];
                    let b = crop.primary();
                    ui.label(
                        RichText::new(format!("{}  {:.0}×{:.0}px", img.file_name, b.w, b.h)).weak(),
                    );
                }
                }
                ui.label(
                    RichText::new(crate::cluster_view::split_shares(ws.split_totals())).weak(),
                )
                .on_hover_text("Frames per split, unsaved split moves included");
                let c = ws.pending_counts();
                let pending = [
                    (c.delete, "D"),
                    (c.good, "G"),
                    (c.replace, "R"),
                    (c.frames, "F"),
                    (c.moves, "split moves"),
                    (c.other, "edits"),
                ]
                .iter()
                .filter(|(n, _)| *n > 0)
                .map(|(n, l)| format!("{l} {n}"))
                .collect::<Vec<_>>();
                if !pending.is_empty() {
                    ui.separator();
                    ui.label(
                        RichText::new(format!("pending: {}", pending.join("  ")))
                            .color(Color32::from_rgb(245, 150, 30)),
                    );
                }
                if let Some(e) = &ws.eval {
                    ui.separator();
                    let k = e.counts;
                    ui.label(
                        RichText::new(format!("paired {}  FP {}  FN {}  extra {}", k.matched, k.fp, k.fn_, k.extra))
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
            let mut cancel = [false; 4];
            for (i, task) in [
                self.uniformity_task.as_ref().map(|t| (t.label, t.progress())),
                self.export_task.as_ref().map(|t| (t.label, t.progress())),
                self.embed_task.as_ref().map(|t| (t.label, t.progress())),
                self.layout_task.as_ref().map(|t| (t.label, t.progress())),
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
                && let Some(t) = &self.uniformity_task
            {
                t.cancel();
            }
            if cancel[1]
                && let Some(t) = &self.export_task
            {
                t.cancel();
            }
            if cancel[2]
                && let Some(t) = &self.embed_task
            {
                t.cancel();
            }
            if cancel[3]
                && let Some(t) = &self.layout_task
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
        self.poll_tasks(&ctx);
        self.poll_cluster_tasks(&ctx);
        self.handle_keys(&ctx);

        egui::Panel::top("toolbar").show(ui, |ui| {
            self.toolbar(ui);
            #[cfg(test)]
            if let Some(id) = self.focus_probe {
                let mut text = String::new();
                ui.add(egui::TextEdit::singleline(&mut text).id(id));
            }
        });
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
            let (Some(ws), Some(thumbs)) = (&mut self.ws, &mut self.thumbs) else { return };
            let margins = Margins { x: self.session.margin_x, y: self.session.margin_y };
            for ev in mosaic::show(ui, ws, thumbs, &mut self.mosaic, &margins) {
                match ev {
                    MosaicEvent::Select(i) => ws.select(i),
                    MosaicEvent::Open(i) => {
                        ws.select(i);
                        ws.focus_selected_crop();
                        self.popup = true;
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
        self.inference.shutdown();
        self.save_session();
    }
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
