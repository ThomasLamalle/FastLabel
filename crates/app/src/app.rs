//! Top-level window: toolbar, status line, mosaic and keyboard shortcuts.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use eframe::egui::{self, Color32, Key, Modifiers, RichText, ViewportCommand};
use fastlabel_core::Split;

use crate::crops::SortKey;
use crate::mosaic::{self, Margins, MosaicEvent, MosaicState};
use crate::session::Session;
use crate::thumbs::Thumbs;
use crate::workspace::{Mark, Workspace};

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
        let mut app = Self {
            session,
            session_path,
            ws: None,
            thumbs: None,
            mosaic: MosaicState::default(),
            status: None,
            exit_ok: false,
            #[cfg(test)]
            focus_probe: None,
        };
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
        match Workspace::open(dir, s.view.clone(), s.thresholds, s.columns) {
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
        let start = self.session.last_dataset.clone().unwrap_or_default();
        if let Some(dir) = rfd::FileDialog::new().set_directory(start).pick_folder() {
            // Pending edits of the current dataset stay in its journal.
            self.open(ctx, &dir, false);
        }
    }

    /// Saves a new version. Returns false when saving failed.
    fn save(&mut self) -> bool {
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
                ] {
                    if n > 0 {
                        parts.push(format!("{n} {what}"));
                    }
                }
                if let Some(t) = &mut self.thumbs {
                    t.retry_failed();
                }
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
                        "{} — could not move to trash: {}",
                        parts.join(" · "),
                        files.join(", ")
                    ));
                }
                self.mosaic.ensure_visible = true;
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

    fn save_session(&mut self) {
        if let Some(ws) = &self.ws {
            self.session.selected = ws.selected;
            self.session.view = ws.opts.clone();
            self.session.thresholds = ws.thresholds;
            self.session.columns = ws.columns;
        }
        if let Some(path) = &self.session_path
            && let Err(e) = self.session.save(path)
        {
            tracing::warn!("cannot save session: {e}");
        }
    }

    fn handle_keys(&mut self, ctx: &egui::Context) {
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        let events = ctx.input(|i| i.events.clone());
        for event in events {
            let egui::Event::Key { key, pressed: true, repeat, modifiers, .. } = event else {
                continue;
            };
            self.handle_key(ctx, key, repeat, modifiers);
        }
    }

    fn handle_key(&mut self, ctx: &egui::Context, key: Key, repeat: bool, m: Modifiers) {
        let page = self.mosaic.rows_per_screen.max(1) as isize;
        let Some(ws) = &mut self.ws else {
            if key == Key::O && m.command {
                self.pick_and_open(ctx);
            }
            return;
        };
        let mut result = Ok(());
        let mut moved = true;
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
            Key::U if !m.command => result = ws.undo(),
            Key::Z if m.command && m.shift => result = ws.redo(),
            Key::Z if m.command => result = ws.undo(),
            Key::Y if m.command => result = ws.redo(),
            _ => moved = false,
        }
        if moved {
            self.mosaic.ensure_visible = true;
        }
        if let Err(e) = result {
            self.error(e);
            return;
        }
        match key {
            Key::S if !repeat && !m.command => {
                self.save();
            }
            Key::Q if !repeat && !m.command => {
                if self.save() {
                    self.exit_ok = true;
                    ctx.send_viewport_cmd(ViewportCommand::Close);
                }
            }
            Key::O if m.command => self.pick_and_open(ctx),
            _ => {}
        }
    }

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        ui.horizontal_wrapped(|ui| {
            if ui.button("📂 Open…").on_hover_text("Open a dataset folder (Ctrl+O)").clicked() {
                self.pick_and_open(&ctx);
            }
            let has_ws = self.ws.is_some();
            if ui.add_enabled(has_ws, egui::Button::new("💾 Save (S)")).clicked() {
                self.save();
            }
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
            let dir = if opts.descending { "Descending" } else { "Ascending" };
            if ui.button(dir).clicked() {
                opts.descending = !opts.descending;
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
                ui.toggle_value(&mut opts.fp_only, "FP only");
                ui.toggle_value(&mut opts.fn_only, "FN only");
            });
            if opts != ws.opts {
                ws.opts = opts;
                ws.rebuild();
                ws.select(0);
                self.mosaic.ensure_visible = true;
            }
        });
    }

    fn status_line(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if let Some(ws) = &self.ws {
                let n = ws.crops.len();
                ui.label(format!("{} / {n}", if n == 0 { 0 } else { ws.selected + 1 }));
                if let Some(crop) = ws.selected_crop() {
                    let img = &ws.project.dataset.images()[crop.image_idx];
                    let b = crop.primary();
                    ui.label(
                        RichText::new(format!("{}  {:.0}×{:.0}px", img.file_name, b.w, b.h)).weak(),
                    );
                }
                let c = ws.pending_counts();
                let pending = [
                    (c.delete, "D"),
                    (c.good, "G"),
                    (c.replace, "R"),
                    (c.frames, "F"),
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
                        RichText::new(format!("TP {}  FP {}  FN {}", k.tp, k.fp, k.fn_)).weak(),
                    );
                }
                if self.thumbs_busy() > 0 {
                    ui.separator();
                    ui.spinner();
                }
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
        egui::CentralPanel::default().show(ui, |ui| {
            let (Some(ws), Some(thumbs)) = (&mut self.ws, &mut self.thumbs) else {
                ui.centered_and_justified(|ui| {
                    if ui.button("Open a dataset folder (Ctrl+O)").clicked() {
                        self.pick_and_open(&ctx);
                    }
                });
                return;
            };
            let margins = Margins { x: self.session.margin_x, y: self.session.margin_y };
            for ev in mosaic::show(ui, ws, thumbs, &mut self.mosaic, &margins) {
                match ev {
                    MosaicEvent::Select(i) | MosaicEvent::Open(i) => ws.select(i),
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
    }

    fn on_exit(&mut self) {
        self.save_session();
    }
}
