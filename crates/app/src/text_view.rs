//! Text projects (OCR): a grid of images with their text, to accept (G) or
//! reject (D) OCR readings or mark images unreadable (X), and a one image at
//! a time view to type the texts (Enter).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use eframe::egui::{
    self, Align2, Color32, FontId, Key, Modifiers, Pos2, Rect, RichText, Sense, Stroke, StrokeKind,
    Vec2,
};
use fastlabel_core::text::{Label, Status, TextPrediction, TextProject};
use fastlabel_infer::BatchItem;
use fastlabel_media::Job;

use crate::crops::SplitFilter;
use crate::mosaic::{GT_COLOR, PRED_COLOR};
use crate::tasks::Task;
use crate::thumbs::{Slot, Thumbs};

const CAPTION_H: f32 = 40.0;
const GAP: f32 = 6.0;
const SELECT_COLOR: Color32 = Color32::from_rgb(255, 215, 0);
const PENDING_COLOR: Color32 = Color32::from_rgb(245, 150, 30);
const TEXT_EDIT: &str = "text-edit";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Filter {
    #[default]
    All,
    /// Not annotated yet.
    Todo,
    /// To do with an OCR reading to accept or reject.
    Predicted,
    /// To do without a usable reading: to type.
    ToType,
    Ok,
    Unreadable,
}

impl Filter {
    const ALL: [Filter; 6] = [
        Filter::All,
        Filter::Todo,
        Filter::Predicted,
        Filter::ToType,
        Filter::Ok,
        Filter::Unreadable,
    ];

    fn label(self) -> &'static str {
        match self {
            Filter::All => "All",
            Filter::Todo => "To do",
            Filter::Predicted => "To do, OCR read",
            Filter::ToType => "To type",
            Filter::Ok => "Done",
            Filter::Unreadable => "Unreadable",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Sort {
    #[default]
    File,
    ScoreAsc,
    ScoreDesc,
    Text,
}

impl Sort {
    const ALL: [Sort; 4] = [Sort::File, Sort::ScoreAsc, Sort::ScoreDesc, Sort::Text];

    fn label(self) -> &'static str {
        match self {
            Sort::File => "File",
            Sort::ScoreAsc => "OCR score, lowest first",
            Sort::ScoreDesc => "OCR score, highest first",
            Sort::Text => "Text",
        }
    }
}

/// Finished OCR run.
pub struct OcrDone {
    model: String,
    readings: Vec<(usize, fastlabel_infer::Reading)>,
    failed: Vec<(usize, String)>,
    cancelled: bool,
}

pub struct TextView {
    pub project: TextProject,
    /// Labeling rules (`.fastlabel/rules.md`), shown while typing.
    pub rules: String,
    /// Rows shown, in order.
    pub order: Vec<usize>,
    /// Index into `order`.
    pub selected: usize,
    pub filter: Filter,
    pub split: SplitFilter,
    pub sort: Sort,
    /// One image at a time, typing its text.
    pub typing: bool,
    /// Text being typed and the row it belongs to.
    buf: Option<(usize, String)>,
    pub columns: usize,
    ensure_visible: bool,
    offset: f32,
    view_h: f32,
    pub ocr: Option<Task<Result<OcrDone, String>>>,
    /// Ctrl+S while typing: the app saves.
    pub save_requested: bool,
}

/// Outcome of an action, for the status line.
pub type ActionResult = Result<(), String>;

impl TextView {
    pub fn open(dir: &Path, columns: usize) -> anyhow::Result<Self> {
        let project = TextProject::open(dir)?;
        let rules = std::fs::read_to_string(
            project.dir.join(fastlabel_core::project::STATE_DIR).join(crate::workspace::RULES_FILE),
        )
        .unwrap_or_default();
        let mut v = Self {
            project,
            rules,
            order: Vec::new(),
            selected: 0,
            filter: Filter::default(),
            split: SplitFilter::default(),
            sort: Sort::default(),
            typing: false,
            buf: None,
            columns: columns.max(1),
            ensure_visible: true,
            offset: 0.0,
            view_h: 0.0,
            ocr: None,
            save_requested: false,
        };
        v.refresh();
        Ok(v)
    }

    fn matches(&self, i: usize) -> bool {
        let p = &self.project;
        let (label, pred) = (p.label(i), p.prediction(i));
        let usable = pred.is_some() && !label.rejected;
        let todo = label.status == Status::Todo;
        self.split.matches(p.rows[i].split)
            && match self.filter {
                Filter::All => true,
                Filter::Todo => todo,
                Filter::Predicted => todo && usable,
                Filter::ToType => todo && !usable,
                Filter::Ok => label.status == Status::Ok,
                Filter::Unreadable => label.status == Status::Unreadable,
            }
    }

    /// Rebuilds the list (after a filter, sort or save), keeping the
    /// selected row when it is still shown. Marks do not rebuild it, so the
    /// grid stays put while reviewing.
    pub fn refresh(&mut self) {
        let keep = self.selected_row();
        let p = &self.project;
        let mut order: Vec<usize> = (0..p.rows.len()).filter(|&i| self.matches(i)).collect();
        let score = |i: usize| p.prediction(i).map_or(-1.0, |r| r.score);
        let text = |i: usize| {
            let l = p.label(i);
            if l.status == Status::Ok {
                l.text.clone()
            } else {
                p.prediction(i).map(|r| r.text.clone()).unwrap_or_default()
            }
        };
        match self.sort {
            Sort::File => {}
            Sort::ScoreAsc => order.sort_by(|&a, &b| score(a).total_cmp(&score(b))),
            Sort::ScoreDesc => order.sort_by(|&a, &b| score(b).total_cmp(&score(a))),
            Sort::Text => order.sort_by_cached_key(|&i| text(i)),
        }
        self.order = order;
        self.selected = keep
            .and_then(|r| self.order.iter().position(|&x| x == r))
            .unwrap_or(self.selected)
            .min(self.order.len().saturating_sub(1));
        self.buf = None;
        self.ensure_visible = true;
    }

    pub fn selected_row(&self) -> Option<usize> {
        self.order.get(self.selected).copied()
    }

    fn select(&mut self, k: usize) {
        if !self.order.is_empty() {
            self.selected = k.min(self.order.len() - 1);
            self.ensure_visible = true;
        }
    }

    fn move_by(&mut self, delta: isize) {
        let k = (self.selected as isize + delta).max(0) as usize;
        self.select(k);
    }

    /// Next shown row after the selection still waiting for `what`.
    fn advance(&mut self, what: impl Fn(&Self, usize) -> bool) {
        if let Some(k) = (self.selected + 1..self.order.len()).find(|&k| what(self, self.order[k]))
        {
            self.select(k);
        }
    }

    /// To do with an OCR reading not rejected: a G / D decision is waited for.
    fn undecided(&self, i: usize) -> bool {
        let l = self.project.label(i);
        l.status == Status::Todo && !l.rejected && self.project.prediction(i).is_some()
    }

    fn set(&mut self, i: usize, label: Label) -> ActionResult {
        self.project.set(i, label).map_err(|e| format!("Journal write failed: {e}"))
    }

    /// G: the OCR reading is right (an empty reading: unreadable). Again on an
    /// accepted reading: back to to do.
    pub fn accept(&mut self) -> ActionResult {
        let Some(i) = self.selected_row() else { return Ok(()) };
        let Some(pred) = self.project.prediction(i) else {
            return Err("No OCR reading here: Enter to type the text".into());
        };
        let text = self.project.config.filter(&pred.text);
        let label = self.project.label(i).clone();
        let accepted = if text.is_empty() {
            Label { status: Status::Unreadable, ..Default::default() }
        } else {
            Label::ok(text)
        };
        let next = if label.status == accepted.status && label.text == accepted.text {
            Label::default()
        } else {
            accepted
        };
        self.set(i, next)?;
        self.advance(Self::undecided);
        Ok(())
    }

    /// D: the OCR reading is wrong (the image goes to "To type"); toggles.
    pub fn reject(&mut self) -> ActionResult {
        let Some(i) = self.selected_row() else { return Ok(()) };
        if self.project.prediction(i).is_none() {
            return Err("No OCR reading to reject here".into());
        }
        let rejected = !self.project.label(i).rejected;
        self.set(i, Label { status: Status::Todo, text: String::new(), rejected })?;
        self.advance(Self::undecided);
        Ok(())
    }

    /// X: nothing readable; toggles.
    pub fn unreadable(&mut self) -> ActionResult {
        let Some(i) = self.selected_row() else { return Ok(()) };
        let l = self.project.label(i).clone();
        let status = if l.status == Status::Unreadable { Status::Todo } else { Status::Unreadable };
        self.set(i, Label { status, text: String::new(), rejected: l.rejected })?;
        self.advance(Self::undecided);
        Ok(())
    }

    pub fn undo(&mut self, redo: bool) -> ActionResult {
        let p = &mut self.project;
        let row = if redo { p.redo() } else { p.undo() }.map_err(|e| e.to_string())?;
        if let Some(k) = row.and_then(|r| self.order.iter().position(|&x| x == r)) {
            self.select(k);
        }
        self.buf = None;
        Ok(())
    }

    pub fn start_typing(&mut self) {
        if !self.order.is_empty() {
            self.typing = true;
            self.buf = None;
        }
    }

    /// The text field's starting value: the text, else the OCR reading.
    fn initial_text(&self, i: usize) -> String {
        let l = self.project.label(i);
        match (l.status, self.project.prediction(i)) {
            (Status::Ok, _) => l.text.clone(),
            (Status::Todo, Some(p)) if !l.rejected => self.project.config.filter(&p.text),
            _ => String::new(),
        }
    }

    /// Enter while typing: the text (empty: unreadable), then the next image to do.
    fn commit(&mut self, i: usize, text: &str) -> ActionResult {
        let text = self.project.config.filter(text.trim());
        let rejected = self.project.label(i).rejected;
        let label = if text.is_empty() {
            Label { status: Status::Unreadable, text, rejected }
        } else {
            Label { status: Status::Ok, text, rejected }
        };
        self.set(i, label)?;
        self.buf = None;
        self.advance(|v, i| v.project.label(i).status == Status::Todo);
        Ok(())
    }

    /// Grid keys; `None` when the key is not a text view key.
    pub fn key(&mut self, key: Key, m: Modifiers) -> Option<ActionResult> {
        let cols = self.columns as isize;
        let plain = !m.command && !m.alt;
        let step = match key {
            Key::ArrowLeft | Key::H if plain => Some(-1),
            Key::ArrowRight | Key::L if plain => Some(1),
            Key::ArrowUp | Key::K if plain => Some(-cols),
            Key::ArrowDown | Key::J if plain => Some(cols),
            Key::PageUp => Some(-cols * 4),
            Key::PageDown => Some(cols * 4),
            Key::Home => Some(isize::MIN / 2),
            Key::End => Some(isize::MAX / 2),
            _ => None,
        };
        if let Some(step) = step {
            self.move_by(step);
            return Some(Ok(()));
        }
        Some(match key {
            Key::G if plain => self.accept(),
            Key::D if plain => self.reject(),
            Key::X if plain => self.unreadable(),
            Key::Enter | Key::E if plain => {
                self.start_typing();
                Ok(())
            }
            Key::U if plain => self.undo(false),
            Key::Z if m.command => self.undo(m.shift),
            Key::Y if m.command => self.undo(true),
            _ => return None,
        })
    }

    /// Starts reading the to-do images with an OCR model in the background.
    pub fn start_ocr(&mut self, ctx: &egui::Context, model: PathBuf) {
        if self.ocr.is_some() {
            return;
        }
        let p = &self.project;
        let items: Vec<BatchItem> = (0..p.rows.len())
            .filter(|&i| p.label(i).status == Status::Todo)
            .map(|i| BatchItem { key: i, path: p.dir.join(&p.rows[i].file) })
            .collect();
        let charset = p.config.charset.clone();
        self.ocr = Some(Task::spawn("OCR", ctx, move |progress| {
            let mut reader = fastlabel_infer::TextReader::load(&model, 0)
                .map_err(|e| format!("{}: {e}", model.display()))?;
            let r = fastlabel_infer::run_ocr(&mut reader, items, &charset, progress);
            let name = model.file_name().map(|n| n.to_string_lossy().into_owned());
            Ok(OcrDone {
                model: name.unwrap_or_default(),
                readings: r.readings,
                failed: r.failed,
                cancelled: r.cancelled,
            })
        }));
    }

    /// Stores a finished OCR run; the status line message.
    pub fn poll_ocr(&mut self) -> Option<Result<String, String>> {
        let done = self.ocr.as_mut()?.poll()?;
        self.ocr = None;
        let done = match done {
            Ok(d) => d,
            Err(e) => return Some(Err(format!("OCR failed: {e}"))),
        };
        let p = &mut self.project;
        let texts: BTreeMap<String, TextPrediction> = done
            .readings
            .iter()
            .map(|(i, r)| {
                let pred = TextPrediction { text: r.text.clone(), score: r.score, rejected: false };
                (p.rows[*i].file.clone(), pred)
            })
            .collect();
        let n = texts.len();
        // Readings done before a cancel are kept: each one stands alone.
        if let Err(e) = p.merge_predictions(done.model, texts) {
            return Some(Err(format!("OCR readings not stored: {e}")));
        }
        self.filter = Filter::Predicted;
        self.sort = Sort::ScoreAsc;
        self.refresh();
        self.select(0);
        let mut msg =
            format!("OCR read {n} images — G: right, D: wrong, X: unreadable, lowest scores first");
        if done.cancelled {
            msg.push_str(" (cancelled)");
        }
        Some(match done.failed.first() {
            None => Ok(msg),
            Some((i, why)) => Err(format!(
                "{msg}; {} failed ({}: {why})",
                done.failed.len(),
                self.project.rows[*i].file
            )),
        })
    }

    /// Filters, sort and counts.
    pub fn view_bar(&mut self, ui: &mut egui::Ui) {
        let counts = self.project.counts();
        let n = |s| counts.get(&s).copied().unwrap_or(0);
        let (mut filter, mut split, mut sort) = (self.filter, self.split, self.sort);
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(format!("{} of {} images", self.order.len(), self.project.rows.len())).strong());
            ui.label(
                RichText::new(format!(
                    "to do {} · done {} · unreadable {}",
                    n(Status::Todo),
                    n(Status::Ok),
                    n(Status::Unreadable)
                ))
                .weak(),
            );
            ui.separator();
            egui::ComboBox::from_id_salt("text-filter").selected_text(filter.label()).show_ui(
                ui,
                |ui| {
                    for f in Filter::ALL {
                        ui.selectable_value(&mut filter, f, f.label());
                    }
                },
            );
            egui::ComboBox::from_id_salt("text-split").selected_text(split.label()).show_ui(
                ui,
                |ui| {
                    for f in SplitFilter::choices() {
                        ui.selectable_value(&mut split, f, f.label());
                    }
                },
            );
            ui.label("Sort");
            egui::ComboBox::from_id_salt("text-sort").selected_text(sort.label()).show_ui(
                ui,
                |ui| {
                    for s in Sort::ALL {
                        ui.selectable_value(&mut sort, s, s.label());
                    }
                },
            );
            ui.add(egui::Slider::new(&mut self.columns, 2..=16).text("per line"));
            ui.separator();
            if self.typing {
                ui.label(
                    RichText::new("Enter: save the text (empty: unreadable), next · Up/Down: previous/next · Ctrl+S: save · Esc: grid")
                        .weak(),
                );
            } else {
                ui.label(
                    RichText::new(
                        "G: OCR right · D: OCR wrong · X: unreadable · Enter/E: type · U: undo · S: save",
                    )
                    .weak(),
                );
            }
        });
        if (filter, split, sort) != (self.filter, self.split, self.sort) {
            (self.filter, self.split, self.sort) = (filter, split, sort);
            self.refresh();
        }
    }

    pub fn show(&mut self, ui: &mut egui::Ui, thumbs: &mut Thumbs) -> ActionResult {
        if self.order.is_empty() {
            self.typing = false;
            ui.centered_and_justified(|ui| ui.label("No image here (check the filter)."));
            return Ok(());
        }
        if self.typing {
            return self.show_single(ui, thumbs);
        }
        self.show_grid(ui, thumbs);
        Ok(())
    }

    fn show_grid(&mut self, ui: &mut egui::Ui, thumbs: &mut Thumbs) {
        let cols = self.columns.max(1);
        let n = self.order.len();
        let avail_w = ui.available_width();
        let cell_w = ((avail_w - GAP * (cols - 1) as f32) / cols as f32).max(60.0);
        let img_h = cell_w * 0.6;
        let row_h = img_h + CAPTION_H;
        let pitch = row_h + ui.spacing().item_spacing.y;
        let mut area = egui::ScrollArea::vertical().auto_shrink(false);
        if self.ensure_visible && self.view_h > 0.0 {
            self.ensure_visible = false;
            let top = (self.selected / cols) as f32 * pitch;
            let mut offset = self.offset;
            if top < offset {
                offset = top;
            } else if top + row_h > offset + self.view_h {
                offset = top + row_h - self.view_h;
            }
            area = area.vertical_scroll_offset(offset);
        }
        let (mut select, mut open) = (None, None);
        let out = area.show_rows(ui, row_h, n.div_ceil(cols), |ui, range| {
            let mut jobs = Vec::new();
            for row in range {
                let (rect, _) = ui.allocate_exact_size(Vec2::new(avail_w, row_h), Sense::hover());
                for c in 0..cols {
                    let k = row * cols + c;
                    let Some(&i) = self.order.get(k) else { break };
                    let cell = Rect::from_min_size(
                        rect.min + Vec2::new(c as f32 * (cell_w + GAP), 0.0),
                        Vec2::new(cell_w, row_h),
                    );
                    let resp = ui.interact(cell, ui.id().with(("text", i)), Sense::click());
                    if resp.double_clicked() {
                        open = Some(k);
                    } else if resp.clicked() {
                        select = Some(k);
                    }
                    let job = self.job(i, 256);
                    self.paint_cell(
                        ui.painter(),
                        cell,
                        img_h,
                        i,
                        thumbs.get(job.id()),
                        k == self.selected,
                    );
                    jobs.push(job);
                }
            }
            thumbs.want(jobs);
        });
        self.offset = out.state.offset.y;
        self.view_h = out.inner_rect.height();
        if let Some(k) = select.or(open) {
            self.select(k);
        }
        if open.is_some() {
            self.start_typing();
        }
    }

    fn job(&self, i: usize, max_side: u32) -> Job {
        Job::Frame { path: self.project.dir.join(&self.project.rows[i].file), max_side }
    }

    /// (text, color, OCR score) shown for row `i`.
    fn caption(&self, i: usize) -> (String, Color32, Option<f32>) {
        let l = self.project.label(i);
        let pred = self.project.prediction(i);
        match l.status {
            Status::Ok => (l.text.clone(), GT_COLOR, None),
            Status::Unreadable => ("unreadable".into(), Color32::from_gray(150), None),
            Status::Todo => match pred {
                Some(p) if l.rejected => {
                    (format!("{} (rejected)", p.text), Color32::from_gray(110), None)
                }
                Some(p) if p.text.is_empty() => ("OCR: nothing".into(), PRED_COLOR, Some(p.score)),
                Some(p) => (p.text.clone(), PRED_COLOR, Some(p.score)),
                None => ("—".into(), Color32::from_gray(110), None),
            },
        }
    }

    fn paint_cell(
        &self,
        painter: &egui::Painter,
        cell: Rect,
        img_h: f32,
        i: usize,
        slot: Option<&Slot>,
        selected: bool,
    ) {
        let painter = &painter.with_clip_rect(cell.expand(4.0));
        let img_area = Rect::from_min_size(cell.min, Vec2::new(cell.width(), img_h));
        painter.rect_filled(img_area, 2.0, Color32::from_gray(28));
        paint_image(painter, img_area, slot);
        let (text, color, score) = self.caption(i);
        let caption = Pos2::new(cell.left() + 2.0, img_area.bottom() + 2.0);
        painter.text(caption, Align2::LEFT_TOP, &text, FontId::proportional(18.0), color);
        if let Some(s) = score {
            painter.text(
                Pos2::new(cell.right() - 2.0, img_area.bottom() + 4.0),
                Align2::RIGHT_TOP,
                format!("{s:.2}"),
                FontId::proportional(12.0),
                Color32::from_gray(160),
            );
        }
        let row = &self.project.rows[i];
        let name = row.file.rsplit('/').next().unwrap_or(&row.file);
        painter.text(
            caption + Vec2::new(0.0, 22.0),
            Align2::LEFT_TOP,
            format!("{} · {name}", row.split.map_or("no split", |s| s.name())),
            FontId::proportional(11.0),
            Color32::from_gray(140),
        );
        if self.project.is_pending(i) {
            let l = self.project.label(i);
            let mark = match l.status {
                Status::Ok => ("G", GT_COLOR),
                Status::Unreadable => ("X", Color32::from_gray(190)),
                Status::Todo if l.rejected => ("D", PRED_COLOR),
                Status::Todo => ("•", PENDING_COLOR),
            };
            painter.text(
                img_area.right_top() + Vec2::new(-6.0, 4.0),
                Align2::RIGHT_TOP,
                mark.0,
                FontId::proportional(20.0),
                mark.1,
            );
        }
        if selected {
            painter.rect_stroke(img_area, 3.0, Stroke::new(2.5, SELECT_COLOR), StrokeKind::Inside);
        }
    }

    /// One image, large, with the text field.
    fn show_single(&mut self, ui: &mut egui::Ui, thumbs: &mut Thumbs) -> ActionResult {
        let Some(i) = self.selected_row() else { return Ok(()) };
        let id = egui::Id::new(TEXT_EDIT);
        let mut result = Ok(());
        // Keys taken before the text field sees them.
        let (enter, esc, prev, next, undo, save) = ui.input_mut(|inp| {
            let mut take = |m, k| inp.consume_key(m, k);
            (
                take(Modifiers::NONE, Key::Enter),
                take(Modifiers::NONE, Key::Escape),
                take(Modifiers::NONE, Key::ArrowUp) || take(Modifiers::NONE, Key::PageUp),
                take(Modifiers::NONE, Key::ArrowDown) || take(Modifiers::NONE, Key::PageDown),
                take(Modifiers::COMMAND, Key::Z),
                take(Modifiers::COMMAND, Key::S),
            )
        });
        if esc {
            self.typing = false;
            self.buf = None;
            ui.ctx().memory_mut(|m| m.surrender_focus(id));
            return Ok(());
        }
        if undo {
            result = self.undo(false);
        }
        if enter {
            let text = self.buf.as_ref().filter(|(r, _)| *r == i).map(|(_, t)| t.clone());
            let text = text.unwrap_or_else(|| self.initial_text(i));
            result = self.commit(i, &text);
        } else if prev || next {
            self.move_by(if prev { -1 } else { 1 });
            self.buf = None;
        }
        let Some(i) = self.selected_row() else { return result };
        if self.buf.as_ref().is_none_or(|(r, _)| *r != i) {
            let text = self.initial_text(i);
            // Selected, so typing replaces the OCR reading.
            let mut state = egui::TextEdit::load_state(ui.ctx(), id).unwrap_or_default();
            let all = egui::text::CCursorRange::two(
                egui::text::CCursor::new(0),
                egui::text::CCursor::new(text.chars().count()),
            );
            state.cursor.set_char_range(Some(all));
            state.store(ui.ctx(), id);
            self.buf = Some((i, text));
        }

        let avail = ui.available_size();
        let img_h = (avail.y - 190.0).max(80.0);
        let (img_rect, _) = ui.allocate_exact_size(Vec2::new(avail.x, img_h), Sense::hover());
        ui.painter().rect_filled(img_rect, 4.0, Color32::from_gray(20));
        let job = self.job(i, 1024);
        let slot = thumbs.get(job.id());
        paint_image(ui.painter(), img_rect.shrink(8.0), slot);
        thumbs.want_urgent(job);
        // Prefetch the neighbours.
        let around = (self.selected.saturating_sub(2)..self.selected + 4)
            .filter_map(|k| self.order.get(k))
            .map(|&r| self.job(r, 1024))
            .collect();
        thumbs.want(around);

        ui.add_space(8.0);
        let row = &self.project.rows[i];
        let label = self.project.label(i).clone();
        let pred = self.project.prediction(i).cloned();
        let column = self.project.config.text_column.clone();
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("{column}:")).size(22.0));
            let (_, buf) = self.buf.as_mut().expect("set above");
            let resp = ui.add(
                egui::TextEdit::singleline(buf)
                    .id(id)
                    .font(FontId::monospace(26.0))
                    .desired_width(320.0),
            );
            if resp.changed() {
                *buf = self.project.config.filter(buf);
            }
            if !resp.has_focus() {
                resp.request_focus();
            }
            let status = match label.status {
                Status::Ok => RichText::new(format!("saved as \"{}\"", label.text)).color(GT_COLOR),
                Status::Unreadable => RichText::new("unreadable").color(Color32::from_gray(170)),
                Status::Todo => RichText::new("to do").color(PENDING_COLOR),
            };
            ui.label(status.size(16.0));
        });
        ui.horizontal(|ui| {
            let ocr = match &pred {
                Some(p) if label.rejected => format!("OCR: \"{}\" rejected", p.text),
                Some(p) if p.text.is_empty() => "OCR: nothing read (G: unreadable)".into(),
                Some(p) => format!("OCR: \"{}\" (score {:.2})", p.text, p.score),
                None => "OCR: no reading".into(),
            };
            ui.label(RichText::new(ocr).color(PRED_COLOR));
            ui.separator();
            ui.label(format!(
                "{} / {} · {} · {}",
                self.selected + 1,
                self.order.len(),
                row.split.map_or("no split", |s| s.name()),
                row.file
            ));
            if !self.project.config.charset.is_empty() {
                ui.separator();
                ui.label(RichText::new(format!("allowed: {}", self.project.config.charset)).weak());
            }
        });
        if !self.rules.trim().is_empty() {
            ui.add_space(4.0);
            ui.label(RichText::new(self.rules.trim()).weak());
        }
        self.save_requested |= save;
        result
    }
}

fn paint_image(painter: &egui::Painter, area: Rect, slot: Option<&Slot>) {
    match slot {
        Some(Slot::Ready(tex)) => {
            let size = tex.size_vec2();
            let s = (area.width() / size.x).min(area.height() / size.y);
            let r = Rect::from_center_size(area.center(), size * s);
            let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
            painter.image(tex.id(), r, uv, Color32::WHITE);
        }
        Some(Slot::Failed(e)) => {
            painter.text(
                area.center(),
                Align2::CENTER_CENTER,
                e,
                FontId::proportional(11.0),
                PRED_COLOR,
            );
        }
        None => {}
    }
}
