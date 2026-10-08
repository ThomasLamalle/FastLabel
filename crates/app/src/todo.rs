//! To do: frames without any GT box (typically just added), as a grid of
//! whole-frame thumbnails, newest first. Enter opens one to draw boxes, G
//! confirms it as background, F deletes it.

use eframe::egui::{
    self, Align2, Color32, FontId, Pos2, Rect, RichText, Sense, Stroke, StrokeKind, Vec2,
};
use fastlabel_media::{Job, Region};

use crate::mosaic::{GT_COLOR, PRED_COLOR};
use crate::thumbs::{Slot, Thumbs};
use crate::workspace::Workspace;

const CAPTION_H: f32 = 34.0;
const GAP: f32 = 6.0;
const SELECT_COLOR: Color32 = Color32::from_rgb(255, 215, 0);
const DRAWN_COLOR: Color32 = Color32::from_rgb(245, 150, 30);

#[derive(Default)]
pub struct TodoState {
    /// Image indices, newest first.
    pub frames: Vec<usize>,
    pub selected: usize,
    /// Also list the frames saved as background.
    pub show_background: bool,
    pub ensure_visible: bool,
    offset: f32,
    view_h: f32,
}

pub enum TodoEvent {
    Select(usize),
    Open(usize),
    /// The list changed (Show background toggled).
    Refresh,
}

impl TodoState {
    /// Rebuilds the list, keeping the selected frame when it is still there.
    pub fn refresh(&mut self, ws: &mut Workspace) {
        let keep = self.frames.get(self.selected).copied();
        self.frames = ws.todo_frames(self.show_background);
        self.selected = keep
            .and_then(|f| self.frames.iter().position(|&x| x == f))
            .unwrap_or(self.selected)
            .min(self.frames.len().saturating_sub(1));
        self.ensure_visible = true;
    }

    pub fn selected_frame(&self) -> Option<usize> {
        self.frames.get(self.selected).copied()
    }

    pub fn move_selection(&mut self, delta: isize) {
        if self.frames.is_empty() {
            return;
        }
        let last = self.frames.len() as isize - 1;
        self.selected = (self.selected as isize + delta).clamp(0, last) as usize;
        self.ensure_visible = true;
    }

    /// Next frame after the selection with nothing decided yet.
    pub fn advance(&mut self, ws: &Workspace) {
        let next = (self.selected + 1..self.frames.len()).find(|&i| {
            let t = ws.todo_tile(self.frames[i]);
            !t.confirmed && !t.background && !t.deleted && t.drawn == 0
        });
        if let Some(i) = next {
            self.selected = i;
            self.ensure_visible = true;
        }
    }
}

fn frame_job(ws: &Workspace, image_idx: usize, max_side: u32) -> Job {
    let ds = &ws.project.dataset;
    let img = &ds.images()[image_idx];
    let region = Region { x: 0, y: 0, w: img.width.max(1), h: img.height.max(1) };
    Job::Thumb { path: ds.image_path(img), region, max_side }
}

pub fn show(
    ui: &mut egui::Ui,
    ws: &Workspace,
    thumbs: &mut Thumbs,
    state: &mut TodoState,
) -> Vec<TodoEvent> {
    let mut events = Vec::new();
    let n = state.frames.len();
    ui.horizontal(|ui| {
        ui.label(RichText::new(format!("{n} frames without boxes")).strong());
        if ui
            .toggle_value(&mut state.show_background, "Show background")
            .on_hover_text("Also list the frames already saved as background")
            .changed()
        {
            events.push(TodoEvent::Refresh);
        }
        ui.label(
            RichText::new(
                "newest first · Enter/E or double click: draw boxes · G: background (nothing to annotate) · \
                 F: delete frame · arrows/hjkl: move · Esc/T: back · applied on save",
            )
            .weak(),
        );
    });
    if n == 0 {
        ui.centered_and_justified(|ui| {
            ui.label("Nothing to do: every frame has boxes or is confirmed as background (check the split filter).")
        });
        return events;
    }
    let cols = ws.columns.max(1);
    let avail_w = ui.available_width();
    let cell_w = ((avail_w - GAP * (cols - 1) as f32) / cols as f32).max(60.0);
    let img_h = cell_w * 0.5625;
    let row_h = img_h + CAPTION_H;
    let pitch = row_h + ui.spacing().item_spacing.y;
    let max_side = if cell_w * ui.ctx().pixels_per_point() <= 300.0 { 256 } else { 512 };

    let mut area = egui::ScrollArea::vertical().auto_shrink(false);
    if state.ensure_visible && state.view_h > 0.0 {
        state.ensure_visible = false;
        let top = (state.selected / cols) as f32 * pitch;
        let mut offset = state.offset;
        if top < offset {
            offset = top;
        } else if top + row_h > offset + state.view_h {
            offset = top + row_h - state.view_h;
        }
        area = area.vertical_scroll_offset(offset);
    }
    let rows = n.div_ceil(cols);
    let out = area.show_rows(ui, row_h, rows, |ui, range| {
        let mut jobs = Vec::new();
        for row in range.clone() {
            let (rect, _) = ui.allocate_exact_size(Vec2::new(avail_w, row_h), Sense::hover());
            for c in 0..cols {
                let k = row * cols + c;
                let Some(&idx) = state.frames.get(k) else { break };
                let cell = Rect::from_min_size(
                    rect.min + Vec2::new(c as f32 * (cell_w + GAP), 0.0),
                    Vec2::new(cell_w, row_h),
                );
                let resp = ui.interact(cell, ui.id().with(("todo", idx)), Sense::click());
                if resp.double_clicked() {
                    events.push(TodoEvent::Open(k));
                } else if resp.clicked() {
                    events.push(TodoEvent::Select(k));
                }
                let job = frame_job(ws, idx, max_side);
                paint(
                    ui.painter(),
                    cell,
                    img_h,
                    ws,
                    idx,
                    thumbs.get(job.id()),
                    k == state.selected,
                );
                jobs.push(job);
            }
        }
        thumbs.want(jobs);
        range
    });
    state.offset = out.state.offset.y;
    state.view_h = out.inner_rect.height();
    events
}

fn paint(
    painter: &egui::Painter,
    cell: Rect,
    img_h: f32,
    ws: &Workspace,
    idx: usize,
    slot: Option<&Slot>,
    selected: bool,
) {
    let painter = &painter.with_clip_rect(cell.expand(4.0));
    let img_area = Rect::from_min_size(cell.min, Vec2::new(cell.width(), img_h));
    painter.rect_filled(img_area, 2.0, Color32::from_gray(28));
    match slot {
        Some(Slot::Ready(tex)) => {
            let size = tex.size_vec2();
            let s = (img_area.width() / size.x).min(img_area.height() / size.y);
            let r = Rect::from_center_size(img_area.center(), size * s);
            painter.image(
                tex.id(),
                r,
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
        }
        Some(Slot::Failed(e)) => {
            painter.text(
                img_area.center(),
                Align2::CENTER_CENTER,
                e,
                FontId::proportional(11.0),
                PRED_COLOR,
            );
        }
        None => {}
    }
    let tile = ws.todo_tile(idx);
    let mark = if tile.deleted {
        Some(("F", PRED_COLOR))
    } else if tile.confirmed {
        Some(("G background", GT_COLOR))
    } else if tile.drawn > 0 {
        Some(("boxes drawn", DRAWN_COLOR))
    } else if tile.background {
        Some(("background", Color32::from_gray(170)))
    } else {
        None
    };
    if let Some((text, color)) = mark {
        painter.text(
            img_area.right_top() + Vec2::new(-6.0, 4.0),
            Align2::RIGHT_TOP,
            text,
            FontId::proportional(16.0),
            color,
        );
    }
    let img = &ws.project.dataset.images()[idx];
    let name = img
        .extra
        .get(fastlabel_core::add::SOURCE_FIELD)
        .and_then(|v| v.as_str())
        .unwrap_or(&img.file_name);
    let date = tile
        .date
        .map(|t| fastlabel_core::dates::iso(t as i64).replace('T', " ").replace('Z', " UTC"))
        .unwrap_or_default();
    let split = ws.split_of(idx).map_or("no split", |s| s.name());
    let preds = match tile.predictions {
        Some(0) => " · no predictions".to_string(),
        Some(p) => format!(" · {p} predictions (mosaic)"),
        None => " · not predicted".to_string(),
    };
    let caption = Pos2::new(cell.left() + 2.0, img_area.bottom() + 2.0);
    painter.text(
        caption,
        Align2::LEFT_TOP,
        name,
        FontId::proportional(12.0),
        Color32::from_gray(210),
    );
    painter.text(
        caption + Vec2::new(0.0, 15.0),
        Align2::LEFT_TOP,
        format!("{split} · {date}{preds}"),
        FontId::proportional(11.0),
        Color32::from_gray(140),
    );
    if selected {
        painter.rect_stroke(img_area, 3.0, Stroke::new(2.5, SELECT_COLOR), StrokeKind::Inside);
    }
}
