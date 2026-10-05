//! The crop mosaic: a virtualized grid (only visible rows are laid out and
//! loaded) with keyboard-driven selection.

use eframe::egui::{
    self, Align2, Color32, CornerRadius, FontId, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2,
};
use fastlabel_core::BBox;
use fastlabel_media::{Job, Region};

use crate::crops::{Crop, CropStatus};
use crate::thumbs::{Slot, Thumbs};
use crate::workspace::{CropMarks, Mark, Workspace};

pub const GT_COLOR: Color32 = Color32::from_rgb(60, 220, 90);
pub const PRED_COLOR: Color32 = Color32::from_rgb(240, 70, 60);
const REPLACE_COLOR: Color32 = Color32::from_rgb(245, 150, 30);
const SELECT_COLOR: Color32 = Color32::from_rgb(255, 215, 0);
const CELL_BG: Color32 = Color32::from_gray(28);
const CAPTION_H: f32 = 18.0;
const GAP: f32 = 6.0;
/// Rows loaded ahead of / behind the visible ones.
const PREFETCH_SCREENS_AHEAD: usize = 2;
const PREFETCH_SCREENS_BEHIND: usize = 1;

#[derive(Default)]
pub struct MosaicState {
    offset: f32,
    view_h: f32,
    /// Scroll so the selection is visible on the next frame.
    pub ensure_visible: bool,
    pub rows_per_screen: usize,
}

pub enum MosaicEvent {
    Select(usize),
    /// Right click: toggle D on that crop without advancing.
    QuickDelete(usize),
    Open(usize),
}

pub struct Margins {
    pub x: f64,
    pub y: f64,
}

/// Pixel region shown for a crop and the job loading it.
pub fn crop_job(ws: &Workspace, crop: &Crop, m: &Margins, max_side: u32) -> Option<(Job, Region)> {
    let ds = &ws.project.dataset;
    let img = &ds.images()[crop.image_idx];
    let region = Region::around(&crop.frame_box(), m.x, m.y, img.width, img.height)?;
    Some((Job::Thumb { path: ds.image_path(img), region, max_side }, region))
}

fn thumb_side(cell_px: f32) -> u32 {
    if cell_px <= 300.0 { 256 } else { 512 }
}

pub fn show(
    ui: &mut egui::Ui,
    ws: &Workspace,
    thumbs: &mut Thumbs,
    state: &mut MosaicState,
    margins: &Margins,
) -> Vec<MosaicEvent> {
    let mut events = Vec::new();
    let n = ws.crops.len();
    if n == 0 {
        ui.centered_and_justified(|ui| ui.label("No crops match the current filters."));
        return events;
    }
    let cols = ws.columns.max(1);
    let avail_w = ui.available_width();
    let cell_w = ((avail_w - GAP * (cols - 1) as f32) / cols as f32).max(40.0);
    let img_h = cell_w * 0.75;
    let row_h = img_h + CAPTION_H;
    let pitch = row_h + ui.spacing().item_spacing.y;
    let n_rows = n.div_ceil(cols);
    let max_side = thumb_side(cell_w * ui.ctx().pixels_per_point());

    let mut area = egui::ScrollArea::vertical().auto_shrink(false);
    // The viewport height is only known after the first frame.
    if state.ensure_visible && state.view_h > 0.0 {
        state.ensure_visible = false;
        let top = (ws.selected / cols) as f32 * pitch;
        let mut offset = state.offset;
        if top < offset {
            offset = top;
        } else if top + row_h > offset + state.view_h {
            offset = top + row_h - state.view_h;
        }
        if offset != state.offset {
            area = area.vertical_scroll_offset(offset);
        }
    }

    let out = area.show_rows(ui, row_h, n_rows, |ui, rows| {
        for row in rows.clone() {
            let (rect, _) = ui.allocate_exact_size(Vec2::new(avail_w, row_h), Sense::hover());
            for c in 0..cols {
                let idx = row * cols + c;
                if idx >= n {
                    break;
                }
                let cell = Rect::from_min_size(
                    rect.min + Vec2::new(c as f32 * (cell_w + GAP), 0.0),
                    Vec2::new(cell_w, row_h),
                );
                let resp = ui.interact(cell, ui.id().with(("crop", idx)), Sense::click());
                if resp.double_clicked() {
                    events.push(MosaicEvent::Open(idx));
                } else if resp.clicked() {
                    events.push(MosaicEvent::Select(idx));
                } else if resp.secondary_clicked() {
                    events.push(MosaicEvent::QuickDelete(idx));
                }
                let crop = &ws.crops[idx];
                let job = crop_job(ws, crop, margins, max_side);
                let slot = job.as_ref().and_then(|(j, _)| thumbs.get(j.id()));
                resp.on_hover_text(tooltip(ws, crop, slot));
                paint_cell(
                    ui.painter(),
                    cell,
                    img_h,
                    crop,
                    ws.marks_of(crop),
                    slot,
                    job.map(|(_, r)| r),
                    idx == ws.selected,
                    split_name(ws, crop),
                );
            }
        }
        rows
    });
    state.offset = out.state.offset.y;
    state.view_h = out.inner_rect.height();
    state.rows_per_screen = ((state.view_h / pitch).floor() as usize).max(1);

    // Visible crops first (closest to the selection first), then prefetch.
    let visible = out.inner;
    let sel = ws.selected;
    let mut order: Vec<usize> = (visible.start * cols..(visible.end * cols).min(n)).collect();
    order.sort_by_key(|&i| i.abs_diff(sel));
    let screen = state.rows_per_screen * cols;
    let ahead = visible.end * cols..(visible.end * cols + PREFETCH_SCREENS_AHEAD * screen).min(n);
    let behind_start = (visible.start * cols).saturating_sub(PREFETCH_SCREENS_BEHIND * screen);
    order.extend(ahead);
    order.extend((behind_start..visible.start * cols).rev());
    let jobs = order
        .into_iter()
        .filter_map(|i| crop_job(ws, &ws.crops[i], margins, max_side))
        .map(|(j, _)| j);
    thumbs.want(jobs.collect());
    events
}

fn tooltip(ws: &Workspace, crop: &Crop, slot: Option<&Slot>) -> String {
    let img = &ws.project.dataset.images()[crop.image_idx];
    let b = crop.primary();
    let mut lines = vec![img.file_name.clone(), format!("{:.0} × {:.0} px", b.w, b.h)];
    let status = match crop.status {
        CropStatus::Tp => "TP",
        CropStatus::Fn => "FN",
        CropStatus::Fp => "FP (prediction only)",
        CropStatus::Unpredicted => "not predicted",
    };
    match crop.iou {
        Some(iou) => lines.push(format!("{status}, IoU {iou:.2}")),
        None => lines.push(status.into()),
    }
    if let Some(p) = &crop.pred {
        lines.push(format!("confidence {:.2}", p.score));
    }
    if let Some(u) = crop.uniformity {
        lines.push(format!("color uniformity (gray std) {u:.1}"));
    }
    if let Some(Slot::Failed(e)) = slot {
        lines.push(format!("cannot load image: {e}"));
    }
    lines.join("\n")
}

fn split_name(ws: &Workspace, crop: &Crop) -> &'static str {
    ws.project.dataset.split_of(crop.image_idx).map_or("", |s| s.name())
}

#[allow(clippy::too_many_arguments)]
fn paint_cell(
    painter: &egui::Painter,
    cell: Rect,
    img_h: f32,
    crop: &Crop,
    marks: CropMarks,
    slot: Option<&Slot>,
    region: Option<Region>,
    selected: bool,
    split: &str,
) {
    let img_area = Rect::from_min_size(cell.min, Vec2::new(cell.width(), img_h));
    painter.rect_filled(img_area, CornerRadius::same(3), CELL_BG);

    match (slot, region) {
        (Some(Slot::Ready(tex)), Some(region)) => {
            let size = tex.size_vec2();
            let scale = (img_area.width() / size.x).min(img_area.height() / size.y);
            let rect = Rect::from_center_size(img_area.center(), size * scale);
            painter.image(
                tex.id(),
                rect,
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
            let s = rect.width() / region.w as f32;
            let to_screen = |b: &BBox| {
                Rect::from_min_size(
                    rect.min
                        + Vec2::new(
                            (b.x - f64::from(region.x)) as f32,
                            (b.y - f64::from(region.y)) as f32,
                        ) * s,
                    Vec2::new(b.w as f32, b.h as f32) * s,
                )
            };
            let clip = painter.with_clip_rect(rect.expand(1.0));
            if let Some(p) = &crop.pred {
                let r = to_screen(&p.bbox);
                clip.rect_stroke(r, 0, Stroke::new(1.5, PRED_COLOR), StrokeKind::Outside);
                tag(
                    &clip,
                    r.left_bottom() + Vec2::new(0.0, 1.0),
                    Align2::LEFT_TOP,
                    &format!("pred {:.2}", p.score),
                    PRED_COLOR,
                );
            }
            if let Some(g) = &crop.gt {
                let r = to_screen(g);
                clip.rect_stroke(r, 0, Stroke::new(2.0, GT_COLOR), StrokeKind::Outside);
                tag(&clip, r.left_top() - Vec2::new(0.0, 1.0), Align2::LEFT_BOTTOM, "GT", GT_COLOR);
            }
        }
        (Some(Slot::Failed(_)), _) => {
            painter.text(
                img_area.center(),
                Align2::CENTER_CENTER,
                "image unavailable",
                FontId::proportional(12.0),
                PRED_COLOR,
            );
        }
        _ => {
            painter.text(
                img_area.center(),
                Align2::CENTER_CENTER,
                "…",
                FontId::proportional(16.0),
                Color32::GRAY,
            );
        }
    }

    if marks.frame_deleted {
        painter.rect_filled(img_area, CornerRadius::same(3), Color32::from_black_alpha(160));
        painter.text(
            img_area.center(),
            Align2::CENTER_CENTER,
            "F",
            FontId::proportional(img_h * 0.45),
            PRED_COLOR,
        );
    }
    if let Some(m) = marks.mark {
        let (letter, color) = match m {
            Mark::Delete => ("D", PRED_COLOR),
            Mark::Good => ("G", GT_COLOR),
            Mark::Replace => ("R", REPLACE_COLOR),
            Mark::Frame => ("F", PRED_COLOR),
        };
        let badge =
            Rect::from_min_size(img_area.right_top() + Vec2::new(-30.0, 4.0), Vec2::splat(26.0));
        painter.rect_filled(badge, CornerRadius::same(4), color);
        painter.text(
            badge.center(),
            Align2::CENTER_CENTER,
            letter,
            FontId::proportional(18.0),
            Color32::BLACK,
        );
    }

    // Caption: id / kind on the left, status on the right.
    let caption =
        Rect::from_min_max(Pos2::new(cell.left(), img_area.bottom()), cell.right_bottom());
    let font = FontId::proportional(12.0);
    let left = match crop.ann_id() {
        Some(id) => format!("{}#{id}", if marks.reviewed { "✓ " } else { "" }),
        None => format!("{}pred", if marks.reviewed { "✓ " } else { "" }),
    };
    let left_color = if crop.is_pred() { PRED_COLOR } else { Color32::from_gray(200) };
    painter.text(
        caption.left_center() + Vec2::new(3.0, 0.0),
        Align2::LEFT_CENTER,
        left,
        font.clone(),
        left_color,
    );
    let status = match crop.status {
        CropStatus::Tp => format!("TP {:.2}", crop.iou.unwrap_or(0.0)),
        CropStatus::Fn => "FN".into(),
        CropStatus::Fp => "FP".into(),
        CropStatus::Unpredicted => String::new(),
    };
    let right =
        [split, &status].into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join("  ");
    painter.text(
        caption.right_center() - Vec2::new(3.0, 0.0),
        Align2::RIGHT_CENTER,
        right,
        font,
        Color32::from_gray(150),
    );

    if selected {
        painter.rect_stroke(
            cell.expand(2.0),
            CornerRadius::same(4),
            Stroke::new(3.0, SELECT_COLOR),
            StrokeKind::Outside,
        );
    }
}

/// Small label with a dark background.
fn tag(painter: &egui::Painter, pos: Pos2, anchor: Align2, text: &str, color: Color32) {
    let galley = painter.layout_no_wrap(text.to_string(), FontId::proportional(11.0), color);
    let rect = anchor.anchor_size(pos, galley.size()).expand(1.0);
    painter.rect_filled(rect, CornerRadius::same(2), Color32::from_black_alpha(170));
    painter.galley(rect.min + Vec2::splat(1.0), galley, color);
}
