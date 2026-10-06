//! Full-frame view and box editor: draw, select, move, 8-handle resize,
//! zoom (wheel, around the cursor) and pan (middle drag or Space + drag).
//! Returns actions; the caller turns them into pending edits.

use eframe::egui::{
    self, Align2, Color32, CornerRadius, FontId, PointerButton, Pos2, Rect, Sense, Shape, Stroke,
    StrokeKind, TextureHandle, Vec2,
};
use fastlabel_core::{AnnId, BBox, CategoryId, PredRef};

use crate::frame::{EditBox, Sel};
use crate::mosaic::{GT_COLOR, PRED_COLOR};
use crate::workspace::Mark;

const SELECT_COLOR: Color32 = Color32::from_rgb(255, 215, 0);
const DRAW_COLOR: Color32 = Color32::from_rgb(80, 200, 255);
const REPLACE_COLOR: Color32 = Color32::from_rgb(245, 150, 30);
const HANDLE_GRAB: f32 = 7.0;
/// Distinct colors for numbered boxes (no yellow: that is the selection).
const NUMBER_COLORS: [Color32; 10] = [
    Color32::from_rgb(0, 200, 255),
    Color32::from_rgb(255, 80, 200),
    Color32::from_rgb(255, 150, 0),
    Color32::from_rgb(150, 255, 80),
    Color32::from_rgb(170, 130, 255),
    Color32::from_rgb(255, 90, 90),
    Color32::from_rgb(0, 225, 170),
    Color32::from_rgb(255, 175, 210),
    Color32::from_rgb(120, 170, 255),
    Color32::from_rgb(235, 235, 235),
];
const MAX_ZOOM: f32 = 30.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Handle {
    TopLeft,
    Top,
    TopRight,
    Right,
    BottomRight,
    Bottom,
    BottomLeft,
    Left,
}

impl Handle {
    const ALL: [Handle; 8] = [
        Handle::TopLeft,
        Handle::Top,
        Handle::TopRight,
        Handle::Right,
        Handle::BottomRight,
        Handle::Bottom,
        Handle::BottomLeft,
        Handle::Left,
    ];

    /// Position on a box, as fractions of its width and height.
    fn anchor(self) -> (f64, f64) {
        match self {
            Handle::TopLeft => (0.0, 0.0),
            Handle::Top => (0.5, 0.0),
            Handle::TopRight => (1.0, 0.0),
            Handle::Right => (1.0, 0.5),
            Handle::BottomRight => (1.0, 1.0),
            Handle::Bottom => (0.5, 1.0),
            Handle::BottomLeft => (0.0, 1.0),
            Handle::Left => (0.0, 0.5),
        }
    }

    /// `orig` with the edges under this handle moved to image point (x, y).
    pub fn resize(self, orig: &BBox, x: f64, y: f64) -> BBox {
        let (mut x1, mut y1, mut x2, mut y2) = (orig.x, orig.y, orig.x2(), orig.y2());
        let (ax, ay) = self.anchor();
        if ax == 0.0 {
            x1 = x;
        } else if ax == 1.0 {
            x2 = x;
        }
        if ay == 0.0 {
            y1 = y;
        } else if ay == 1.0 {
            y2 = y;
        }
        BBox::from_corners(x1, y1, x2, y2)
    }
}

/// `b` moved by (dx, dy) and kept inside the image.
pub fn translate_within(b: &BBox, dx: f64, dy: f64, img_w: f64, img_h: f64) -> BBox {
    let x = (b.x + dx).clamp(0.0, (img_w - b.w).max(0.0));
    let y = (b.y + dy).clamp(0.0, (img_h - b.h).max(0.0));
    BBox::new(x, y, b.w, b.h)
}

enum Drag {
    Move { id: AnnId, start: Pos2, orig: BBox, cur: BBox },
    Resize { id: AnnId, handle: Handle, orig: BBox, cur: BBox },
    Draw { start: Pos2, cur: Pos2 },
}

#[derive(Default)]
pub struct EditorView {
    zoom: f32,
    pan: Vec2,
    drag: Option<Drag>,
    image_idx: Option<usize>,
    /// Region to zoom onto at the next `show`.
    focus: Option<BBox>,
}

pub enum EditorAction {
    Select(Option<Sel>),
    SetBox(AnnId, BBox),
    AddBox(BBox),
    /// Right click: go to the next frame.
    Next,
}

pub struct FrameData<'a> {
    pub image_idx: usize,
    pub texture: Option<&'a TextureHandle>,
    pub img_w: f64,
    pub img_h: f64,
    pub boxes: &'a [EditBox],
    pub preds: &'a [(PredRef, Option<Mark>)],
    pub sel: Option<Sel>,
    pub frame_deleted: bool,
    pub interactive: bool,
    /// Category names, shown on boxes when there is more than one.
    pub categories: &'a [(CategoryId, String)],
    /// Big labels on some GT boxes (cleaning: the key that keeps the box).
    pub numbers: &'a [(AnnId, String)],
}

struct Xform {
    origin: Pos2,
    scale: f32,
}

impl Xform {
    fn to_screen(&self, x: f64, y: f64) -> Pos2 {
        self.origin + Vec2::new(x as f32, y as f32) * self.scale
    }

    fn rect(&self, b: &BBox) -> Rect {
        Rect::from_min_max(self.to_screen(b.x, b.y), self.to_screen(b.x2(), b.y2()))
    }

    fn to_image(&self, p: Pos2) -> (f64, f64) {
        let v = (p - self.origin) / self.scale;
        (f64::from(v.x), f64::from(v.y))
    }
}

impl EditorView {
    /// Zooms onto `region` (image pixels, with some context) on next show.
    pub fn focus_on(&mut self, region: BBox) {
        self.focus = Some(region);
    }

    fn apply_focus(&mut self, rect: Rect, data: &FrameData<'_>) {
        let Some(b) = self.focus.take() else { return };
        // Context around the region: its own size again, at least 100 px.
        let (mx, my) = (b.w.max(100.0), b.h.max(100.0));
        let (w, h) = (b.w + 2.0 * mx, b.h + 2.0 * my);
        let fit = (rect.width() / data.img_w as f32).min(rect.height() / data.img_h as f32);
        let want = (rect.width() / w as f32).min(rect.height() / h as f32);
        self.zoom = (want / fit).clamp(1.0, MAX_ZOOM);
        let scale = fit * self.zoom;
        let size = Vec2::new(data.img_w as f32, data.img_h as f32) * scale;
        let center = Vec2::new((b.x + b.w / 2.0) as f32, (b.y + b.h / 2.0) as f32) * scale;
        self.pan = size / 2.0 - center;
    }

    pub fn reset_view(&mut self) {
        self.zoom = 1.0;
        self.pan = Vec2::ZERO;
    }

    fn xform(&self, rect: Rect, w: f64, h: f64) -> Xform {
        let fit = (rect.width() / w as f32).min(rect.height() / h as f32);
        let scale = fit * self.zoom.max(1.0);
        let size = Vec2::new(w as f32, h as f32) * scale;
        Xform { origin: rect.center() - size / 2.0 + self.pan, scale }
    }

    pub fn show(&mut self, ui: &mut egui::Ui, data: &FrameData<'_>) -> Vec<EditorAction> {
        if self.image_idx != Some(data.image_idx) {
            self.image_idx = Some(data.image_idx);
            self.drag = None;
            self.reset_view();
        }
        let mut actions = Vec::new();
        let sense = if data.interactive { Sense::click_and_drag() } else { Sense::hover() };
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), sense);
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0, Color32::from_gray(18));
        self.apply_focus(rect, data);

        if data.interactive {
            self.handle_zoom_pan(ui, &resp, rect, data);
        }
        let xf = self.xform(rect, data.img_w, data.img_h);
        if data.interactive {
            self.handle_pointer(ui, &resp, &xf, data, &mut actions);
            if resp.secondary_clicked() {
                actions.push(EditorAction::Next);
            }
        }

        let img_rect = xf.rect(&BBox::new(0.0, 0.0, data.img_w, data.img_h));
        match data.texture {
            Some(tex) => {
                let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
                painter.image(tex.id(), img_rect, uv, Color32::WHITE);
            }
            None => {
                painter.text(
                    rect.center(),
                    Align2::CENTER_CENTER,
                    "loading…",
                    FontId::proportional(16.0),
                    Color32::GRAY,
                );
            }
        }
        self.paint_boxes(&painter, &xf, data);
        if data.frame_deleted {
            painter.rect_filled(img_rect, 0, Color32::from_black_alpha(140));
            painter.text(
                img_rect.center(),
                Align2::CENTER_CENTER,
                "F",
                FontId::proportional(120.0),
                PRED_COLOR,
            );
        }
        actions
    }

    fn handle_zoom_pan(
        &mut self,
        ui: &egui::Ui,
        resp: &egui::Response,
        rect: Rect,
        data: &FrameData<'_>,
    ) {
        if let Some(cursor) = resp.hover_pos() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                let before = self.xform(rect, data.img_w, data.img_h);
                let (ix, iy) = before.to_image(cursor);
                self.zoom = (self.zoom.max(1.0) * (scroll * 0.002).exp()).clamp(1.0, MAX_ZOOM);
                let after = self.xform(rect, data.img_w, data.img_h);
                self.pan += cursor - after.to_screen(ix, iy);
                if self.zoom == 1.0 {
                    self.pan = Vec2::ZERO;
                }
            }
        }
        let space = ui.input(|i| i.key_down(egui::Key::Space));
        if resp.dragged_by(PointerButton::Middle)
            || (space && resp.dragged_by(PointerButton::Primary))
        {
            self.pan += resp.drag_delta();
        }
    }

    fn hit(&self, xf: &Xform, data: &FrameData<'_>, p: Pos2) -> Option<Sel> {
        let (x, y) = xf.to_image(p);
        let inside = |b: &BBox| x >= b.x && x <= b.x2() && y >= b.y && y <= b.y2();
        let gt = data
            .boxes
            .iter()
            .filter(|b| inside(&b.bbox))
            .min_by(|a, b| a.bbox.area().total_cmp(&b.bbox.area()))
            .map(|b| Sel::Gt(b.id));
        gt.or_else(|| {
            data.preds
                .iter()
                .filter(|(p, _)| inside(&p.bbox))
                .min_by(|a, b| a.0.bbox.area().total_cmp(&b.0.bbox.area()))
                .map(|(p, _)| Sel::Pred(*p))
        })
    }

    fn handle_at(xf: &Xform, b: &BBox, p: Pos2) -> Option<Handle> {
        Handle::ALL.into_iter().find(|h| {
            let (ax, ay) = h.anchor();
            xf.to_screen(b.x + ax * b.w, b.y + ay * b.h).distance(p) <= HANDLE_GRAB
        })
    }

    fn handle_pointer(
        &mut self,
        ui: &egui::Ui,
        resp: &egui::Response,
        xf: &Xform,
        data: &FrameData<'_>,
        actions: &mut Vec<EditorAction>,
    ) {
        if ui.input(|i| i.key_down(egui::Key::Space)) {
            return;
        }
        let selected_box = match data.sel {
            Some(Sel::Gt(id)) => data.boxes.iter().find(|b| b.id == id).copied(),
            _ => None,
        };
        if resp.drag_started_by(PointerButton::Primary)
            && let Some(start) = ui.input(|i| i.pointer.press_origin())
        {
            let handle =
                selected_box.and_then(|b| Self::handle_at(xf, &b.bbox, start).map(|h| (b, h)));
            self.drag = if let Some((b, handle)) = handle {
                Some(Drag::Resize { id: b.id, handle, orig: b.bbox, cur: b.bbox })
            } else {
                match self.hit(xf, data, start) {
                    Some(Sel::Gt(id)) => {
                        actions.push(EditorAction::Select(Some(Sel::Gt(id))));
                        let orig = data
                            .boxes
                            .iter()
                            .find(|b| b.id == id)
                            .map(|b| b.bbox)
                            .expect("hit box");
                        Some(Drag::Move { id, start, orig, cur: orig })
                    }
                    _ => {
                        actions.push(EditorAction::Select(None));
                        Some(Drag::Draw { start, cur: start })
                    }
                }
            };
        }
        if resp.dragged_by(PointerButton::Primary)
            && let Some(p) = resp.interact_pointer_pos()
        {
            match &mut self.drag {
                Some(Drag::Move { start, orig, cur, .. }) => {
                    let (sx, sy) = xf.to_image(*start);
                    let (px, py) = xf.to_image(p);
                    *cur = translate_within(orig, px - sx, py - sy, data.img_w, data.img_h);
                }
                Some(Drag::Resize { handle, orig, cur, .. }) => {
                    let (px, py) = xf.to_image(p);
                    let (px, py) = (px.clamp(0.0, data.img_w), py.clamp(0.0, data.img_h));
                    *cur = handle.resize(orig, px, py);
                }
                Some(Drag::Draw { cur, .. }) => *cur = p,
                None => {}
            }
        }
        if resp.drag_stopped() {
            match self.drag.take() {
                Some(Drag::Move { id, orig, cur, .. } | Drag::Resize { id, orig, cur, .. })
                    if cur != orig =>
                {
                    actions.push(EditorAction::SetBox(id, cur));
                }
                Some(Drag::Draw { start, cur }) => {
                    let (x1, y1) = xf.to_image(start);
                    let (x2, y2) = xf.to_image(cur);
                    let b = BBox::from_corners(x1, y1, x2, y2);
                    if b.w >= 2.0 && b.h >= 2.0 {
                        actions.push(EditorAction::AddBox(b));
                    }
                }
                _ => {}
            }
        }
        if resp.clicked()
            && let Some(p) = resp.interact_pointer_pos()
        {
            actions.push(EditorAction::Select(self.hit(xf, data, p)));
        }
    }

    fn paint_boxes(&self, painter: &egui::Painter, xf: &Xform, data: &FrameData<'_>) {
        let label_of = |cat: CategoryId| -> String {
            if data.categories.len() > 1 {
                data.categories
                    .iter()
                    .find(|(id, _)| *id == cat)
                    .map_or(String::new(), |(_, n)| format!(" {n}"))
            } else {
                String::new()
            }
        };
        for (p, mark) in data.preds {
            let selected = data.sel == Some(Sel::Pred(*p));
            let color = match mark {
                Some(Mark::Good) => GT_COLOR,
                Some(_) => Color32::from_gray(120),
                None => PRED_COLOR,
            };
            let r = xf.rect(&p.bbox);
            dashed_rect(
                painter,
                r,
                Stroke::new(
                    if selected { 2.5 } else { 1.5 },
                    if selected { SELECT_COLOR } else { color },
                ),
            );
            let mut text = format!("pred {:.2}{}", p.score, label_of(p.category_id));
            if let Some(m) = mark {
                text.push_str(if *m == Mark::Good { "  G" } else { "  D" });
            }
            tag(painter, r.left_bottom() + Vec2::new(0.0, 1.0), Align2::LEFT_TOP, &text, color);
        }
        for b in data.boxes {
            let bbox = match &self.drag {
                Some(Drag::Move { id, cur, .. } | Drag::Resize { id, cur, .. }) if *id == b.id => {
                    *cur
                }
                _ => b.bbox,
            };
            let r = xf.rect(&bbox);
            let selected = data.sel == Some(Sel::Gt(b.id));
            let number = data.numbers.iter().position(|(id, _)| *id == b.id);
            let (mut color, suffix) = match b.mark {
                Some(Mark::Delete) => (PRED_COLOR, "  D"),
                Some(Mark::Good) => (GT_COLOR, "  G"),
                Some(Mark::Replace) => (REPLACE_COLOR, "  R"),
                _ => (GT_COLOR, ""),
            };
            if let Some(n) = number {
                color = if b.mark == Some(Mark::Delete) {
                    Color32::from_gray(120)
                } else {
                    NUMBER_COLORS[n % NUMBER_COLORS.len()]
                };
            }
            let stroke = Stroke::new(
                if selected { 2.5 } else { 2.0 },
                if selected { SELECT_COLOR } else { color },
            );
            if b.mark == Some(Mark::Delete) {
                dashed_rect(painter, r, stroke);
            } else {
                painter.rect_stroke(r, 0, stroke, StrokeKind::Outside);
            }
            // Numbered boxes carry their number badge instead.
            if number.is_none() {
                let text = format!(
                    "GT{}{}{}",
                    label_of(b.category),
                    if b.added { " new" } else { "" },
                    suffix
                );
                tag(painter, r.left_top() - Vec2::new(0.0, 1.0), Align2::LEFT_BOTTOM, &text, color);
            }
            if selected && data.interactive {
                for h in Handle::ALL {
                    let (ax, ay) = h.anchor();
                    let c = xf.to_screen(bbox.x + ax * bbox.w, bbox.y + ay * bbox.h);
                    painter.rect_filled(
                        Rect::from_center_size(c, Vec2::splat(7.0)),
                        1,
                        SELECT_COLOR,
                    );
                }
            }
        }
        // Number badges in the box color, placed so they never overlap:
        // first free spot among the corners, else stacked below.
        let mut placed: Vec<Rect> = Vec::new();
        for (n, (id, label)) in data.numbers.iter().enumerate() {
            let Some(b) = data.boxes.iter().find(|b| b.id == *id) else { continue };
            let bbox = match &self.drag {
                Some(Drag::Move { id: d, cur, .. } | Drag::Resize { id: d, cur, .. })
                    if d == id =>
                {
                    *cur
                }
                _ => b.bbox,
            };
            let r = xf.rect(&bbox);
            let galley =
                painter.layout_no_wrap(label.clone(), FontId::proportional(20.0), Color32::BLACK);
            let size = galley.size() + Vec2::new(10.0, 2.0);
            let candidates = [
                Align2::LEFT_BOTTOM.anchor_size(r.right_top() + Vec2::new(2.0, 0.0), size),
                Align2::RIGHT_BOTTOM.anchor_size(r.left_top() - Vec2::new(2.0, 0.0), size),
                Align2::LEFT_TOP.anchor_size(r.right_bottom() + Vec2::new(2.0, 0.0), size),
                Align2::RIGHT_TOP.anchor_size(r.left_bottom() - Vec2::new(2.0, 0.0), size),
            ];
            let free = |c: &Rect| placed.iter().all(|p| !p.expand(1.0).intersects(*c));
            let badge = candidates.into_iter().find(free).unwrap_or_else(|| {
                let mut c = candidates[0];
                while !free(&c) {
                    c = c.translate(Vec2::new(0.0, size.y + 2.0));
                }
                c
            });
            placed.push(badge);
            let fill = if b.mark == Some(Mark::Delete) {
                Color32::from_gray(110)
            } else {
                NUMBER_COLORS[n % NUMBER_COLORS.len()]
            };
            painter.rect_filled(badge, CornerRadius::same(4), fill);
            painter.galley(badge.min + Vec2::new(5.0, 1.0), galley, Color32::BLACK);
        }
        if let Some(Drag::Draw { start, cur }) = &self.drag {
            painter.rect_stroke(
                Rect::from_two_pos(*start, *cur),
                0,
                Stroke::new(1.5, DRAW_COLOR),
                StrokeKind::Middle,
            );
        }
    }
}

fn dashed_rect(painter: &egui::Painter, r: Rect, stroke: Stroke) {
    let pts = [r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom(), r.left_top()];
    painter.extend(Shape::dashed_line(&pts, stroke, 6.0, 4.0));
}

fn tag(painter: &egui::Painter, pos: Pos2, anchor: Align2, text: &str, color: Color32) {
    let galley = painter.layout_no_wrap(text.to_string(), FontId::proportional(12.0), color);
    let rect = anchor.anchor_size(pos, galley.size()).expand(1.0);
    painter.rect_filled(rect, CornerRadius::same(2), Color32::from_black_alpha(170));
    painter.galley(rect.min + Vec2::splat(1.0), galley, color);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_move_the_right_edges() {
        let b = BBox::new(10.0, 10.0, 20.0, 20.0);
        assert_eq!(Handle::BottomRight.resize(&b, 40.0, 50.0), BBox::new(10.0, 10.0, 30.0, 40.0));
        assert_eq!(Handle::Top.resize(&b, 999.0, 0.0), BBox::new(10.0, 0.0, 20.0, 30.0));
        assert_eq!(Handle::Left.resize(&b, 5.0, 999.0), BBox::new(5.0, 10.0, 25.0, 20.0));
        // Dragging past the opposite edge flips instead of producing a negative box.
        assert_eq!(Handle::TopLeft.resize(&b, 40.0, 40.0), BBox::new(30.0, 30.0, 10.0, 10.0));
    }

    #[test]
    fn translate_stays_inside() {
        let b = BBox::new(10.0, 10.0, 20.0, 20.0);
        assert_eq!(
            translate_within(&b, -50.0, 5.0, 100.0, 100.0),
            BBox::new(0.0, 15.0, 20.0, 20.0)
        );
        assert_eq!(
            translate_within(&b, 500.0, 500.0, 100.0, 100.0),
            BBox::new(80.0, 80.0, 20.0, 20.0)
        );
    }
}
