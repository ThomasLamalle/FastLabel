//! Cluster mode: frames as a 2D point cloud (UMAP of image embeddings),
//! colored by split or cluster, lasso selection and actions on the selection
//! (move to a split, show in the mosaic, delete frames).

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;

use eframe::egui::{
    self, Align2, Color32, FontId, Mesh, PointerButton, Pos2, Rect, RichText, Sense, Shape, Stroke,
    StrokeKind, Vec2,
};
use fastlabel_core::{ImageId, Split};
use fastlabel_media::{Job, Region};

use crate::cluster::in_polygon;
use crate::embeddings::{self, SavedLayout};
use crate::thumbs::{Slot, Thumbs};
use crate::workspace::Workspace;

const TRAIN_COLOR: Color32 = Color32::from_rgb(70, 140, 240);
const VAL_COLOR: Color32 = Color32::from_rgb(245, 150, 30);
const TEST_COLOR: Color32 = Color32::from_rgb(190, 90, 230);
const NO_SPLIT_COLOR: Color32 = Color32::from_gray(150);
const NOISE_COLOR: Color32 = Color32::from_gray(95);
const DELETED_COLOR: Color32 = Color32::from_gray(55);
const SELECT_COLOR: Color32 = Color32::from_rgb(255, 215, 0);
const BG: Color32 = Color32::from_gray(18);
/// Point half-size in screen pixels.
const POINT_R: f32 = 2.5;
/// Hover / click distance to a point, screen pixels.
const PICK_R: f32 = 7.0;
const STRIP_THUMB: f32 = 120.0;
const STRIP_SIDE: u32 = 192;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ColorBy {
    #[default]
    Split,
    Cluster,
}

pub fn split_color(split: Option<Split>) -> Color32 {
    match split {
        Some(Split::Train) => TRAIN_COLOR,
        Some(Split::Val) => VAL_COLOR,
        Some(Split::Test) => TEST_COLOR,
        None => NO_SPLIT_COLOR,
    }
}

/// Distinct colors for cluster labels (golden-angle hues); noise is grey.
pub fn cluster_color(label: i32) -> Color32 {
    if label < 0 {
        return NOISE_COLOR;
    }
    let hue = (label as f32 * 0.618_034).fract();
    let value = if label % 2 == 0 { 0.95 } else { 0.75 };
    egui::ecolor::Hsva::new(hue, 0.75, value, 1.0).into()
}

pub enum ClusterEvent {
    BrowseModel,
    ComputeEmbeddings,
    /// Recompute the layout (new embeddings).
    Relayout,
    /// Recompute only the clusters (minimum size changed).
    Recluster,
    Move(Split),
    ShowInMosaic,
    ToggleDelete,
    /// Open this frame (image index) in the editor popup.
    Open(usize),
    /// HTML report on all frames, the selection, or one cluster.
    ReportAll,
    ReportSelection,
    ReportCluster(i32),
    /// Assign whole clusters to splits (settings in the session).
    AutoSplit,
    Exit,
}

/// What the app knows about the background work, for the setup bar.
pub struct Setup<'a> {
    pub model_path: &'a mut Option<PathBuf>,
    /// Frames without an embedding yet.
    pub missing: usize,
    pub busy: bool,
    pub auto_split: &'a mut crate::cluster::AutoSplit,
}

pub struct ClusterView {
    pub layout: Option<SavedLayout>,
    /// Point and cluster label per image index; rebuilt when `dirty`.
    points: Vec<Option<([f32; 2], i32)>>,
    /// Frames (with an embedding) not in the layout.
    pub unplaced: usize,
    pub dirty: bool,
    pub selection: BTreeSet<ImageId>,
    /// Lasso being drawn, in layout coordinates.
    lasso: Vec<[f32; 2]>,
    center: [f32; 2],
    /// Screen pixels per layout unit; 0 = fit on the next frame.
    scale: f32,
    /// Frame highlighted from the selection strip.
    focus: Option<usize>,
    /// Cluster of the point the context menu was opened on.
    menu_cluster: Option<i32>,
    pub color_by: ColorBy,
    pub min_cluster_size: usize,
}

impl Default for ClusterView {
    fn default() -> Self {
        Self {
            layout: None,
            points: Vec::new(),
            unplaced: 0,
            dirty: true,
            selection: BTreeSet::new(),
            lasso: Vec::new(),
            center: [0.0, 0.0],
            scale: 0.0,
            focus: None,
            menu_cluster: None,
            color_by: ColorBy::Split,
            min_cluster_size: 20,
        }
    }
}

impl ClusterView {
    pub fn set_layout(&mut self, layout: SavedLayout) {
        let refit = self.layout.as_ref().is_none_or(|l| l.points != layout.points);
        self.layout = Some(layout);
        self.dirty = true;
        if refit {
            self.scale = 0.0;
        }
    }

    /// Maps layout points to the dataset's image indices.
    pub fn sync(&mut self, ws: &Workspace, has_embedding: impl Fn(u64) -> bool) {
        let ds = &ws.project.dataset;
        let by_key: HashMap<u64, usize> = self
            .layout
            .as_ref()
            .map(|l| l.keys.iter().enumerate().map(|(i, k)| (*k, i)).collect())
            .unwrap_or_default();
        let mut unplaced = 0;
        self.points = ds
            .images()
            .iter()
            .map(|img| {
                let k = embeddings::key(img.id, &img.file_name);
                let p = by_key.get(&k).and_then(|&i| {
                    let l = self.layout.as_ref()?;
                    Some((l.points[i], l.labels[i]))
                });
                unplaced += usize::from(p.is_none() && has_embedding(k));
                p
            })
            .collect();
        self.unplaced = unplaced;
        let exists: BTreeSet<ImageId> = ds.images().iter().map(|i| i.id).collect();
        self.selection.retain(|id| exists.contains(id));
        self.dirty = false;
    }

    /// Image indices of the selection, in dataset order.
    pub fn selected_frames(&self, ws: &Workspace) -> Vec<usize> {
        let ds = &ws.project.dataset;
        let mut v: Vec<usize> = self.selection.iter().filter_map(|id| ds.image_idx(*id)).collect();
        v.sort_unstable();
        v
    }

    /// Frames shown with the current split and "No boxes" filters.
    fn visible(&self, ws: &Workspace) -> Vec<(usize, [f32; 2], i32)> {
        let ds = &ws.project.dataset;
        self.points
            .iter()
            .enumerate()
            .filter_map(|(i, p)| p.map(|(pos, l)| (i, pos, l)))
            .filter(|(i, ..)| ws.opts.split.matches(ws.split_of(*i)))
            .filter(|(i, ..)| !ws.opts.no_boxes || ds.annotations_of(*i).next().is_none())
            .collect()
    }

    /// Frames (image indices) grouped for an auto-split: one group per
    /// cluster, one per unclustered frame; frames marked F are left out.
    pub fn split_groups(&self, ws: &Workspace) -> Vec<Vec<usize>> {
        let mut clusters: std::collections::BTreeMap<i32, Vec<usize>> = Default::default();
        let mut groups = Vec::new();
        for (i, p) in self.points.iter().enumerate() {
            let Some((_, label)) = p else { continue };
            if ws.is_frame_deleted_idx(i) {
                continue;
            }
            if *label >= 0 {
                clusters.entry(*label).or_default().push(i);
            } else {
                groups.push(vec![i]);
            }
        }
        groups.extend(clusters.into_values());
        groups
    }

    /// Frames of a cluster (whatever the split filter).
    pub fn cluster_frames(&self, ws: &Workspace, label: i32) -> BTreeSet<ImageId> {
        let ds = &ws.project.dataset;
        self.points
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_some_and(|(_, l)| l == label))
            .map(|(i, _)| ds.images()[i].id)
            .collect()
    }

    pub fn select_all_visible(&mut self, ws: &Workspace) {
        let ds = &ws.project.dataset;
        self.selection = self.visible(ws).iter().map(|(i, ..)| ds.images()[*i].id).collect();
    }

    /// Lasso / click selection: replace, add (Shift) or remove (Ctrl).
    fn apply_selection(&mut self, ws: &Workspace, hits: Vec<usize>, m: egui::Modifiers) {
        let ids = hits.into_iter().map(|i| ws.project.dataset.images()[i].id);
        if m.command {
            for id in ids {
                self.selection.remove(&id);
            }
        } else if m.shift {
            self.selection.extend(ids);
        } else {
            self.selection = ids.collect();
        }
    }

    fn fit(&mut self, rect: Rect) {
        let pts: Vec<[f32; 2]> = self.points.iter().flatten().map(|(p, _)| *p).collect();
        if pts.is_empty() {
            return;
        }
        let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
        for p in &pts {
            for d in 0..2 {
                lo[d] = lo[d].min(p[d]);
                hi[d] = hi[d].max(p[d]);
            }
        }
        self.center = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0];
        let span = ((hi[0] - lo[0]) / rect.width()).max((hi[1] - lo[1]) / rect.height());
        self.scale = if span > 0.0 { 0.92 / span } else { 1.0 };
    }

    fn to_screen(&self, rect: Rect, p: [f32; 2]) -> Pos2 {
        rect.center()
            + Vec2::new((p[0] - self.center[0]) * self.scale, (p[1] - self.center[1]) * self.scale)
    }

    fn to_layout(&self, rect: Rect, p: Pos2) -> [f32; 2] {
        let d = (p - rect.center()) / self.scale;
        [self.center[0] + d.x, self.center[1] + d.y]
    }

    /// Model, computation and cluster settings (above the cloud).
    pub fn setup_bar(&mut self, ui: &mut egui::Ui, setup: Setup) -> Vec<ClusterEvent> {
        let mut events = Vec::new();
        ui.horizontal_wrapped(|ui| {
            ui.label("Embedding model");
            let mut path =
                setup.model_path.as_ref().map(|p| p.display().to_string()).unwrap_or_default();
            if ui
                .add(
                    egui::TextEdit::singleline(&mut path)
                        .hint_text("dinov2_vits14.onnx")
                        .desired_width(260.0),
                )
                .on_hover_text("ONNX image embedder, e.g. exported by scripts/export_dinov2.py")
                .changed()
            {
                *setup.model_path = (!path.is_empty()).then(|| PathBuf::from(&path));
            }
            if ui.button("Browse…").clicked() {
                events.push(ClusterEvent::BrowseModel);
            }
            let idle = !setup.busy;
            if setup.missing > 0
                && ui
                    .add_enabled(
                        idle,
                        egui::Button::new(format!("Compute embeddings ({} frames)", setup.missing)),
                    )
                    .on_hover_text("Runs in the background; vectors are cached in .fastlabel/")
                    .clicked()
            {
                events.push(ClusterEvent::ComputeEmbeddings);
            }
            if self.unplaced > 0
                && ui
                    .add_enabled(
                        idle,
                        egui::Button::new(format!("Update layout (+{} frames)", self.unplaced)),
                    )
                    .clicked()
            {
                events.push(ClusterEvent::Relayout);
            }
            ui.separator();
            ui.label("Color");
            ui.selectable_value(&mut self.color_by, ColorBy::Split, "Split");
            ui.selectable_value(&mut self.color_by, ColorBy::Cluster, "Clusters");
            ui.label("min cluster size");
            ui.add(egui::DragValue::new(&mut self.min_cluster_size).range(2..=1000));
            let changed =
                self.layout.as_ref().is_some_and(|l| l.min_cluster_size != self.min_cluster_size);
            if ui.add_enabled(idle && changed, egui::Button::new("Recluster")).clicked() {
                events.push(ClusterEvent::Recluster);
            }
            if ui.button("Fit view").clicked() {
                self.scale = 0.0;
            }
            let has_layout = self.layout.is_some();
            ui.add_enabled_ui(has_layout, |ui| {
                ui.menu_button("Auto-split…", |ui| {
                    ui.set_min_width(330.0);
                    let a = &mut *setup.auto_split;
                    let mut val = a.val * 100.0;
                    let mut test = a.test * 100.0;
                    let mut cap = a.max_group_share * 100.0;
                    ui.label(
                        RichText::new(
                            "Whole clusters go to one split, so similar frames never sit on both sides. \
                             Unclustered frames are placed one by one. Frames marked F are left alone.",
                        )
                        .weak(),
                    );
                    egui::Grid::new("auto-split").num_columns(2).show(ui, |ui| {
                        ui.label("val");
                        ui.add(egui::DragValue::new(&mut val).range(0.0..=50.0).suffix(" %"));
                        ui.end_row();
                        ui.label("test");
                        ui.add(egui::DragValue::new(&mut test).range(0.0..=50.0).suffix(" %"));
                        ui.end_row();
                        ui.label("max cluster share")
                            .on_hover_text("A cluster may fill at most this share of the val / test target (more scenes per split); larger clusters stay in train");
                        ui.add(egui::DragValue::new(&mut cap).range(5.0..=100.0).suffix(" %"));
                        ui.end_row();
                    });
                    a.val = val / 100.0;
                    a.test = test / 100.0;
                    a.max_group_share = cap / 100.0;
                    ui.horizontal(|ui| {
                        if ui.button("Apply").clicked() {
                            events.push(ClusterEvent::AutoSplit);
                            ui.close();
                        }
                        if ui
                            .button("Reroll")
                            .on_hover_text("Another random draw; it replaces the previous auto-split")
                            .clicked()
                        {
                            a.seed = a.seed.wrapping_add(1);
                            events.push(ClusterEvent::AutoSplit);
                        }
                        ui.label(RichText::new(format!("draw #{}", a.seed)).weak());
                    });
                    ui.label(RichText::new("Pending until saved; U undoes it in one step. Check the result with the report.").weak());
                });
            });
            if ui
                .button("Report: all frames")
                .on_hover_text("HTML report: split balance, box statistics, input size guide, similarity between splits")
                .clicked()
            {
                events.push(ClusterEvent::ReportAll);
            }
        });
        events
    }

    /// Counts and shares per split (pending moves included) of `frames`.
    fn split_counts(ws: &Workspace, frames: impl Iterator<Item = usize>) -> String {
        let mut counts = [0usize; 4];
        for i in frames {
            counts[ws.split_of(i).map_or(3, |s| s as usize)] += 1;
        }
        split_shares(counts)
    }

    /// The cloud and the selection strip.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        ws: &Workspace,
        thumbs: &mut Thumbs,
    ) -> Vec<ClusterEvent> {
        let mut events = Vec::new();
        let visible = self.visible(ws);
        let selected = self.selected_frames(ws);
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(format!("{} frames", visible.len())).strong());
            ui.label(RichText::new(Self::split_counts(ws, visible.iter().map(|v| v.0))).weak());
            if !selected.is_empty() {
                ui.separator();
                ui.label(
                    RichText::new(format!("selected {}", selected.len())).color(SELECT_COLOR),
                );
                ui.label(RichText::new(Self::split_counts(ws, selected.iter().copied())).weak());
            }
            ui.separator();
            ui.label(
                RichText::new(
                    "drag: lasso (Shift add, Ctrl remove) · right click: actions · 1/2/3: move to train/val/test · \
                     F: delete frames · Enter: show in mosaic · Ctrl+A: all · wheel: zoom · middle drag: pan · Esc: back",
                )
                .weak(),
            );
        });

        if !selected.is_empty() {
            egui::Panel::bottom("cluster-strip")
                .resizable(true)
                .default_size(STRIP_THUMB + 40.0)
                .min_size(60.0)
                .show(ui, |ui| self.strip(ui, ws, thumbs, &selected, &mut events));
        }

        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, BG);
        if self.layout.is_none() || visible.is_empty() {
            let text = if self.layout.is_none() {
                "No layout yet: choose an embedding model and compute the embeddings."
            } else {
                "No frames to show with the current filters."
            };
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                text,
                FontId::proportional(16.0),
                Color32::GRAY,
            );
            return events;
        }
        if self.scale <= 0.0 {
            self.fit(rect);
        }
        self.zoom_pan(ui, &resp, rect);

        let pointer = resp.hover_pos();
        let hovered = pointer.and_then(|p| {
            visible
                .iter()
                .map(|(i, pos, _)| (*i, self.to_screen(rect, *pos).distance(p)))
                .filter(|(_, d)| *d <= PICK_R)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(i, _)| i)
        });

        // Lasso.
        if resp.drag_started_by(PointerButton::Primary) {
            self.lasso.clear();
        }
        if resp.dragged_by(PointerButton::Primary)
            && let Some(p) = resp.interact_pointer_pos()
        {
            let lp = self.to_layout(rect, p);
            let far = self.lasso.last().is_none_or(|q| self.to_screen(rect, *q).distance(p) > 2.0);
            if far {
                self.lasso.push(lp);
            }
        }
        if resp.drag_stopped_by(PointerButton::Primary) {
            let poly = std::mem::take(&mut self.lasso);
            let hits =
                visible.iter().filter(|(_, pos, _)| in_polygon(*pos, &poly)).map(|v| v.0).collect();
            let m = ui.input(|i| i.modifiers);
            self.apply_selection(ws, hits, m);
        }
        if resp.double_clicked() {
            if let Some(i) = hovered {
                events.push(ClusterEvent::Open(i));
            }
        } else if resp.clicked() {
            let m = ui.input(|i| i.modifiers);
            match hovered {
                Some(i) if m.shift || m.command => {
                    let id = ws.project.dataset.images()[i].id;
                    if !self.selection.remove(&id) {
                        self.selection.insert(id);
                    }
                }
                Some(i) => self.apply_selection(ws, vec![i], m),
                None if !m.shift && !m.command => self.selection.clear(),
                None => {}
            }
        }
        if resp.secondary_clicked() {
            self.menu_cluster =
                hovered.and_then(|i| self.points[i]).map(|p| p.1).filter(|l| *l >= 0);
            if let Some(i) = hovered
                && !self.selection.contains(&ws.project.dataset.images()[i].id)
            {
                self.selection = [ws.project.dataset.images()[i].id].into();
            }
        }

        self.paint(&painter, rect, ws, &visible, hovered);

        if let Some(i) = hovered {
            let img = &ws.project.dataset.images()[i];
            let job = frame_job(ws, i);
            let slot = thumbs.get(job.id()).map(|s| match s {
                Slot::Ready(t) => Ok(t.clone()),
                Slot::Failed(e) => Err(e.clone()),
            });
            thumbs.want_urgent(job);
            let label = self.points[i].map_or(-1, |p| p.1);
            let split = ws.split_of(i).map_or("no split", |s| s.name());
            let deleted = ws.is_frame_deleted_idx(i);
            egui::Tooltip::always_open(
                ui.ctx().clone(),
                ui.layer_id(),
                ui.id().with("hover"),
                egui::PopupAnchor::Pointer,
            )
            .gap(12.0)
            .show(|ui| {
                match slot {
                    Some(Ok(tex)) => {
                        let size = tex.size_vec2();
                        let s = (240.0 / size.x.max(size.y)).min(1.0);
                        ui.image((tex.id(), size * s));
                    }
                    Some(Err(e)) => {
                        ui.label(format!("cannot load image: {e}"));
                    }
                    None => {
                        ui.spinner();
                    }
                }
                ui.label(&img.file_name);
                let cluster =
                    if label < 0 { "noise".to_string() } else { format!("cluster {label}") };
                ui.label(format!(
                    "{split} · {cluster}{}",
                    if deleted { " · F (deleted on save)" } else { "" }
                ));
            });
        }

        let n = selected.len();
        let all_deleted = n > 0 && selected.iter().all(|&i| ws.is_frame_deleted_idx(i));
        let menu_cluster = self.menu_cluster;
        resp.context_menu(|ui| {
            if let Some(label) = menu_cluster {
                let frames = self.cluster_frames(ws, label);
                if ui.button(format!("Select cluster #{label} ({} frames)", frames.len())).clicked()
                {
                    self.selection = frames;
                    ui.close();
                }
                if ui.button(format!("Report on cluster #{label}")).clicked() {
                    events.push(ClusterEvent::ReportCluster(label));
                    ui.close();
                }
                ui.separator();
            }
            if n == 0 {
                ui.label("Draw around points to select frames");
                return;
            }
            ui.label(RichText::new(format!("{n} frames selected")).strong());
            for (k, s) in Split::ALL.iter().enumerate() {
                if ui.button(format!("Move to {}  ({})", s.name(), k + 1)).clicked() {
                    events.push(ClusterEvent::Move(*s));
                    ui.close();
                }
            }
            ui.separator();
            if ui.button("Show in mosaic  (Enter)").clicked() {
                events.push(ClusterEvent::ShowInMosaic);
                ui.close();
            }
            let label = if all_deleted { "Keep frames (undo F)" } else { "Delete frames  (F)" };
            if ui.button(label).clicked() {
                events.push(ClusterEvent::ToggleDelete);
                ui.close();
            }
            if ui.button("Report on selection").clicked() {
                events.push(ClusterEvent::ReportSelection);
                ui.close();
            }
            ui.separator();
            if ui.button("Clear selection").clicked() {
                self.selection.clear();
                ui.close();
            }
        });
        events
    }

    fn zoom_pan(&mut self, ui: &egui::Ui, resp: &egui::Response, rect: Rect) {
        if let Some(p) = resp.hover_pos() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                let anchor = self.to_layout(rect, p);
                self.scale *= (scroll * 0.002).exp();
                let after = self.to_layout(rect, p);
                self.center[0] += anchor[0] - after[0];
                self.center[1] += anchor[1] - after[1];
            }
        }
        let space = ui.input(|i| i.key_down(egui::Key::Space));
        if resp.dragged_by(PointerButton::Middle)
            || (space && resp.dragged_by(PointerButton::Primary))
        {
            let d = resp.drag_delta() / self.scale;
            self.center[0] -= d.x;
            self.center[1] -= d.y;
            if space {
                self.lasso.clear();
            }
        }
    }

    fn paint(
        &self,
        painter: &egui::Painter,
        rect: Rect,
        ws: &Workspace,
        visible: &[(usize, [f32; 2], i32)],
        hovered: Option<usize>,
    ) {
        let ds = &ws.project.dataset;
        let mut mesh = Mesh::default();
        let mut sel_mesh = Mesh::default();
        let half = Vec2::splat(POINT_R);
        for &(i, pos, label) in visible {
            let c = self.to_screen(rect, pos);
            if !rect.expand(POINT_R).contains(c) {
                continue;
            }
            let color = if ws.is_frame_deleted_idx(i) {
                DELETED_COLOR
            } else {
                match self.color_by {
                    ColorBy::Split => split_color(ws.split_of(i)),
                    ColorBy::Cluster => cluster_color(label),
                }
            };
            if self.selection.contains(&ds.images()[i].id) {
                sel_mesh.add_colored_rect(
                    Rect::from_center_size(c, half * 2.0 + Vec2::splat(3.0)),
                    SELECT_COLOR,
                );
                sel_mesh.add_colored_rect(Rect::from_center_size(c, half * 2.0), color);
            } else {
                mesh.add_colored_rect(Rect::from_center_size(c, half * 2.0), color);
            }
        }
        painter.add(Shape::mesh(mesh));
        painter.add(Shape::mesh(sel_mesh));
        for i in [hovered, self.focus].into_iter().flatten() {
            if let Some((pos, _)) = self.points.get(i).copied().flatten() {
                let c = self.to_screen(rect, pos);
                painter.circle_stroke(c, PICK_R, Stroke::new(1.5, Color32::WHITE));
            }
        }
        if self.lasso.len() > 1 {
            let mut pts: Vec<Pos2> = self.lasso.iter().map(|p| self.to_screen(rect, *p)).collect();
            pts.push(pts[0]);
            painter.add(Shape::line(pts, Stroke::new(1.5, SELECT_COLOR)));
        }
        if self.color_by == ColorBy::Split {
            let mut y = rect.top() + 10.0;
            for s in Split::ALL {
                painter.rect_filled(
                    Rect::from_min_size(Pos2::new(rect.left() + 10.0, y), Vec2::splat(10.0)),
                    2.0,
                    split_color(Some(s)),
                );
                painter.text(
                    Pos2::new(rect.left() + 26.0, y + 5.0),
                    Align2::LEFT_CENTER,
                    s.name(),
                    FontId::proportional(13.0),
                    Color32::LIGHT_GRAY,
                );
                y += 18.0;
            }
        }
    }

    /// Thumbnails of the selected frames (virtualized rows).
    fn strip(
        &mut self,
        ui: &mut egui::Ui,
        ws: &Workspace,
        thumbs: &mut Thumbs,
        selected: &[usize],
        events: &mut Vec<ClusterEvent>,
    ) {
        let gap = 4.0;
        let width = ui.available_width();
        let cols = ((width + gap) / (STRIP_THUMB + gap)).floor().max(1.0) as usize;
        let rows = selected.len().div_ceil(cols);
        let cell = Vec2::new(STRIP_THUMB, STRIP_THUMB * 0.6);
        let mut focus = None;
        egui::ScrollArea::vertical().auto_shrink(false).show_rows(ui, cell.y, rows, |ui, range| {
            let mut jobs = Vec::new();
            for row in range {
                let (row_rect, _) =
                    ui.allocate_exact_size(Vec2::new(width, cell.y), Sense::hover());
                for c in 0..cols {
                    let Some(&i) = selected.get(row * cols + c) else { break };
                    let r = Rect::from_min_size(
                        row_rect.min + Vec2::new(c as f32 * (cell.x + gap), 0.0),
                        cell,
                    );
                    let resp = ui.interact(r, ui.id().with(("strip", i)), Sense::click());
                    let job = frame_job(ws, i);
                    let painter = ui.painter();
                    painter.rect_filled(r, 2.0, Color32::from_gray(28));
                    if let Some(Slot::Ready(tex)) = thumbs.get(job.id()) {
                        let size = tex.size_vec2();
                        let s = (r.width() / size.x).min(r.height() / size.y);
                        let img = Rect::from_center_size(r.center(), size * s);
                        painter.image(
                            tex.id(),
                            img,
                            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                            Color32::WHITE,
                        );
                    }
                    jobs.push(job);
                    let split = ws.split_of(i);
                    painter.rect_filled(
                        Rect::from_min_size(r.min + Vec2::splat(3.0), Vec2::splat(8.0)),
                        1.0,
                        split_color(split),
                    );
                    if ws.is_frame_deleted_idx(i) {
                        painter.text(
                            r.right_top() + Vec2::new(-4.0, 2.0),
                            Align2::RIGHT_TOP,
                            "F",
                            FontId::proportional(16.0),
                            Color32::from_rgb(240, 70, 60),
                        );
                    }
                    if resp.hovered() {
                        focus = Some(i);
                        painter.rect_stroke(
                            r,
                            2.0,
                            Stroke::new(1.5, Color32::WHITE),
                            StrokeKind::Inside,
                        );
                    }
                    if resp.double_clicked() {
                        events.push(ClusterEvent::Open(i));
                    }
                    resp.on_hover_text(&ws.project.dataset.images()[i].file_name);
                }
            }
            thumbs.want(jobs);
        });
        self.focus = focus;
    }
}

/// `train 4271 (86.5%) · val 298 (6.0%) · …` for counts (train, val, test, none).
pub fn split_shares(counts: [usize; 4]) -> String {
    let total: usize = counts.iter().sum();
    let share = |n: usize| if total == 0 { 0.0 } else { 100.0 * n as f64 / total as f64 };
    let mut parts: Vec<String> = Split::ALL
        .iter()
        .map(|s| {
            let n = counts[*s as usize];
            format!("{} {n} ({:.1}%)", s.name(), share(n))
        })
        .collect();
    if counts[3] > 0 {
        parts.push(format!("no split {} ({:.1}%)", counts[3], share(counts[3])));
    }
    parts.join(" · ")
}

/// Whole-frame thumbnail job.
fn frame_job(ws: &Workspace, image_idx: usize) -> Job {
    let ds = &ws.project.dataset;
    let img = &ds.images()[image_idx];
    let region = Region { x: 0, y: 0, w: img.width.max(1), h: img.height.max(1) };
    Job::Thumb { path: ds.image_path(img), region, max_side: STRIP_SIDE }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_shares_format() {
        assert_eq!(split_shares([3, 1, 0, 0]), "train 3 (75.0%) · val 1 (25.0%) · test 0 (0.0%)");
        assert!(split_shares([0, 0, 0, 2]).ends_with("no split 2 (100.0%)"));
        assert!(split_shares([0; 4]).starts_with("train 0 (0.0%)"));
    }

    #[test]
    fn colors_are_distinct_and_noise_is_grey() {
        let colors: std::collections::HashSet<_> = (0..20).map(cluster_color).collect();
        assert_eq!(colors.len(), 20);
        assert_eq!(cluster_color(-1), NOISE_COLOR);
        assert_ne!(split_color(Some(Split::Train)), split_color(Some(Split::Val)));
    }
}
