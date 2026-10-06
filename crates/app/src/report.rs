//! Dataset report: split balance, box statistics, a model input size guide,
//! similarity between splits and per-split model results, as a
//! self-contained HTML page with SVG plots (no external resources).

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;

use fastlabel_core::{BBox, Split};

/// Split slots: train, val, test, no split.
const SLOTS: usize = 4;
const SLOT_NAMES: [&str; SLOTS] = ["train", "val", "test", "no split"];
const SLOT_COLORS: [&str; SLOTS] = ["#468cf0", "#f5961e", "#be5ae6", "#969696"];
/// Candidate model input sizes for the size guide.
pub const IMGSZ: [u32; 9] = [320, 480, 640, 800, 960, 1024, 1280, 1536, 1920];
/// Boxes whose short side is below this at the model input are hard to detect
/// (smallest YOLO stride is 8).
const SMALL_AT_INPUT: f64 = 10.0;
const NEAR_DUPLICATE: f32 = 0.95;
const VERY_SIMILAR: f32 = 0.90;
const HEAT_W: usize = 24;
const HEAT_H: usize = 16;

fn slot(split: Option<Split>) -> usize {
    split.map_or(3, |s| s as usize)
}

/// Model result for one GT box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BoxEval {
    Unpredicted,
    Paired(f64),
    Missed,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReportBox {
    pub category: String,
    pub bbox: BBox,
    pub eval: BoxEval,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReportFrame {
    pub file_name: String,
    pub split: Option<Split>,
    pub width: u32,
    pub height: u32,
    pub boxes: Vec<ReportBox>,
    /// Embedding key (similarity section).
    pub key: u64,
    pub cluster: Option<i32>,
    /// Part of the report scope (other frames only serve as neighbours).
    pub in_scope: bool,
    /// False positives on the frame, `None` when the model was not run on it.
    pub fp: Option<usize>,
}

pub struct ReportInput {
    pub title: String,
    pub dataset: String,
    /// Every frame of the dataset (out-of-scope ones for the similarity).
    pub frames: Vec<ReportFrame>,
    pub notes: Vec<String>,
}

/// Nearest frame of another split, for val / test frames in scope.
#[derive(Clone, Debug, PartialEq)]
pub struct Neighbour {
    pub frame: usize,
    pub other: usize,
    pub cos: f32,
}

/// For each val / test frame in scope, the most similar frame of any other
/// split (cosine of unit vectors). Frames without a vector are skipped.
pub fn cross_split_neighbours(
    frames: &[ReportFrame],
    vector: &(dyn Fn(u64) -> Option<Vec<f32>> + Sync),
) -> Vec<Neighbour> {
    let vecs: Vec<Option<Vec<f32>>> = frames.iter().map(|f| vector(f.key)).collect();
    let queries: Vec<usize> = (0..frames.len())
        .filter(|&i| {
            let f = &frames[i];
            f.in_scope && matches!(f.split, Some(Split::Val | Split::Test)) && vecs[i].is_some()
        })
        .collect();
    let threads = std::thread::available_parallelism().map_or(2, |n| n.get()).max(1);
    let chunk = queries.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        let handles: Vec<_> = queries
            .chunks(chunk)
            .map(|part| {
                let vecs = &vecs;
                s.spawn(move || {
                    part.iter()
                        .filter_map(|&i| {
                            let q = vecs[i].as_ref()?;
                            let (other, cos) = (0..frames.len())
                                .filter(|&j| frames[j].split != frames[i].split)
                                .filter_map(|j| {
                                    let v = vecs[j].as_ref()?;
                                    Some((j, q.iter().zip(v).map(|(a, b)| a * b).sum::<f32>()))
                                })
                                .max_by(|a, b| a.1.total_cmp(&b.1))?;
                            Some(Neighbour { frame: i, other, cos })
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().expect("similarity thread")).collect()
    })
}

/// `q`-quantile (0..=1) of sorted values.
fn quantile(sorted: &[f64], q: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let i = ((sorted.len() - 1) as f64 * q).round() as usize;
    Some(sorted[i])
}

/// Per input size: (imgsz, p5 / median short side at input, share of boxes
/// below `SMALL_AT_INPUT`).
pub fn input_size_guide(frames: &[&ReportFrame]) -> Vec<(u32, f64, f64, f64)> {
    IMGSZ
        .iter()
        .map(|&sz| {
            let mut sides: Vec<f64> = frames
                .iter()
                .flat_map(|f| {
                    let scale = f64::from(sz) / f64::from(f.width.max(f.height).max(1));
                    f.boxes.iter().map(move |b| b.bbox.w.min(b.bbox.h) * scale)
                })
                .collect();
            sides.sort_by(f64::total_cmp);
            let small = sides.iter().filter(|s| **s < SMALL_AT_INPUT).count();
            let share = if sides.is_empty() { 0.0 } else { small as f64 / sides.len() as f64 };
            (sz, quantile(&sides, 0.05).unwrap_or(0.0), quantile(&sides, 0.5).unwrap_or(0.0), share)
        })
        .collect()
}

/// Smallest input size keeping at most 2 % of boxes below `SMALL_AT_INPUT`
/// at the input (never above the largest native frame side).
pub fn recommend_imgsz(guide: &[(u32, f64, f64, f64)], max_native: u32) -> Option<u32> {
    let capped: Vec<_> = guide.iter().filter(|g| g.0 <= max_native.max(IMGSZ[0])).collect();
    capped.iter().find(|g| g.3 <= 0.02).or(capped.last()).map(|g| g.0)
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

fn pct(n: usize, total: usize) -> String {
    if total == 0 { "–".into() } else { format!("{:.1}%", 100.0 * n as f64 / total as f64) }
}

/// Grouped bar chart: one group per label, one bar per series.
fn bars(labels: &[String], series: &[(usize, Vec<f64>)], percent: bool) -> String {
    let (w, h, left, bottom, top) = (720.0, 240.0, 48.0, 40.0, 12.0);
    let max = series.iter().flat_map(|s| s.1.iter().copied()).fold(0.0f64, f64::max).max(1e-9);
    let plot_w = w - left - 8.0;
    let plot_h = h - bottom - top;
    let group = plot_w / labels.len().max(1) as f64;
    let bar = (group * 0.8) / series.len().max(1) as f64;
    let mut svg = format!(r#"<svg viewBox="0 0 {w} {h}" class="chart">"#);
    for t in 0..=4 {
        let v = max * f64::from(t) / 4.0;
        let y = top + plot_h - plot_h * f64::from(t) / 4.0;
        let label = if percent { format!("{:.0}%", v * 100.0) } else { format!("{v:.0}") };
        let _ = write!(
            svg,
            r#"<line x1="{left}" x2="{}" y1="{y}" y2="{y}" class="grid"/><text x="{}" y="{}" class="tick" text-anchor="end">{label}</text>"#,
            w - 8.0,
            left - 4.0,
            y + 4.0
        );
    }
    for (g, label) in labels.iter().enumerate() {
        let x0 = left + g as f64 * group + group * 0.1;
        for (k, (slot, values)) in series.iter().enumerate() {
            let v = values.get(g).copied().unwrap_or(0.0);
            let bh = plot_h * v / max;
            let x = x0 + k as f64 * bar;
            let tip = if percent { format!("{:.1}%", v * 100.0) } else { format!("{v:.0}") };
            let _ = write!(
                svg,
                r#"<rect x="{x:.1}" y="{:.1}" width="{:.1}" height="{bh:.1}" fill="{}"><title>{} {}: {tip}</title></rect>"#,
                top + plot_h - bh,
                (bar - 1.0).max(1.0),
                SLOT_COLORS[*slot],
                SLOT_NAMES[*slot],
                esc(label)
            );
        }
        let _ = write!(
            svg,
            r#"<text x="{:.1}" y="{}" class="tick" text-anchor="middle">{}</text>"#,
            x0 + group * 0.4,
            h - bottom + 16.0,
            esc(label)
        );
    }
    svg.push_str("</svg>");
    svg
}

fn legend(slots: &[usize]) -> String {
    let mut s = String::from(r#"<div class="legend">"#);
    for &k in slots {
        let _ = write!(
            s,
            r#"<span><i style="background:{}"></i>{}</span>"#,
            SLOT_COLORS[k], SLOT_NAMES[k]
        );
    }
    s.push_str("</div>");
    s
}

/// Density of box centers (normalized image coordinates).
fn heatmap(grid: &[f64], color: &str, title: &str) -> String {
    let max = grid.iter().copied().fold(0.0f64, f64::max).max(1e-9);
    let (cw, ch) = (10.0, 10.0);
    let mut svg = format!(
        r#"<figure><svg viewBox="0 0 {} {}" class="heat"><rect width="100%" height="100%" class="heatbg"/>"#,
        HEAT_W as f64 * cw,
        HEAT_H as f64 * ch
    );
    for (i, v) in grid.iter().enumerate() {
        if *v <= 0.0 {
            continue;
        }
        let (x, y) = ((i % HEAT_W) as f64 * cw, (i / HEAT_W) as f64 * ch);
        let _ = write!(
            svg,
            r#"<rect x="{x}" y="{y}" width="{cw}" height="{ch}" fill="{color}" fill-opacity="{:.3}"/>"#,
            (v / max).sqrt()
        );
    }
    let _ = write!(svg, "</svg><figcaption>{}</figcaption></figure>", esc(title));
    svg
}

/// Value bins `[edges[i], edges[i + 1])`, last bin open.
fn bin_of(v: f64, edges: &[f64]) -> usize {
    edges.iter().rposition(|e| v >= *e).unwrap_or(0)
}

fn bin_labels(edges: &[f64], fmt: impl Fn(f64) -> String) -> Vec<String> {
    edges
        .iter()
        .enumerate()
        .map(|(i, e)| match edges.get(i + 1) {
            Some(n) => format!("{}–{}", fmt(*e), fmt(*n)),
            None => format!("≥{}", fmt(*e)),
        })
        .collect()
}

/// Per-slot normalized histogram of a box value.
fn box_hist(
    by_slot: &[Vec<&ReportFrame>; SLOTS],
    present: &[usize],
    edges: &[f64],
    value: impl Fn(&ReportFrame, &ReportBox) -> f64,
) -> Vec<(usize, Vec<f64>)> {
    present
        .iter()
        .map(|&k| {
            let mut counts = vec![0.0; edges.len()];
            let mut n = 0.0;
            for f in &by_slot[k] {
                for b in &f.boxes {
                    counts[bin_of(value(f, b), edges)] += 1.0;
                    n += 1.0;
                }
            }
            (k, counts.into_iter().map(|c| if n > 0.0 { c / n } else { 0.0 }).collect())
        })
        .collect()
}

/// Renders the report; `neighbours` from [`cross_split_neighbours`] (empty
/// when no embeddings).
pub fn render(input: &ReportInput, neighbours: &[Neighbour]) -> String {
    let scope: Vec<&ReportFrame> = input.frames.iter().filter(|f| f.in_scope).collect();
    let mut by_slot: [Vec<&ReportFrame>; SLOTS] = Default::default();
    for f in &scope {
        by_slot[slot(f.split)].push(f);
    }
    let present: Vec<usize> = (0..SLOTS).filter(|&k| !by_slot[k].is_empty()).collect();
    let total_images = scope.len();
    let total_boxes: usize = scope.iter().map(|f| f.boxes.len()).sum();
    let mut h = String::new();
    let _ = write!(
        h,
        "<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>{}</title><style>{}</style></head><body><main>",
        esc(&input.title),
        CSS
    );
    let _ = write!(
        h,
        "<h1>{}</h1><p class=\"sub\">{} · {} frames · {} boxes</p>",
        esc(&input.title),
        esc(&input.dataset),
        total_images,
        total_boxes
    );
    for n in &input.notes {
        let _ = write!(h, "<p class=\"note\">{}</p>", esc(n));
    }
    if scope.is_empty() {
        h.push_str("<p>No frames in scope.</p></main></body></html>");
        return h;
    }

    // Split balance.
    h.push_str("<h2>Split balance</h2><table><tr><th>split</th><th>images</th><th>% images</th><th>boxes</th><th>% boxes</th><th>images without boxes</th><th>boxes / image</th><th>median</th><th>max</th></tr>");
    for &k in &present {
        let frames = &by_slot[k];
        let boxes: usize = frames.iter().map(|f| f.boxes.len()).sum();
        let empty = frames.iter().filter(|f| f.boxes.is_empty()).count();
        let mut per: Vec<f64> = frames.iter().map(|f| f.boxes.len() as f64).collect();
        per.sort_by(f64::total_cmp);
        let _ = write!(
            h,
            "<tr><td><i class=\"dot\" style=\"background:{}\"></i>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{} ({})</td><td>{:.2}</td><td>{:.0}</td><td>{:.0}</td></tr>",
            SLOT_COLORS[k],
            SLOT_NAMES[k],
            frames.len(),
            pct(frames.len(), total_images),
            boxes,
            pct(boxes, total_boxes),
            empty,
            pct(empty, frames.len()),
            boxes as f64 / frames.len() as f64,
            quantile(&per, 0.5).unwrap_or(0.0),
            per.last().copied().unwrap_or(0.0)
        );
    }
    h.push_str("</table>");
    let mut cats: BTreeMap<&str, [usize; SLOTS]> = BTreeMap::new();
    for f in &scope {
        for b in &f.boxes {
            cats.entry(&b.category).or_default()[slot(f.split)] += 1;
        }
    }
    if cats.len() > 1 {
        h.push_str("<h3>Boxes per category</h3><table><tr><th>category</th>");
        for &k in &present {
            let _ = write!(h, "<th>{}</th>", SLOT_NAMES[k]);
        }
        h.push_str("</tr>");
        for (c, counts) in &cats {
            let _ = write!(h, "<tr><td>{}</td>", esc(c));
            for &k in &present {
                let _ = write!(h, "<td>{}</td>", counts[k]);
            }
            h.push_str("</tr>");
        }
        h.push_str("</table>");
    }

    // Boxes per image.
    h.push_str("<h2>Boxes per image</h2><p class=\"help\">Share of each split's images with that many boxes. Similar shapes mean the splits see similar crowd densities.</p>");
    let max_count = 10usize;
    let labels: Vec<String> = (0..=max_count)
        .map(|n| if n == max_count { format!("{n}+") } else { n.to_string() })
        .collect();
    let series: Vec<(usize, Vec<f64>)> = present
        .iter()
        .map(|&k| {
            let mut c = vec![0.0; max_count + 1];
            for f in &by_slot[k] {
                c[f.boxes.len().min(max_count)] += 1.0;
            }
            let n = by_slot[k].len() as f64;
            (k, c.into_iter().map(|v| v / n).collect())
        })
        .collect();
    h.push_str(&legend(&present));
    h.push_str(&bars(&labels, &series, true));

    // Box centers.
    h.push_str("<h2>Box centers</h2><p class=\"help\">Where boxes are in the frame (x, y normalized to the image), per split. The mean center and spread are below each map.</p><div class=\"row\">");
    for &k in &present {
        let mut grid = vec![0.0; HEAT_W * HEAT_H];
        let (mut sx, mut sy, mut sxx, mut syy, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for f in &by_slot[k] {
            for b in &f.boxes {
                let cx =
                    ((b.bbox.x + b.bbox.w / 2.0) / f64::from(f.width.max(1))).clamp(0.0, 0.999);
                let cy =
                    ((b.bbox.y + b.bbox.h / 2.0) / f64::from(f.height.max(1))).clamp(0.0, 0.999);
                grid[(cy * HEAT_H as f64) as usize * HEAT_W + (cx * HEAT_W as f64) as usize] += 1.0;
                sx += cx;
                sy += cy;
                sxx += cx * cx;
                syy += cy * cy;
                n += 1.0;
            }
        }
        let caption = if n > 0.0 {
            let (mx, my) = (sx / n, sy / n);
            format!(
                "{} · center ({:.2}, {:.2}) · spread ({:.2}, {:.2})",
                SLOT_NAMES[k],
                mx,
                my,
                (sxx / n - mx * mx).max(0.0).sqrt(),
                (syy / n - my * my).max(0.0).sqrt()
            )
        } else {
            format!("{} · no boxes", SLOT_NAMES[k])
        };
        h.push_str(&heatmap(&grid, SLOT_COLORS[k], &caption));
    }
    h.push_str("</div>");

    // Box sizes.
    h.push_str("<h2>Box sizes</h2><p class=\"help\">Short side of the boxes in original image pixels, share of each split's boxes.</p>");
    let edges = [0.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 256.0];
    h.push_str(&legend(&present));
    h.push_str(&bars(
        &bin_labels(&edges, |v| format!("{v:.0}")),
        &box_hist(&by_slot, &present, &edges, |_, b| b.bbox.w.min(b.bbox.h)),
        true,
    ));
    h.push_str("<h3>Box height relative to the image height</h3>");
    let edges = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3];
    h.push_str(&bars(
        &bin_labels(&edges, |v| format!("{}%", (v * 1000.0).round() / 10.0)),
        &box_hist(&by_slot, &present, &edges, |f, b| b.bbox.h / f64::from(f.height.max(1))),
        true,
    ));
    h.push_str("<h3>Aspect ratio (width / height)</h3>");
    let edges = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0];
    h.push_str(&bars(
        &bin_labels(&edges, |v| format!("{v}")),
        &box_hist(&by_slot, &present, &edges, |_, b| {
            if b.bbox.h > 0.0 { b.bbox.w / b.bbox.h } else { 0.0 }
        }),
        true,
    ));

    // Image sizes.
    h.push_str("<h2>Image sizes</h2><table><tr><th>resolution</th>");
    for &k in &present {
        let _ = write!(h, "<th>{}</th>", SLOT_NAMES[k]);
    }
    h.push_str("<th>total</th></tr>");
    let mut sizes: HashMap<(u32, u32), [usize; SLOTS]> = HashMap::new();
    for f in &scope {
        sizes.entry((f.width, f.height)).or_default()[slot(f.split)] += 1;
    }
    let mut sizes: Vec<_> = sizes.into_iter().collect();
    sizes.sort_by_key(|(s, c)| (std::cmp::Reverse(c.iter().sum::<usize>()), *s));
    for ((w, ht), c) in sizes.iter().take(12) {
        let orient = match w.cmp(ht) {
            std::cmp::Ordering::Greater => "landscape",
            std::cmp::Ordering::Less => "portrait",
            std::cmp::Ordering::Equal => "square",
        };
        let _ = write!(h, "<tr><td>{w} × {ht} <span class=\"help\">{orient}</span></td>");
        for &k in &present {
            let _ = write!(h, "<td>{}</td>", c[k]);
        }
        let _ = write!(h, "<td>{}</td></tr>", c.iter().sum::<usize>());
    }
    if sizes.len() > 12 {
        let _ = write!(
            h,
            "<tr><td colspan=\"9\" class=\"help\">… {} more resolutions</td></tr>",
            sizes.len() - 12
        );
    }
    h.push_str("</table>");

    // Input size guide.
    let guide = input_size_guide(&scope);
    let max_native = scope.iter().map(|f| f.width.max(f.height)).max().unwrap_or(0);
    let rec = recommend_imgsz(&guide, max_native);
    h.push_str("<h2>Model input size guide</h2><p class=\"help\">Frames are letterboxed so their long side equals the input size. Short side of the boxes once scaled; boxes under 10 px at the input are hard for YOLO (smallest stride 8). Compute grows with the square of the size.</p><table><tr><th>imgsz</th><th>p5 short side</th><th>median short side</th><th>boxes &lt; 10 px</th><th>relative compute</th></tr>");
    for (sz, p5, med, share) in &guide {
        let cls = if Some(*sz) == rec { " class=\"rec\"" } else { "" };
        let _ = write!(
            h,
            "<tr{cls}><td>{sz}{}</td><td>{p5:.1} px</td><td>{med:.1} px</td><td>{:.1}%</td><td>{:.2}×</td></tr>",
            if *sz > max_native { " <span class=\"help\">(upscales)</span>" } else { "" },
            share * 100.0,
            (f64::from(*sz) / 640.0).powi(2)
        );
    }
    h.push_str("</table>");
    if let Some(r) = rec {
        let _ = write!(
            h,
            "<p class=\"rec-text\">Suggested input size: <b>{r}</b> (smallest size with at most 2% of boxes under 10 px at the input). Use rectangular inference (dynamic export) to avoid padding landscape frames.</p>"
        );
    }

    // Quality flags.
    h.push_str("<h2>Box quality flags</h2><table><tr><th>split</th><th>tiny (&lt; 8 px)</th><th>touching the border</th><th>extreme aspect (w/h &gt; 3 or &lt; 0.25)</th></tr>");
    for &k in &present {
        let (mut tiny, mut border, mut aspect, mut n) = (0, 0, 0, 0);
        for f in &by_slot[k] {
            for b in &f.boxes {
                let bb = b.bbox;
                n += 1;
                tiny += usize::from(bb.w.min(bb.h) < 8.0);
                border += usize::from(
                    bb.x <= 1.0
                        || bb.y <= 1.0
                        || bb.x2() >= f64::from(f.width) - 1.0
                        || bb.y2() >= f64::from(f.height) - 1.0,
                );
                let r = if bb.h > 0.0 { bb.w / bb.h } else { 0.0 };
                aspect += usize::from(!(0.25..=3.0).contains(&r));
            }
        }
        let _ = write!(
            h,
            "<tr><td>{}</td><td>{tiny} ({})</td><td>{border} ({})</td><td>{aspect} ({})</td></tr>",
            SLOT_NAMES[k],
            pct(tiny, n),
            pct(border, n),
            pct(aspect, n)
        );
    }
    h.push_str("</table>");

    // Model results per split.
    let predicted = scope.iter().any(|f| f.fp.is_some());
    if predicted {
        h.push_str("<h2>Model results per split</h2><p class=\"help\">From predictions.json on predicted frames (no thresholds: a GT box is paired with its best-overlapping prediction). Train results are optimistic if the model was trained on them; a large gap between val and test means one of them is not representative.</p><table><tr><th>split</th><th>GT boxes</th><th>paired (recall)</th><th>missed</th><th>false positives</th><th>precision</th><th>mean IoU</th><th>paired IoU &lt; 0.5</th></tr>");
        for &k in &present {
            let (mut gt, mut paired, mut missed, mut fp, mut ious) = (0, 0, 0, 0, Vec::new());
            for f in by_slot[k].iter().filter(|f| f.fp.is_some()) {
                fp += f.fp.unwrap_or(0);
                for b in &f.boxes {
                    gt += 1;
                    match b.eval {
                        BoxEval::Paired(iou) => {
                            paired += 1;
                            ious.push(iou);
                        }
                        BoxEval::Missed => missed += 1,
                        BoxEval::Unpredicted => {}
                    }
                }
            }
            let mean =
                if ious.is_empty() { 0.0 } else { ious.iter().sum::<f64>() / ious.len() as f64 };
            let loose = ious.iter().filter(|i| **i < 0.5).count();
            let _ = write!(
                h,
                "<tr><td>{}</td><td>{gt}</td><td>{paired} ({})</td><td>{missed}</td><td>{fp}</td><td>{}</td><td>{mean:.3}</td><td>{loose} ({})</td></tr>",
                SLOT_NAMES[k],
                pct(paired, gt),
                pct(paired, paired + fp),
                pct(loose, paired)
            );
        }
        h.push_str("</table>");
    }

    // Similarity between splits.
    h.push_str("<h2>Similarity between splits</h2>");
    if neighbours.is_empty() {
        h.push_str("<p class=\"help\">Compute the embeddings in cluster mode to see near-duplicate frames across splits.</p>");
    } else {
        let _ = write!(
            h,
            "<p class=\"help\">For each val / test frame, cosine similarity (DINOv2 embeddings) to the most similar frame of another split. ≥ {NEAR_DUPLICATE}: near duplicate (consecutive video frames, same scene); such frames leak: the metric on that split overestimates generalization.</p><table><tr><th>split</th><th>frames</th><th>near duplicates ≥ {NEAR_DUPLICATE}</th><th>very similar ≥ {VERY_SIMILAR}</th><th>median similarity</th></tr>"
        );
        for k in [1usize, 2] {
            let mut cos: Vec<f64> = neighbours
                .iter()
                .filter(|n| slot(input.frames[n.frame].split) == k)
                .map(|n| f64::from(n.cos))
                .collect();
            if cos.is_empty() {
                continue;
            }
            cos.sort_by(f64::total_cmp);
            let near = cos.iter().filter(|c| **c >= f64::from(NEAR_DUPLICATE)).count();
            let similar = cos.iter().filter(|c| **c >= f64::from(VERY_SIMILAR)).count();
            let _ = write!(
                h,
                "<tr><td>{}</td><td>{}</td><td>{near} ({})</td><td>{similar} ({})</td><td>{:.3}</td></tr>",
                SLOT_NAMES[k],
                cos.len(),
                pct(near, cos.len()),
                pct(similar, cos.len()),
                quantile(&cos, 0.5).unwrap_or(0.0)
            );
        }
        h.push_str("</table>");
        let edges = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98];
        let series: Vec<(usize, Vec<f64>)> = [1usize, 2]
            .into_iter()
            .filter_map(|k| {
                let vals: Vec<f64> = neighbours
                    .iter()
                    .filter(|n| slot(input.frames[n.frame].split) == k)
                    .map(|n| f64::from(n.cos))
                    .collect();
                if vals.is_empty() {
                    return None;
                }
                let mut c = vec![0.0; edges.len()];
                for v in &vals {
                    c[bin_of(*v, &edges)] += 1.0;
                }
                Some((k, c.into_iter().map(|x| x / vals.len() as f64).collect()))
            })
            .collect();
        h.push_str(&legend(&series.iter().map(|s| s.0).collect::<Vec<_>>()));
        h.push_str(&bars(&bin_labels(&edges, |v| format!("{v}")), &series, true));
        let mut top: Vec<&Neighbour> = neighbours.iter().collect();
        top.sort_by(|a, b| b.cos.total_cmp(&a.cos));
        h.push_str("<h3>Most similar cross-split pairs</h3><table><tr><th>frame</th><th>most similar frame in another split</th><th>cosine</th></tr>");
        for n in top.iter().take(15) {
            let _ = write!(
                h,
                "<tr><td>{}</td><td>{}</td><td>{:.3}</td></tr>",
                esc(&input.frames[n.frame].file_name),
                esc(&input.frames[n.other].file_name),
                n.cos
            );
        }
        h.push_str("</table>");
    }

    // Clusters × splits.
    let mut clusters: BTreeMap<i32, [usize; SLOTS]> = BTreeMap::new();
    for f in &scope {
        if let Some(c) = f.cluster {
            clusters.entry(c).or_default()[slot(f.split)] += 1;
        }
    }
    if !clusters.is_empty() {
        let mut rows: Vec<(i32, [usize; SLOTS])> = clusters.into_iter().collect();
        rows.sort_by_key(|(c, n)| (std::cmp::Reverse(n.iter().sum::<usize>()), *c));
        let only_train: usize =
            rows.iter().filter(|(c, n)| *c >= 0 && n[1] == 0 && n[2] == 0).map(|(_, n)| n[0]).sum();
        let train_total: usize = rows.iter().map(|(_, n)| n[0]).sum();
        let _ = write!(
            h,
            "<h2>Clusters × splits</h2><p class=\"help\">Clusters from cluster mode (similar scenes). A cluster with only train frames is a scene the val / test metrics never measure; a cluster split across train and val / test is fine only if its frames are not near duplicates. {} of train frames ({}) are in clusters absent from val and test.</p><table><tr><th>cluster</th><th>frames</th>",
            only_train,
            pct(only_train, train_total)
        );
        for &k in &present {
            let _ = write!(h, "<th>{}</th>", SLOT_NAMES[k]);
        }
        h.push_str("<th>mix</th></tr>");
        for (c, n) in rows.iter().take(40) {
            let total: usize = n.iter().sum();
            let name = if *c < 0 { "noise".to_string() } else { format!("#{c}") };
            let _ = write!(h, "<tr><td>{name}</td><td>{total}</td>");
            for &k in &present {
                let _ =
                    write!(h, "<td>{} <span class=\"help\">{}</span></td>", n[k], pct(n[k], total));
            }
            h.push_str("<td><div class=\"mix\">");
            for &k in &present {
                let _ = write!(
                    h,
                    "<i style=\"width:{:.1}%;background:{}\"></i>",
                    100.0 * n[k] as f64 / total.max(1) as f64,
                    SLOT_COLORS[k]
                );
            }
            h.push_str("</div></td></tr>");
        }
        h.push_str("</table>");
    }
    h.push_str("</main></body></html>");
    h
}

/// Writes the report to `dir/report-<time>-<slug>.html`, using the cached
/// embeddings (if any) for the similarity section.
pub fn write(
    input: &ReportInput,
    embeddings: &crate::embeddings::Embeddings,
    dir: &std::path::Path,
    slug: &str,
) -> std::io::Result<std::path::PathBuf> {
    let neighbours = if embeddings.len() > 0 {
        cross_split_neighbours(&input.frames, &|k| embeddings.get(k).map(<[f32]>::to_vec))
    } else {
        Vec::new()
    };
    let html = render(input, &neighbours);
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!(
        "report-{}-{slug}.html",
        fastlabel_core::dates::compact(fastlabel_core::dates::now())
    ));
    std::fs::write(&path, html)?;
    Ok(path)
}

const CSS: &str = r#"
:root { --bg:#ffffff; --fg:#1d1d1f; --muted:#6e6e73; --grid:#e5e5ea; --card:#f5f5f7; --heat:#f0f0f3; --accent:#c48a00; }
@media (prefers-color-scheme: dark) { :root { --bg:#161617; --fg:#f2f2f2; --muted:#9a9aa0; --grid:#2c2c2e; --card:#1f1f21; --heat:#232325; --accent:#ffd60a; } }
body { background:var(--bg); color:var(--fg); font:14px/1.5 system-ui, sans-serif; margin:0; }
main { max-width:980px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:26px; margin:0 0 4px; } h2 { font-size:19px; margin:36px 0 8px; border-bottom:1px solid var(--grid); padding-bottom:4px; } h3 { font-size:15px; margin:20px 0 6px; }
.sub, .help { color:var(--muted); } .note { background:var(--card); padding:6px 10px; border-radius:6px; }
table { border-collapse:collapse; margin:8px 0; font-variant-numeric:tabular-nums; display:block; overflow-x:auto; }
th, td { padding:4px 10px; text-align:right; border-bottom:1px solid var(--grid); white-space:nowrap; } th:first-child, td:first-child { text-align:left; }
tr.rec td { color:var(--accent); font-weight:600; } .rec-text b { color:var(--accent); }
.chart { width:100%; max-width:720px; height:auto; display:block; } .chart .grid { stroke:var(--grid); } .chart .tick { fill:var(--muted); font-size:11px; }
.legend span { margin-right:14px; color:var(--muted); } .legend i, .dot { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:-1px; }
.row { display:flex; flex-wrap:wrap; gap:12px; } figure { margin:0; } figcaption { color:var(--muted); font-size:12px; }
.heat { width:240px; height:160px; display:block; border-radius:4px; } .heatbg { fill:var(--heat); }
.mix { display:flex; width:140px; height:10px; border-radius:3px; overflow:hidden; background:var(--grid); } .mix i { display:block; height:100%; }
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(split: Option<Split>, boxes: &[(f64, f64)], key: u64, in_scope: bool) -> ReportFrame {
        ReportFrame {
            file_name: format!("f{key}.jpg"),
            split,
            width: 1920,
            height: 1080,
            boxes: boxes
                .iter()
                .map(|&(x, side)| ReportBox {
                    category: "bib".into(),
                    bbox: BBox::new(x, 500.0, side, side),
                    eval: BoxEval::Paired(0.8),
                })
                .collect(),
            key,
            cluster: Some(0),
            in_scope,
            fp: Some(1),
        }
    }

    #[test]
    fn neighbours_look_only_in_other_splits() {
        let frames = vec![
            frame(Some(Split::Train), &[], 1, true),
            frame(Some(Split::Val), &[], 2, true),
            frame(Some(Split::Val), &[], 3, true),
            frame(Some(Split::Test), &[], 4, false),
        ];
        let vecs: HashMap<u64, Vec<f32>> =
            [(1, vec![1.0, 0.0]), (2, vec![0.0, 1.0]), (3, vec![0.0, 1.0]), (4, vec![0.6, 0.8])]
                .into();
        let mut n = cross_split_neighbours(&frames, &|k| vecs.get(&k).cloned());
        n.sort_by_key(|n| n.frame);
        // Frame 2's twin is in val too: its neighbour is the test frame.
        assert_eq!(n.len(), 2, "out-of-scope test frame is not a query");
        assert_eq!((n[0].frame, n[0].other), (1, 3));
        assert!((n[0].cos - 0.8).abs() < 1e-6);
    }

    #[test]
    fn input_size_guide_and_recommendation() {
        // 1920-wide frames; boxes with a 20 px and a 40 px short side.
        let f = frame(Some(Split::Train), &[(0.0, 20.0), (100.0, 40.0)], 1, true);
        let guide = input_size_guide(&[&f]);
        let at = |sz| guide.iter().find(|g| g.0 == sz).unwrap();
        assert!((at(960).2 - 20.0).abs() < 1e-9, "median 20*0.5..40*0.5 rounds to the upper");
        assert_eq!(at(640).3, 0.5, "20 px → 6.7 px at 640");
        assert_eq!(at(960).3, 0.0);
        assert_eq!(recommend_imgsz(&guide, 1920), Some(960));
        assert_eq!(recommend_imgsz(&guide, 500), Some(480), "capped to the native size");
    }

    #[test]
    fn report_renders_every_section() {
        let mut frames = vec![
            frame(Some(Split::Train), &[(0.0, 20.0), (100.0, 40.0)], 1, true),
            frame(Some(Split::Val), &[(0.0, 30.0)], 2, true),
            frame(Some(Split::Test), &[], 3, true),
        ];
        frames[2].cluster = Some(-1);
        let input = ReportInput {
            title: "All frames".into(),
            dataset: "/d".into(),
            frames,
            notes: vec!["note <1>".into()],
        };
        let n = [Neighbour { frame: 1, other: 0, cos: 0.97 }];
        let html = render(&input, &n);
        for s in [
            "Split balance",
            "Boxes per image",
            "Box centers",
            "Box sizes",
            "Image sizes",
            "Model input size guide",
            "Box quality flags",
            "Model results per split",
            "Similarity between splits",
            "Clusters × splits",
            "note &lt;1&gt;",
            "<svg",
        ] {
            assert!(html.contains(s), "missing {s}");
        }
        assert!(html.contains("<td>1 (100.0%)</td>"), "the val frame is a near duplicate");
        let empty = ReportInput { frames: vec![], ..input };
        assert!(render(&empty, &[]).contains("No frames in scope"));
    }
}
