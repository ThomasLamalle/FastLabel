# FastLabel: implementation plan

Rust rebuild of the bib review tool. Functional source of truth: [`features.md`](../features.md).

## Constraints that drive the design

| Constraint | Consequence |
|---|---|
| Dev machine: 12 cores, **5 GB RAM**, **no GPU** | Bounded memory everywhere: never hold decoded full frames except the one shown; thumbnail LRU with a fixed byte budget (default 256 MB); CPU inference. |
| Plan for **100k images / ~250k boxes** | Indexed in-memory data model (Vec + id→index maps), no O(n²) passes; disk thumbnail cache; virtualized mosaic. |
| Keyboard triage must never lag | UI thread never decodes, never writes JSON, never runs inference. Every key action is an O(1) append to the edit log. |
| Stability | Pure, heavily tested core crate; atomic writes; crash journal; metadata written before images are moved. |

## Architecture

Cargo workspace:

```
crates/
  core/     fastlabel-core   — no GUI, no I/O threads. Data model, COCO/YOLO I/O, versioning,
                               predictions, IoU + matching, edit log (apply/undo), journal,
                               review state, trash, zip export.
  media/    fastlabel-media  — JPEG decode, crop extraction, thumbnail disk cache, memory LRU,
                               background worker pool, prefetch, color-uniformity metric.
  infer/    fastlabel-infer  — `Detector` trait, ONNX Runtime session, YOLO26 + RF-DETR adapters,
                               pre/post-processing, background run with progress/cancel.
  app/      fastlabel        — eframe/egui GUI (mosaic, frame mode, popup, panels) + CLI (clap).
```

### Key crates

| Need | Choice | Why |
|---|---|---|
| GUI | `eframe` / `egui` (wgpu backend) | GPU-drawn, immediate mode suits a keyboard-driven grid; `ScrollArea::show_rows` gives row virtualization for free. |
| JPEG decode | `zune-jpeg` (fallback `turbojpeg` if benchmarks demand) | Pure Rust, fast; validated in the M2 benchmark. |
| Image ops | `image`, `fast_image_resize` | Crop + SIMD downscale. |
| Parallelism | `rayon` (CPU batches), `crossbeam-channel` (job queues) | |
| Inference | `ort` 2.x (ONNX Runtime, CPU EP) | No Python; same path for YOLO26 and RF-DETR. |
| Serialization | `serde`, `serde_json` | Unknown fields preserved via `#[serde(flatten)] extra` so round-trips don't lose data. |
| Export | `zip` (stored, no recompression of JPEGs) | |
| Errors / logs | `thiserror` (libs), `anyhow` (app), `tracing` | |
| Tests | `proptest`, `insta`, `criterion`, `egui_kittest` | Property tests for IoU/matching/edit log; snapshot tests for UI. |

### Data model and edits (core)

- `Dataset { images, annotations, categories, extra }` loaded from COCO; ids preserved; split derived from `file_name`.
- `Predictions` loaded from `predictions.json`; keyed by image id; validated against image name/size (mismatch → "unpredicted").
- **Every user action is an `Op`** appended to an `EditLog`:
  `ToggleDeleteAnn`, `ToggleFrameDelete`, `ToggleAcceptPred`, `ToggleRejectPred`, `ToggleConfirmAnn`, `ReplaceWithPred`, `AddBox`, `ModifyBox`, `SetCategory`, `SetTags`.
  - Undo/redo = cursor in the log. Row actions are one grouped op.
  - Each op is also appended to `.fastlabel/journal.jsonl` (buffered, flushed on a background thread) → replayed on startup after a crash.
  - Views render `base dataset + pending ops` (cheap lookups: per-id mark sets), so the mosaic layout stays stable.
- **Save** = `apply(dataset, ops) → new dataset` (pure, tested) → write `annotations.v{N+1}.json` via temp file + fsync + rename → update `review.json` → move deleted frames to `.fastlabel/trash/v{N+1}/` → truncate journal → rebuild views.
- Matching: per image, same category, greedy highest-IoU one-to-one ≥ match IoU → TP/FN/FP + per-crop IoU; plus best-IoU per box for the FP/FN filters. Parallel over images with rayon; recomputed after save or threshold change.

### Thumbnail pipeline (media)

- Crop key = (image id, bbox, margins, thumb size). Thumbnails stored at max 256 px on the longest side.
- Workers group requested crops **by frame**: decode the JPEG once (with DCT downscale when the crop permits), cut all its crops, encode to the disk cache (`.fastlabel/cache/`, small JPEG/QOI files in sharded dirs), upload to the GPU as egui textures.
- Priority queue: visible rows first, then ±N rows of prefetch in the scroll direction; stale requests are dropped when scrolling fast.
- Margin change invalidates keys lazily (visible first). Memory LRU holds textures within the byte budget.

### Inference (infer)

```rust
trait Detector { fn detect(&self, img: &RgbImage, conf: f32) -> Vec<Detection>; fn class_names(&self) -> &[String]; }
```

- YOLO26: letterbox to the input size (1280 for exp-10). Two output layouts, chosen from the `end2end` metadata / output shape:
  - raw head `[1, 4+nc, A]` (cx, cy, w, h in input pixels + class scores) → confidence filter + class-wise NMS (exp-10 is this one);
  - end-to-end `[1, N, 6]` = x1, y1, x2, y2, score, class (no NMS).
  Class names from the ONNX `names` metadata. At 1280 on CPU expect a few hundred ms per image, so a full 5k run is tens of minutes in the background.
- RF-DETR: resize + ImageNet normalize, outputs `dets` (cx, cy, w, h normalized) + `labels` logits → sigmoid → top-k. Exact shapes are checked with a real export before coding the adapter.
- Batch run: worker thread, decode prefetch with rayon, `intra_threads` tuned for 12 cores, progress + cancel via an atomic, results written atomically to `predictions.json` only on success.

## Milestones

Each milestone ends with `cargo fmt --check && cargo clippy -D warnings && cargo test` green and a commit on `main`.

### M0: Setup
- Install the Rust toolchain (`rustup`, stable) and the system libs eframe needs on Wayland.
- Workspace skeleton, cargo aliases in `.cargo/config.toml` (`check-all`, `t`), `.gitignore`, rustfmt/clippy config, `CLAUDE.md`.
- Synthetic fixture datasets are generated by test helpers at runtime (tiny JPEGs, COCO + predictions + YOLO variants), so no binaries are committed.
- `~/Downloads/exp-10.onnx` probed (`cargo run -p fastlabel-infer --example probe`): YOLO26s, input `[1,3,1280,1280]`, **raw head** output `[1, 4+nc, 33600]` (`end2end=False`) → the adapter must run NMS; class names in the `names` metadata.

### M1: Core data (no GUI)
- COCO load/save with lossless round-trip of `annotations.v022.json` (semantic equality test).
- Version discovery (latest `vNNN` vs legacy draft), next-version naming.
- `predictions.json` load/save + validation; YOLO import (`data.yaml`, clipping, empty/missing labels).
- IoU, matching, best-IoU, TP/FP/FN classification (proptests: symmetry, bounds, one-to-one).
- `Op` / `EditLog` / undo-redo / journal replay / `apply` / review state / rejected-prediction memory (IoU ≥ 0.9).
- Atomic save + trash move + zip export.
- Bench: load + match on v1_exp0 and on a synthetic 100k-image dataset (target: < 1 s and < 3 s).

### M2: Media pipeline
- Decode, crop with margins (clipped to image), downscale, disk cache, LRU, priority worker pool.
- Bench: cold crop throughput on v1_exp0 (target ≥ 300 crops/s on 12 cores), warm cache ≥ 3000 crops/s.

### M3: Mosaic MVP (first daily-usable build)
- eframe window, open dataset (CLI arg + picker), virtualized grid, crops-per-line, H/W margins.
- Navigation (arrows / hjkl, page-up/down, home/end), selection always scrolled into view.
- D / F / G (+ Shift row variants), toggles, auto-advance, markers (D red, G green, F on all sibling crops).
- Undo / redo, journal-backed crash recovery, S (new version), Q (save + quit), status bar.
- Sorting by id / width / height, asc/desc; split badge + split filter.
- Session restore.

### M4: Predictions in the mosaic
- Load `predictions.json`; TP crops with both boxes + `GT`/`pred` labels + confidence; FP crops.
- Threshold panel (match, FP, FN), FP-only / FN-only filters, IoU and confidence sorts.
- R (replace GT with prediction), D on a prediction = reject (persisted), G on a prediction = accept.
- Reviewed state + "hide reviewed" filter.

### M5: Inference (CPU)
- `ort` session, YOLO26 adapter, validated against the Python `predictions.json` on the same images (boxes within ±1 px, same scores ±1e-3).
- Run on all frames (background, progress, cancel) and on a single frame; model path / confidence controls; class-name mapping errors reported.
- RF-DETR adapter (after a real RF-DETR ONNX export is available to test against).

### M6: Frame mode + popup editor
- Tab toggles modes; frame navigation (+ Ctrl jump to annotated frames).
- Draw / select / move / 8-handle resize, zoom + pan, D delete, 1–9 category, tags, F.
- Same editor as a popup on Enter from the mosaic; Hold-Space peek overlay; run model on this frame.

### M7: Metrics & export polish
- Color uniformity (grayscale std dev) computed in the background with progress, cached in `.fastlabel/`, sortable.
- Export-to-zip button with progress.

### M8: Hardening
- 100k-image synthetic stress test (memory stays under budget, scrolling stays smooth).
- Fault injection: kill during save → previous version intact, journal replays.
- Fuzz the COCO / YOLO / predictions loaders (`cargo fuzz`) and harden against malformed input.
- Release profile (LTO, `codegen-units = 1`), install instructions.

## Keyboard map

Everywhere: `S` save new version · `Q` save and quit (closes the popup when open) · `U` / `Ctrl+Z` undo · `Ctrl+Shift+Z` / `Ctrl+Y` redo · `Tab` / `M` mosaic ⇄ frame mode · `Ctrl+O` open dataset.

| Mosaic | Action |
|---|---|
| ← → ↑ ↓ / h l k j | Move selection |
| PgUp / PgDn, Home / End | Scroll by screen / to ends |
| D / G / F / R | Delete-or-reject / accept-or-confirm / frame delete / replace (toggles, advance to the next undecided crop; F clears D/G/R on its frame; D/G/R unmark F) |
| Shift + D / G / F | Same, from the selection to the end of the row, then first crop of the next row |
| Right click | D on the clicked crop (no advance) |
| Enter / E / double click | Open the frame editor popup |
| Space (hold) | Peek at the full frame |

| Frame mode and popup | Action |
|---|---|
| h / l, ← / → (Ctrl: annotated only), Home / End, right click | Previous / next / first / last frame, next frame (frame mode only) |
| j / k, ↓ / ↑ | Next / previous box (GT, then predictions) |
| D / G | Delete-or-reject / accept-or-confirm the selected box |
| F | Toggle frame deletion (frame mode: then next frame) |
| 1–9 | Category of the selected box and of new boxes |
| Drag on empty area / box / handle | Draw / move / resize |
| Wheel, middle drag or Space + drag | Zoom around cursor, pan |
| Esc | Deselect (frame mode) / close (popup, also Enter / E) |

| To do (T) | Action |
|---|---|
| Arrows / hjkl, PgUp / PgDn, Home / End | Move |
| Enter / E / double click | Frame mode through the to-do frames only (h / l next / previous, G background + next, Tab / Esc back to the grid) |
| G / F | Background (nothing to annotate) / delete frame (toggles, advance) |
| Esc, T, M, Tab, Q | Back to the mosaic |

| Cluster mode (C) | Action |
|---|---|
| Drag (Shift: add, Ctrl: remove) | Lasso selection |
| Click / Shift+click, Ctrl+A | Select a frame / toggle it, select all visible |
| Right click | Actions: move to a split, show in mosaic, delete frames |
| 1 / 2 / 3 | Move the selection to train / val / test (on save) |
| F | Delete the selected frames (all marked: unmark) |
| Enter | Show the selection in the mosaic |
| Double click | Open the frame in the editor popup |
| Wheel, middle drag or Space + drag | Zoom around cursor, pan |
| Esc, C, M, Tab, Q | Back to the mosaic |

| Cleaning: overlapping GT | Action |
|---|---|
| 1–9 (10+: digits then Enter) | Keep that box, delete the others of the group, next group |
| A | Keep all, next group |
| D | Delete all boxes of the group, next group |
| F | Delete frame, next group |
| Right / Left (l / h, n / p) | Next / previous group |
| Mouse, G, j / k | Edit boxes as in the popup |
| Esc, M, Tab, Q | Back to the mosaic |

## Risks

| Risk | Mitigation |
|---|---|
| YOLO26 / RF-DETR ONNX output layouts differ from assumptions | Validate with real exports in M0/M5 before writing adapters; adapter tests compare against Python output. |
| Cold thumbnail generation too slow on first open of 100k images | Visible-first priority queue + persistent cache; optional `fastlabel cache warm` CLI. |
| 5 GB RAM machine (builds + app + browser) | Fixed LRU budget, no full-frame caching; `cargo` with limited `-j` if the linker runs out of memory. |
| Wayland quirks with winit | Test on Hyprland early (M3); X11 fallback via env var. |

## Status

| Milestone | State | Notes |
|---|---|---|
| M0 | done | Workspace, aliases, model probed. |
| M1 | done | 48 tests. All 24 real annotation files round-trip losslessly. v1_exp0: open 25 ms, evaluate 2 ms. Synthetic 100k images / 200k boxes / 300k preds: index 53 ms, evaluate 53 ms, apply 87 ms, save 108 ms, load 187 ms, peak RSS 230 MB. |
| M2 | done | Pure-Rust decode (zune-jpeg, 7.8 ms per 1080p frame). v1_exp0 10,385 crops: cold 911 crops/s, warm (disk cache) 34k crops/s. Indexed priority queue, one decode per frame per batch. |
| M3 | done | egui 0.36 mosaic: virtualized rows, hjkl/arrows/PgUp/PgDn/Home/End, D/G/F/R + Shift row toggles (no auto-repeat), right-click = D, undo/redo, S/Q, save on window close, session restore, sort + split + hide-reviewed + FP/FN filters. Headless UI tests with egui_kittest (offscreen wgpu): full keyboard flow on a synthetic dataset; opt-in screenshot test on a real copy. |
| M4 | done | TP crops show GT + matched prediction, FP crops, IoU/confidence sorts, FP/FN filters, thresholds menu (match / FP / FN IoU), R replace, persistent rejects, reviewed state + hide-reviewed, model info tooltip. |
| M5 | done | ort CPU detector: YOLO raw head + class-wise NMS (exp-10), YOLO end-to-end, DETR-style (RF-DETR, **not yet validated against a real export**). Square or Ultralytics-"rect" letterbox (dynamic-input models). Input tensor matches OpenCV/Ultralytics within 0.13/255 mean. vs the `.pt` with the same square letterbox: 146/147 boxes matched, mean IoU 0.983, max score diff 0.027 (exp-10.onnx has FP16 weights). Speed on Ryzen 5 5500U: 1.16 s/frame at 1280² (6 threads); PyTorch rect 1280×736: 0.42 s. Re-exporting FP32 + `dynamic=True` should close most of the gap. App: Model menu (path, confidence, run all / run on frame), background run with progress + cancel, save blocked during runs, cancelled full runs keep previous predictions, auto IoU-ascending sort after a full run. |
| M6 | done | Tab frame mode (h/l, Ctrl annotated-only), editor shared with the Enter/double-click popup: draw, select, move, 8-handle resize, zoom around cursor, pan (middle / Space drag), j/k box cycling, D/G on boxes and predictions, 1–9 categories, tags, F, run model on frame. Hold-Space peek over the mosaic. Edits are pending ops (visible immediately in frame view, applied on save). UI test drives Tab, mouse draw, D, l, Enter, Esc. |
| M7 | done | Color uniformity (gray std dev in the box) for GT + prediction boxes, computed in the background on all cores (v1_exp0: 23,580 boxes in 11.7 s), cached incrementally in `.fastlabel/metrics.json`, sortable asc/desc, shown in tooltips; already surfaces GT boxes drawn on a sign. Export zip button: saves pending edits, then annotation file + images (stored) with progress and cancel. |
| M8 | done | SIGKILL-during-save test (8 random kills: every version parses, project reopens). Property tests: random bytes into every loader and the JPEG decoder, random datasets + ops through evaluate/apply (result always valid, new boxes inside the image). 100k-image UI stress (symlinked 1080p JPEGs, 200k boxes, 100k predictions): open + first frame 0.56 s, worst frame while jumping 15 ms, 2,000 marks + undos 3.4 s, re-evaluate 108 ms, +297 MB RSS. Empty-dataset key sweep. README. |

### Changes after first use (2026-10-05)

- Matching has no thresholds anymore: pairs need any overlap; FP / FN mean zero overlap; duplicates are "extra".
- D/G/R on a frame marked F apply and unmark the frame.
- Cleaning menu, first workflow: overlapping GT boxes (390 groups on the v023 copy).

### M9: Cluster mode (split management)

- `Op::SetSplit` (pending like every edit). The split is a `split` field of each COCO image (path as fallback for older files), written for every frame on save; files never move. Trash moves are recorded in `.fastlabel/moves.json` before the annotation file is written and finished on the next open after a crash (closes the gap where a crash between save and trash move re-added deleted frames as new images).
- `fastlabel-infer::Embedder`: ONNX `[N,3,H,W] → [N,D]`, antialiased resize + ImageNet normalization (matches PIL bilinear exactly: cos 1.0000 vs PyTorch on the same frame), batches of 8, 2 decoder threads. DINOv2-S/14 at 224: ~140 ms/frame on the Ryzen 5 5500U.
- `app::cluster` (pure, no new dependency): PCA (covariance + Jacobi), exact kNN (threads), UMAP (fuzzy graph + SGD with negative sampling, PCA init, seeded), HDBSCAN (mutual reachability MST, condensed tree, excess of mass). Exact kNN and Prim's MST are O(n²): fine for 10k frames, minutes for 100k.
- v1_exp0 (4,936 frames): embeddings 8.5 min (~103 ms/frame, once, cached), layout 6.5 s, 19 clusters + 51 noise frames; val and test each form their own clusters (no visible leakage).
- Export (`core::export`): YOLO, COCO per split (Roboflow / RF-DETR), COCO single file, Ultralytics NDJSON; zip (atomic) or folder (temp sibling + rename, hardlinks with copy fallback); `--export` CLI. BibBoxes (4,612 frames): Ultralytics' YOLO loader and NDJSON converter read the exports with identical counts (one truncated source JPEG, `images/test/0000023_0355.jpg`, is skipped by Ultralytics).
- Add images (`core::add`): content-hash names (SHA-256, 16 hex), duplicates skipped, temp + rename copies, `source_name` field; `Project::register_added` assigns ids, writes `.fastlabel/sources.json` and rewrites the journal header so a crash before the save replays onto the same frames.
- `date_added` image field (set by Add, used by the date sort; BibBoxes backfilled in v003 from the original files' creation times). Split filter gains *no split* (`SplitFilter`, reads the old session format).
- To do tab (`T`): frames without GT boxes, newest first; G confirms background (`Op::ConfirmFrame`, `review.empty_frames`), F deletes, Enter opens frame mode restricted to the to-do list (`Workspace::frame_order`).
- BibBoxes (`~/Projects/Datasets/BibBoxes`): fresh dataset from v1_exp0 v042, real copies under content-hash names, split + source_name fields, predictions and embedding/cluster caches re-keyed.
- Auto-split (`cluster::auto_split`): whole clusters, capped share per cluster, seeded order, fillers last; one undo step.
- Reports (`app::report`, pure + SVG, no dependency): all frames / selection / cluster, `--report` CLI. v1_exp0: 0.24 s including the cross-split similarity (665 × 4,271 cosines).
- Cluster view: mesh-drawn points, lasso, context menu, selection strip (virtualized thumbnails), split filter shared with the mosaic, mosaic filter by selected frames.
