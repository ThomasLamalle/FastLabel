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

## Keyboard map (mosaic)

| Key | Action |
|---|---|
| ← → ↑ ↓ / h l k j | Move selection |
| PgUp / PgDn, Home / End | Scroll by screen / to ends |
| D / G / F / R | Delete-or-reject / accept-or-confirm / frame delete / replace (toggles, auto-advance) |
| Shift + D / G / F | Same, for the whole row, then next row |
| U, Ctrl+Z / Ctrl+Shift+Z, Ctrl+Y | Undo / redo |
| Enter | Open frame editor popup |
| Space (hold) | Peek full frame |
| Tab | Switch mosaic / frame mode |
| S / Q | Save new version / save and quit |

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
