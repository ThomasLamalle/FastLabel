# FastLabel

Fast keyboard-driven triage of bounding-box datasets. Review annotations and model predictions as a mosaic of crops: one key per decision instead of drawing boxes. Written in Rust (egui, ONNX Runtime). CPU-only machines are fully supported.

- What it does and why: [`features.md`](features.md)
- Architecture, milestones, measured performance, full key map: [`docs/PLAN.md`](docs/PLAN.md)

## Build and run

```sh
cargo build --release
./target/release/fastlabel /path/to/dataset   # or no argument: reopens the last dataset
```

Linux / Wayland first (X11 works too). ONNX Runtime is downloaded at build time by the `ort` crate.

## Dataset folder

| File | Role |
|---|---|
| `annotations.vNNN.json` | COCO annotations, one new file per save (never overwritten). The newest is opened. Legacy `annotations.draft.json` / `annotations.json` are read too. |
| `images/…` | Images, referenced by `file_name`. The split is the image's `split` field (older files: the `train` / `val` / `test` folder in the path). New files copied here are picked up on open (then *Model → Run on frames without predictions*, sort by *Image date*). |
| `data.yaml` + `labels/` | YOLO layout, imported when there is no annotation file yet (first save writes `annotations.v001.json`). |
| `predictions.json` | Model predictions (written by the Model menu). |
| `.fastlabel/` | Tool state: crash journal of unsaved actions, review state (reviewed boxes, rejected predictions), thumbnail cache, uniformity metrics, image embeddings and cluster layout, `trash/` with deleted frames. |

Keyboard troubleshooting: `RUST_LOG=fastlabel=debug ./target/release/fastlabel` logs every key event the app receives.

Nothing changes on disk until you save (`S`). Every action is journaled, so a crash or power cut loses nothing: pending actions are replayed when the dataset is reopened. Deleted frames are moved to `.fastlabel/trash/<version>/`, never deleted. Changing a split only changes the `split` field; files never move. A save interrupted between the annotation file and the trash moves is finished on the next open.

## Essential keys

| Mosaic | |
|---|---|
| hjkl / arrows | move |
| D / G / F / R | delete-or-reject / accept-or-confirm / delete frame / replace GT by prediction (toggles, advance to the next undecided crop) |
| Shift + D / G / F | from the selection to the end of the row |
| U, Ctrl+Z / Ctrl+Shift+Z | undo / redo |
| Enter or E, Space (hold) | frame editor popup, peek at the frame |
| Tab or M | frame mode (draw, move, resize, 1–9 class, tags) |
| S / Q | save new version / save and quit |

*Clusters* (C) shows every frame as a point (image embeddings, UMAP, HDBSCAN), colored by split or cluster. Drag around points to select them (Shift adds, Ctrl removes), then right click or 1 / 2 / 3 to move them to train / val / test, F to delete the frames, Enter to review them in the mosaic. It needs an ONNX image embedder, e.g. DINOv2-small:

```sh
python scripts/export_dinov2.py dinov2_vits14.onnx   # any Python with torch + onnx; downloads the weights once
```

*Export…* writes YOLO (Ultralytics / YOLO26), COCO per split (Roboflow layout, RF-DETR), COCO single file or Ultralytics NDJSON, laid out by split, to a zip or a folder of hardlinks. Headless: `fastlabel --export out.zip --format yolo <dataset>`. In cluster mode, *Auto-split…* assigns whole clusters to train / val / test.

*Report* writes an HTML report (split balance, box distributions per split, model input size guide, per-split model results, near duplicates across splits, clusters × splits) and opens it; in cluster mode it can cover the selection or one cluster. Headless: `fastlabel --report out.html <dataset>`.

Embeddings take ~140 ms per frame on a 6-core laptop CPU, computed once in the background and cached.

*Cleaning → Overlapping GT boxes* reviews double annotations one group at a time: 1–9 keeps that box and deletes the others, A keeps all, D deletes all, Right/Left moves between groups, Esc goes back.

## Models

Model menu → choose an `.onnx` file, confidence, then *Run on all frames* (background, with progress and cancel), *Run on frames without predictions* or *Run on the selected crop's frame*. Supported: Ultralytics YOLO exports (raw head with NMS, or end-to-end), DETR-style models (RF-DETR, experimental). Class names are matched to dataset categories by name.

For the fastest CPU inference, export FP32 with a dynamic input so frames are letterboxed to their real aspect ratio (1280×736 instead of 1280×1280):

```sh
yolo export model=best.pt format=onnx dynamic=True
```

## Tests

```sh
cargo fmt --all --check && cargo check-all && cargo t
```

This includes headless UI tests (egui_kittest, offscreen wgpu), a SIGKILL-during-save crash test and property tests on all loaders. Opt-in tests on real data:

```sh
FASTLABEL_SHOT_DATASET=<copy of a dataset> FASTLABEL_SHOT_DIR=<png out dir> FASTLABEL_MODEL=<model.onnx> \
  cargo test --release -p fastlabel -- --ignored --nocapture
```

Run the opt-in tests on a copy: they save versions and run models. A hardlinked copy (`cp -al images …`) costs no disk space.
