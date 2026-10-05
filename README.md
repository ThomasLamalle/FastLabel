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
| `images/{train,val,test}/…` | Images, referenced by `file_name`. The split comes from the path. |
| `data.yaml` + `labels/` | YOLO layout, imported when there is no annotation file yet (first save writes `annotations.v001.json`). |
| `predictions.json` | Model predictions (written by the Model menu). |
| `.fastlabel/` | Tool state: crash journal of unsaved actions, review state (reviewed boxes, rejected predictions), thumbnail cache, uniformity metrics, `trash/` with deleted frames. |

Nothing changes on disk until you save (`S`). Every action is journaled, so a crash or power cut loses nothing: pending actions are replayed when the dataset is reopened. Deleted frames are moved to `.fastlabel/trash/<version>/`, never deleted.

## Essential keys

| Mosaic | |
|---|---|
| hjkl / arrows | move |
| D / G / F / R | delete-or-reject / accept-or-confirm / delete frame / replace GT by prediction (toggles, auto-advance) |
| Shift + D / G / F | whole row |
| U, Ctrl+Z / Ctrl+Shift+Z | undo / redo |
| Enter, Space (hold) | frame editor popup, peek at the frame |
| Tab | frame mode (draw, move, resize, 1–9 class, tags) |
| S / Q | save new version / save and quit |

## Models

Model menu → choose an `.onnx` file, confidence, then *Run on all frames* (background, with progress and cancel) or *Run on the selected crop's frame*. Supported: Ultralytics YOLO exports (raw head with NMS, or end-to-end), DETR-style models (RF-DETR, experimental). Class names are matched to dataset categories by name.

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
