# FastLabel

Rust rebuild of a bounding-box annotation triage tool. Functional source of truth: `features.md`. Architecture and milestones: `docs/PLAN.md`.

## Layout
- `crates/core` (`fastlabel-core`): data model, COCO/YOLO I/O, versioning, predictions, IoU/matching, edit log, journal, save/trash/export. Pure logic, no GUI, no threads.
- `crates/media` (`fastlabel-media`): JPEG decode, crops, thumbnail cache, worker pool, image metrics.
- `crates/infer` (`fastlabel-infer`): ONNX Runtime detectors (YOLO26 raw + end2end, RF-DETR later).
- `crates/app` (`fastlabel`): eframe/egui GUI + CLI.

## Rules
- UI thread never decodes images, writes files or runs inference.
- Every user action is an `Op` in the edit log; the dataset only changes on save.
- Writes are atomic (temp file + rename); metadata is written before images are moved to trash.
- Never hard-delete user images.

## Commands
- `cargo fmt --all --check && cargo check-all && cargo t` before every commit.
- Real dataset for manual tests: `../Datasets/Ultralytics/v1_exp0`; model: `~/Downloads/exp-10.onnx` (YOLO26s, raw head, 1280 input, NMS needed).
- Machine: 12 cores, 5 GB RAM, no GPU. Keep memory bounded.
