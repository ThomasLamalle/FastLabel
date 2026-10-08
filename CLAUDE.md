# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# FastLabel

Rust rebuild of a bounding-box annotation triage tool. Functional source of truth: `features.md`. Architecture and milestones: `docs/PLAN.md`.

## Layout
- `crates/core` (`fastlabel-core`): data model, COCO/YOLO I/O, versioning, predictions, IoU/matching, edit log, journal, save/trash/export. Pure logic, no GUI, no threads.
- `crates/media` (`fastlabel-media`): JPEG decode, crops, thumbnail cache, worker pool, image metrics.
- `crates/infer` (`fastlabel-infer`): ONNX Runtime detectors (YOLO26 raw + end2end, RF-DETR later).
- `crates/app` (`fastlabel`): eframe/egui GUI. Pure logic (`crops`, `workspace`, `frame`, `metrics`) is unit-tested; `ui_tests.rs` drives the real app headlessly with egui_kittest.

## Rules
- UI thread never decodes images or runs inference (loader pool, inference and task threads do). Its only writes are the one-line journal append per action and explicit saves.
- Every user action is an `Op` in the edit log; the dataset only changes on save.
- Writes are atomic (temp file + rename); metadata is written before images are moved to trash.
- Never hard-delete user images.

## Commands
- `cargo fmt --all --check && cargo check-all && cargo t` before every commit.
- `cargo check-all` is clippy only (`-D warnings`), despite the comment in `.cargo/config.toml`; `cargo t` = `cargo test --workspace`.
- Single test: `cargo test -p fastlabel-core <name>` (app crate is `-p fastlabel`; UI tests live in `crates/app/src/ui_tests.rs`).
- Opt-in `#[ignore]` UI tests read `FASTLABEL_SHOT_DATASET` (dataset copy), `FASTLABEL_SHOT_DIR` (screenshots out), `FASTLABEL_MODEL`, `FASTLABEL_EMBED_MODEL`: e.g. `FASTLABEL_SHOT_DATASET=target/test-data/v1_exp0 FASTLABEL_SHOT_DIR=<out> cargo test -p fastlabel real_dataset_shot -- --ignored`. Scale test: `cargo test --release -p fastlabel stress_100k -- --ignored --nocapture`.
- Headless: `fastlabel --export out.zip --format yolo <dataset>`, `fastlabel --report out.html <dataset>`. Key-event logging: `RUST_LOG=fastlabel=debug`.
- When a task is finished, run `cargo build --release`: the user runs `./fastlabel`, a symlink to `target/release/fastlabel`, so a stale build hides the changes.
- Never run the app or opt-in tests on `../Datasets/...` directly: use the hardlinked copy in `target/test-data/v1_exp0` (saves create versions, F moves images to trash). Model: `~/Downloads/exp-10.onnx` (YOLO26s, raw head, 1280 input, FP16 weights, NMS needed).
- egui is 0.36: `App::ui(&mut self, ui, frame)`, `egui::Panel::top(..).show(ui, ..)`. Check APIs in `~/.cargo/registry/src/*/egui-0.36*` rather than guessing.
- Machine: 12 cores, 5 GB RAM, no GPU. Keep memory bounded.
