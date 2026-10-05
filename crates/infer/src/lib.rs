//! FastLabel inference: ONNX detectors (YOLO raw / end-to-end, DETR-style)
//! and a background batch runner.

pub mod batch;
pub mod detector;
pub mod prepost;

pub use batch::{BatchItem, BatchResult, Progress, run_batch};
pub use detector::{Arch, Detector, InferError};
pub use prepost::Detection;
