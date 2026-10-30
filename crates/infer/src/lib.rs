//! FastLabel inference: ONNX detectors (YOLO raw / end-to-end, DETR-style),
//! image embedders, OCR text recognizers and background batch runners.

pub mod batch;
pub mod detector;
pub mod embedder;
pub mod ocr;
pub mod prepost;

pub use batch::{BatchItem, BatchResult, Progress, run_batch};
pub use detector::{Arch, Detector, InferError};
pub use embedder::{EmbedResult, Embedder, run_embed};
pub use ocr::{OcrResult, Reading, TextReader, run_ocr};
pub use prepost::Detection;
