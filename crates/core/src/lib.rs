//! FastLabel core: dataset model, versioning, predictions, matching and
//! pending edits. Pure logic and file I/O; no GUI, no threads.

pub mod coco;
pub mod commit;
pub mod dataset;
pub mod edits;
pub mod error;
pub mod export;
pub mod fsutil;
pub mod geom;
pub mod journal;
pub mod matching;
pub mod predictions;
pub mod project;
pub mod review;
pub mod versions;
pub mod yolo;

pub use coco::{AnnId, CategoryId, ImageId};
pub use dataset::{Dataset, Split};
pub use edits::{AnnMark, EditLog, Op, PendingState, PredMark};
pub use error::{Error, Result};
pub use geom::BBox;
pub use matching::{Evaluation, GtStatus, PredStatus};
pub use predictions::{PredRef, Predictions};
pub use project::Project;
