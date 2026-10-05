//! FastLabel media: decoding, crops, thumbnail cache, background loading and
//! per-box image metrics.

pub mod cache;
pub mod crop;
pub mod decode;
pub mod loader;
pub mod metrics;

pub use cache::DiskCache;
pub use crop::Region;
pub use loader::{Job, JobId, Loaded, Loader, Notify};
