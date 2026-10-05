use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid JSON in {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("invalid data in {path}: {message}")]
    Invalid { path: PathBuf, message: String },
    #[error("zip error on {path}: {source}")]
    Zip {
        path: PathBuf,
        #[source]
        source: zip::result::ZipError,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn io(path: impl AsRef<Path>, source: std::io::Error) -> Self {
        Self::Io { path: path.as_ref().to_path_buf(), source }
    }

    pub fn json(path: impl AsRef<Path>, source: serde_json::Error) -> Self {
        Self::Json { path: path.as_ref().to_path_buf(), source }
    }

    pub fn invalid(path: impl AsRef<Path>, message: impl Into<String>) -> Self {
        Self::Invalid { path: path.as_ref().to_path_buf(), message: message.into() }
    }
}
