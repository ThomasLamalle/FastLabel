//! Crash-safe file writes.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use serde::Serialize;

use crate::error::{Error, Result};

/// Writes `path` through a temporary sibling file, fsync and rename, so a
/// crash leaves either the old file or the complete new one.
pub fn write_atomic(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<File>) -> Result<()>,
) -> Result<()> {
    let dir = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
    let file_name = path.file_name().map(|n| n.to_string_lossy()).unwrap_or_default();
    let tmp = dir.join(format!(".{file_name}.tmp-{}", std::process::id()));
    let result = (|| {
        let file = File::create(&tmp).map_err(|e| Error::io(&tmp, e))?;
        let mut w = BufWriter::new(file);
        write(&mut w)?;
        w.flush().map_err(|e| Error::io(&tmp, e))?;
        w.get_ref().sync_all().map_err(|e| Error::io(&tmp, e))?;
        fs::rename(&tmp, path).map_err(|e| Error::io(path, e))
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

/// Atomic JSON write with 2-space indentation (as the previous tool wrote).
pub fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    write_atomic(path, |w| {
        let fmt = serde_json::ser::PrettyFormatter::with_indent(b"  ");
        let mut ser = serde_json::Serializer::with_formatter(&mut *w, fmt);
        value.serialize(&mut ser).map_err(|e| Error::json(path, e))?;
        w.write_all(b"\n").map_err(|e| Error::io(path, e))
    })
}

pub fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).map_err(|e| Error::io(path, e))?;
    serde_json::from_slice(&bytes).map_err(|e| Error::json(path, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_write_replaces_content_and_leaves_no_temp_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.json");
        write_json_atomic(&path, &serde_json::json!({"a": 1})).unwrap();
        write_json_atomic(&path, &serde_json::json!({"a": 2})).unwrap();
        let v: serde_json::Value = read_json(&path).unwrap();
        assert_eq!(v["a"], 2);
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn failed_write_keeps_previous_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.json");
        write_json_atomic(&path, &serde_json::json!({"a": 1})).unwrap();
        let err = write_atomic(&path, |_| Err(Error::invalid(&path, "boom")));
        assert!(err.is_err());
        let v: serde_json::Value = read_json(&path).unwrap();
        assert_eq!(v["a"], 1);
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }
}
