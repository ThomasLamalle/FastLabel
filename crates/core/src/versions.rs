//! Annotation file versioning: `annotations.vNNN.json` snapshots, plus the
//! legacy `annotations.draft.json` / `annotations.json` files.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::error::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VersionKind {
    /// `annotations.json`
    Base,
    /// `annotations.draft.json` written by the previous tool.
    Draft,
    /// `annotations.vNNN.json`
    Numbered(u32),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VersionFile {
    pub path: PathBuf,
    pub kind: VersionKind,
    pub modified: SystemTime,
}

impl VersionFile {
    pub fn file_name(&self) -> String {
        self.path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    }
}

pub fn parse_kind(file_name: &str) -> Option<VersionKind> {
    match file_name {
        "annotations.json" => Some(VersionKind::Base),
        "annotations.draft.json" => Some(VersionKind::Draft),
        _ => {
            let digits = file_name
                .strip_prefix("annotations.v")?
                .strip_suffix(".json")?;
            if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            digits.parse().ok().map(VersionKind::Numbered)
        }
    }
}

pub fn numbered_file_name(n: u32) -> String {
    format!("annotations.v{n:03}.json")
}

/// All annotation files in `dir`, oldest numbered first.
pub fn discover(dir: &Path) -> Result<Vec<VersionFile>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| Error::io(dir, e))? {
        let entry = entry.map_err(|e| Error::io(dir, e))?;
        let name = entry.file_name();
        let Some(kind) = parse_kind(&name.to_string_lossy()) else {
            continue;
        };
        let meta = entry.metadata().map_err(|e| Error::io(entry.path(), e))?;
        if !meta.is_file() {
            continue;
        }
        let modified = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        out.push(VersionFile {
            path: entry.path(),
            kind,
            modified,
        });
    }
    out.sort_by_key(|v| v.kind);
    Ok(out)
}

/// The file to open: the newest numbered snapshot, unless the legacy draft
/// was modified strictly after it; `annotations.json` only as a last resort.
pub fn latest(files: &[VersionFile]) -> Option<&VersionFile> {
    let numbered = files
        .iter()
        .filter(|f| matches!(f.kind, VersionKind::Numbered(_)))
        .max_by_key(|f| f.kind);
    let draft = files.iter().find(|f| f.kind == VersionKind::Draft);
    let base = files.iter().find(|f| f.kind == VersionKind::Base);
    match (numbered, draft) {
        (Some(n), Some(d)) if d.modified > n.modified => Some(d),
        (Some(n), _) => Some(n),
        (None, Some(d)) => Some(d),
        (None, None) => base,
    }
}

/// Path of the next snapshot (`max + 1`, never overwriting an existing one).
pub fn next_path(dir: &Path, files: &[VersionFile]) -> PathBuf {
    let max = files
        .iter()
        .filter_map(|f| match f.kind {
            VersionKind::Numbered(n) => Some(n),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    let mut n = max + 1;
    while dir.join(numbered_file_name(n)).exists() {
        n += 1;
    }
    dir.join(numbered_file_name(n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn vf(kind: VersionKind, secs: u64) -> VersionFile {
        VersionFile {
            path: PathBuf::from(format!("{kind:?}")),
            kind,
            modified: SystemTime::UNIX_EPOCH + Duration::from_secs(secs),
        }
    }

    #[test]
    fn parses_file_names() {
        assert_eq!(
            parse_kind("annotations.v022.json"),
            Some(VersionKind::Numbered(22))
        );
        assert_eq!(
            parse_kind("annotations.v1234.json"),
            Some(VersionKind::Numbered(1234))
        );
        assert_eq!(
            parse_kind("annotations.draft.json"),
            Some(VersionKind::Draft)
        );
        assert_eq!(parse_kind("annotations.json"), Some(VersionKind::Base));
        assert_eq!(parse_kind("annotations.v.json"), None);
        assert_eq!(parse_kind("annotations.vx1.json"), None);
        assert_eq!(parse_kind("predictions.json"), None);
    }

    #[test]
    fn latest_prefers_newest_numbered_unless_draft_is_newer() {
        let files = vec![
            vf(VersionKind::Base, 1),
            vf(VersionKind::Numbered(1), 5),
            vf(VersionKind::Numbered(2), 10),
            vf(VersionKind::Draft, 10),
        ];
        assert_eq!(latest(&files).unwrap().kind, VersionKind::Numbered(2));
        let files = vec![vf(VersionKind::Numbered(2), 10), vf(VersionKind::Draft, 11)];
        assert_eq!(latest(&files).unwrap().kind, VersionKind::Draft);
        assert_eq!(
            latest(&[vf(VersionKind::Base, 1)]).unwrap().kind,
            VersionKind::Base
        );
        assert!(latest(&[]).is_none());
    }

    #[test]
    fn next_path_never_overwrites() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("annotations.v002.json"), "{}").unwrap();
        fs::write(dir.path().join("annotations.v003.json"), "{}").unwrap();
        let files = discover(dir.path()).unwrap();
        assert_eq!(files.len(), 2);
        assert_eq!(
            next_path(dir.path(), &files),
            dir.path().join("annotations.v004.json")
        );
        assert_eq!(
            next_path(dir.path(), &[]),
            dir.path().join("annotations.v001.json")
        );
    }
}
