//! Append-only log of pending edits, replayed after a crash.
//!
//! Line 1 is a header naming the annotation file the edits apply to; every
//! following line is one [`Entry`]. A truncated last line (crash mid-write) is
//! ignored on replay.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::edits::{EditLog, Op};
use crate::error::{Error, Result};

pub const JOURNAL_FILE: &str = "journal.jsonl";
const FORMAT: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Header {
    fastlabel_journal: u32,
    base: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Entry {
    Op { op: Op },
    Undo,
    Redo,
}

pub struct Journal {
    path: PathBuf,
    file: File,
}

impl Journal {
    /// Starts a fresh journal for edits on top of `base` (an annotation file name).
    pub fn create(path: &Path, base: &str) -> Result<Self> {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
        }
        let mut file = File::create(path).map_err(|e| Error::io(path, e))?;
        let header = Header {
            fastlabel_journal: FORMAT,
            base: base.to_string(),
        };
        writeln!(
            file,
            "{}",
            serde_json::to_string(&header).expect("header serializes")
        )
        .map_err(|e| Error::io(path, e))?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
        })
    }

    /// Continues an existing journal (after a replay).
    pub fn open_append(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .append(true)
            .open(path)
            .map_err(|e| Error::io(path, e))?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
        })
    }

    /// Appends one entry. Unbuffered write of a single line: the OS page
    /// cache makes this cheap, and a crash loses at most the last line.
    pub fn append(&mut self, entry: &Entry) -> Result<()> {
        let mut line = serde_json::to_vec(entry).expect("entry serializes");
        line.push(b'\n');
        self.file
            .write_all(&line)
            .map_err(|e| Error::io(&self.path, e))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Contents of a journal file: the base it applies to and its entries.
#[derive(Debug, PartialEq)]
pub struct Recorded {
    pub base: String,
    pub entries: Vec<Entry>,
}

/// Reads a journal; `Ok(None)` when it does not exist or has no valid header.
pub fn read(path: &Path) -> Result<Option<Recorded>> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(Error::io(path, e)),
    };
    let mut lines = BufReader::new(file).lines();
    let Some(Ok(first)) = lines.next() else {
        return Ok(None);
    };
    let Ok(header) = serde_json::from_str::<Header>(&first) else {
        return Ok(None);
    };
    let mut entries = Vec::new();
    for line in lines {
        let line = line.map_err(|e| Error::io(path, e))?;
        match serde_json::from_str(&line) {
            Ok(entry) => entries.push(entry),
            // Only the last line can be partial; stop at the first bad one.
            Err(_) => break,
        }
    }
    Ok(Some(Recorded {
        base: header.base,
        entries,
    }))
}

pub fn replay(entries: &[Entry], log: &mut EditLog) {
    for entry in entries {
        match entry {
            Entry::Op { op } => log.push(op.clone()),
            Entry::Undo => {
                log.undo();
            }
            Entry::Redo => {
                log.redo();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edits::AnnMark;

    #[test]
    fn write_read_replay_with_truncated_tail() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".fastlabel").join(JOURNAL_FILE);
        let mut j = Journal::create(&path, "annotations.v003.json").unwrap();
        j.append(&Entry::Op {
            op: Op::AnnMark {
                ann: 1,
                mark: AnnMark::Delete,
            },
        })
        .unwrap();
        j.append(&Entry::Op {
            op: Op::FrameDelete { image: 2 },
        })
        .unwrap();
        j.append(&Entry::Undo).unwrap();
        drop(j);
        let mut j = Journal::open_append(&path).unwrap();
        j.append(&Entry::Op {
            op: Op::FrameDelete { image: 5 },
        })
        .unwrap();
        drop(j);
        // Simulate a crash mid-line.
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(br#"{"kind":"op","op":{"op":"frame_del"#)
            .unwrap();

        let rec = read(&path).unwrap().unwrap();
        assert_eq!(rec.base, "annotations.v003.json");
        assert_eq!(rec.entries.len(), 4);
        let mut log = EditLog::new(10);
        replay(&rec.entries, &mut log);
        assert_eq!(log.state().ann_mark(1), Some(AnnMark::Delete));
        assert!(!log.state().is_frame_deleted(2));
        assert!(log.state().is_frame_deleted(5));
    }

    #[test]
    fn missing_or_garbage_journal_is_none() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read(&dir.path().join("nope")).unwrap().is_none());
        let p = dir.path().join("bad");
        fs::write(&p, "not json\n").unwrap();
        assert!(read(&p).unwrap().is_none());
    }
}
