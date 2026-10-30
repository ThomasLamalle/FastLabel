//! Text projects: one text per image, typically crops exported by
//! [`crate::export::ExportFormat::Crops`] to train or check an OCR model.
//!
//! - `labels.csv`: `file,<text column>,split,status` then any other columns
//!   (kept as they are). `status` is `todo` (not annotated yet), `ok` (the
//!   text is right) or `unreadable` (empty text). Images found under
//!   `images/` without a row are added as `todo`.
//! - `text_predictions.json`: OCR readings (pseudo labels) with their score
//!   and whether the user rejected them.
//! - Edits are pending (edit log, undo/redo, `.fastlabel/text_journal.jsonl`
//!   replayed after a crash) until saved; a save keeps the previous
//!   `labels.csv` in `.fastlabel/history/`.

use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::{Kind, ProjectConfig};
use crate::dataset::Split;
use crate::error::{Error, Result};
use crate::fsutil;
use crate::project::STATE_DIR;

pub const LABELS_FILE: &str = "labels.csv";
pub const PREDICTIONS_FILE: &str = "text_predictions.json";
const JOURNAL_FILE: &str = "text_journal.jsonl";
const HISTORY_DIR: &str = "history";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    #[default]
    Todo,
    Ok,
    Unreadable,
}

impl Status {
    pub fn name(self) -> &'static str {
        match self {
            Status::Todo => "todo",
            Status::Ok => "ok",
            Status::Unreadable => "unreadable",
        }
    }

    fn parse(s: &str) -> Option<Status> {
        [Status::Todo, Status::Ok, Status::Unreadable].into_iter().find(|v| v.name() == s.trim())
    }
}

/// The annotation of one image.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Label {
    pub status: Status,
    /// Empty unless `status` is `ok`.
    #[serde(default)]
    pub text: String,
    /// The OCR reading of this image was rejected (D).
    #[serde(default)]
    pub rejected: bool,
}

impl Label {
    pub fn ok(text: impl Into<String>) -> Self {
        Label { status: Status::Ok, text: text.into(), rejected: false }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    /// Path relative to the project folder.
    pub file: String,
    pub split: Option<Split>,
    /// Values of the extra columns, in header order.
    extra: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextPrediction {
    pub text: String,
    /// Mean probability of the characters kept.
    pub score: f32,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub rejected: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TextPredictions {
    pub model: String,
    /// By [`Row::file`].
    pub texts: BTreeMap<String, TextPrediction>,
}

/// One user action: the label of `file` goes from `before` to `after`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextOp {
    pub file: String,
    pub before: Label,
    pub after: Label,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Entry {
    Op { op: TextOp },
    Undo,
    Redo,
}

pub struct TextProject {
    pub dir: PathBuf,
    pub config: ProjectConfig,
    pub rows: Vec<Row>,
    /// Current labels (saved + pending edits), one per row.
    labels: Vec<Label>,
    saved: Vec<Label>,
    /// Rows added on open (images without a row); recorded by the next save.
    added: usize,
    extra_columns: Vec<String>,
    index: HashMap<String, usize>,
    log: Vec<TextOp>,
    cursor: usize,
    pub predictions: TextPredictions,
    journal: Option<File>,
    /// Non-fatal problems found while opening.
    pub warnings: Vec<String>,
}

impl TextProject {
    /// Writes the config of a new text project in `dir` and lists its images.
    pub fn create(dir: &Path, config: &ProjectConfig) -> Result<Self> {
        config.save(dir)?;
        let mut p = Self::open(dir)?;
        p.save()?;
        Ok(p)
    }

    pub fn open(dir: &Path) -> Result<Self> {
        let config = ProjectConfig::load(dir)?;
        if config.kind != Kind::Text {
            return Err(Error::invalid(dir, "not a text project"));
        }
        let mut p = Self {
            dir: dir.to_path_buf(),
            config,
            rows: Vec::new(),
            labels: Vec::new(),
            saved: Vec::new(),
            added: 0,
            extra_columns: Vec::new(),
            index: HashMap::new(),
            log: Vec::new(),
            cursor: 0,
            predictions: TextPredictions::default(),
            journal: None,
            warnings: Vec::new(),
        };
        let csv = dir.join(LABELS_FILE);
        if csv.exists() {
            p.read_labels(&csv)?;
        }
        let images = dir.join("images");
        let root = if images.is_dir() { images } else { dir.to_path_buf() };
        for rel in crate::yolo::list_images(&root, dir)? {
            let file = rel.to_string_lossy().replace('\\', "/");
            if !p.index.contains_key(&file) {
                p.push_row(Row { file, split: None, extra: Vec::new() }, Label::default());
                p.added += 1;
            }
        }
        if p.added > 0 {
            p.warnings.push(format!("{} images without a row added", p.added));
        }
        let preds = dir.join(PREDICTIONS_FILE);
        if preds.exists() {
            p.predictions = fsutil::read_json(&preds)?;
            for (file, pred) in &p.predictions.texts {
                if let Some(&i) = p.index.get(file) {
                    p.labels[i].rejected = pred.rejected;
                }
            }
        }
        p.saved = p.labels.clone();
        p.replay()?;
        Ok(p)
    }

    fn push_row(&mut self, row: Row, label: Label) {
        self.index.insert(row.file.clone(), self.rows.len());
        self.rows.push(row);
        self.labels.push(label);
    }

    fn read_labels(&mut self, path: &Path) -> Result<()> {
        let text = fs::read_to_string(path).map_err(|e| Error::io(path, e))?;
        let mut records = parse_csv(&text).into_iter();
        let header = records.next().unwrap_or_default();
        let col = |name: &str| header.iter().position(|h| h.trim() == name);
        let file_col = col("file").ok_or_else(|| Error::invalid(path, "no `file` column"))?;
        let text_col = col(&self.config.text_column);
        let (split_col, status_col) = (col("split"), col("status"));
        let known = [Some(file_col), text_col, split_col, status_col];
        let extra: Vec<usize> = (0..header.len()).filter(|i| !known.contains(&Some(*i))).collect();
        self.extra_columns = extra.iter().map(|&i| header[i].clone()).collect();
        for (n, rec) in records.enumerate() {
            let get = |c: Option<usize>| c.and_then(|c| rec.get(c)).map_or("", |s| s.as_str());
            let file = get(Some(file_col)).trim().to_string();
            if file.is_empty() || self.index.contains_key(&file) {
                continue;
            }
            let text = get(text_col).trim().to_string();
            let status = match status_col {
                Some(_) => Status::parse(get(status_col))
                    .ok_or_else(|| Error::invalid(path, format!("line {}: bad status", n + 2)))?,
                // Plain `file,text` lists: a text is a label.
                None if text.is_empty() => Status::Todo,
                None => Status::Ok,
            };
            let text = if status == Status::Ok { text } else { String::new() };
            let split = Split::parse(get(split_col));
            let extra = extra.iter().map(|&c| get(Some(c)).to_string()).collect();
            self.push_row(Row { file, split, extra }, Label { status, text, rejected: false });
        }
        Ok(())
    }

    fn journal_path(&self) -> PathBuf {
        self.dir.join(STATE_DIR).join(JOURNAL_FILE)
    }

    /// Re-applies the journal of a session that did not save. The ops hold
    /// absolute labels, so replaying them after a save that did not get to
    /// truncate the journal changes nothing.
    fn replay(&mut self) -> Result<()> {
        let path = self.journal_path();
        if let Ok(text) = fs::read_to_string(&path) {
            for line in text.lines() {
                // A truncated last line (crash mid-write) is ignored.
                let Ok(entry) = serde_json::from_str::<Entry>(line) else { continue };
                match entry {
                    Entry::Op { op } if self.index.contains_key(&op.file) => {
                        self.log.truncate(self.cursor);
                        self.apply(&op.file, op.after.clone());
                        self.log.push(op);
                        self.cursor += 1;
                    }
                    Entry::Op { .. } => {}
                    Entry::Undo => {
                        let _ = self.step(false);
                    }
                    Entry::Redo => {
                        let _ = self.step(true);
                    }
                }
            }
        }
        if self.cursor > 0 {
            self.warnings.push(format!("{} unsaved edits recovered", self.cursor));
        }
        let state = self.dir.join(STATE_DIR);
        fs::create_dir_all(&state).map_err(|e| Error::io(&state, e))?;
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| Error::io(&path, e))?;
        self.journal = Some(file);
        Ok(())
    }

    fn write_journal(&mut self, entry: &Entry) -> Result<()> {
        let path = self.journal_path();
        if let Some(f) = &mut self.journal {
            let line = serde_json::to_string(entry).expect("journal entry serializes");
            writeln!(f, "{line}").map_err(|e| Error::io(&path, e))?;
        }
        Ok(())
    }

    fn apply(&mut self, file: &str, label: Label) -> Option<usize> {
        let i = *self.index.get(file)?;
        self.labels[i] = label;
        Some(i)
    }

    pub fn label(&self, i: usize) -> &Label {
        &self.labels[i]
    }

    /// Row `i` has an unsaved edit.
    pub fn is_pending(&self, i: usize) -> bool {
        self.labels[i] != self.saved[i]
    }

    /// The OCR reading of row `i`.
    pub fn prediction(&self, i: usize) -> Option<&TextPrediction> {
        self.predictions.texts.get(&self.rows[i].file)
    }

    /// Records the new label of row `i` (no-op when unchanged).
    pub fn set(&mut self, i: usize, after: Label) -> Result<()> {
        let before = self.labels[i].clone();
        if before == after {
            return Ok(());
        }
        let op = TextOp { file: self.rows[i].file.clone(), before, after };
        self.labels[i] = op.after.clone();
        self.write_journal(&Entry::Op { op: op.clone() })?;
        self.log.truncate(self.cursor);
        self.log.push(op);
        self.cursor += 1;
        Ok(())
    }

    fn step(&mut self, redo: bool) -> Option<usize> {
        if redo {
            let op = self.log.get(self.cursor)?.clone();
            self.cursor += 1;
            self.apply(&op.file, op.after)
        } else {
            self.cursor = self.cursor.checked_sub(1)?;
            let op = self.log[self.cursor].clone();
            self.apply(&op.file, op.before)
        }
    }

    /// Undoes the last edit; returns the row it changed.
    pub fn undo(&mut self) -> Result<Option<usize>> {
        let row = self.step(false);
        if row.is_some() {
            self.write_journal(&Entry::Undo)?;
        }
        Ok(row)
    }

    pub fn redo(&mut self) -> Result<Option<usize>> {
        let row = self.step(true);
        if row.is_some() {
            self.write_journal(&Entry::Redo)?;
        }
        Ok(row)
    }

    pub fn can_undo(&self) -> bool {
        self.cursor > 0
    }

    pub fn can_redo(&self) -> bool {
        self.cursor < self.log.len()
    }

    /// Rows whose label differs from the saved one.
    pub fn unsaved(&self) -> usize {
        self.labels.iter().zip(&self.saved).filter(|(a, b)| a != b).count()
    }

    pub fn has_unsaved_changes(&self) -> bool {
        self.added > 0 || self.labels != self.saved
    }

    /// Number of rows per status.
    pub fn counts(&self) -> HashMap<Status, usize> {
        let mut out = HashMap::new();
        for l in &self.labels {
            *out.entry(l.status).or_default() += 1;
        }
        out
    }

    /// Writes `labels.csv` (the previous one goes to `.fastlabel/history/`)
    /// and the rejections; `None` when there was nothing to save.
    pub fn save(&mut self) -> Result<Option<PathBuf>> {
        if !self.has_unsaved_changes() {
            return Ok(None);
        }
        if self.labels.iter().zip(&self.saved).any(|(a, b)| a.rejected != b.rejected) {
            for (i, row) in self.rows.iter().enumerate() {
                if let Some(p) = self.predictions.texts.get_mut(&row.file) {
                    p.rejected = self.labels[i].rejected;
                }
            }
            fsutil::write_json_atomic(&self.dir.join(PREDICTIONS_FILE), &self.predictions)?;
        }
        let path = self.dir.join(LABELS_FILE);
        if path.exists() {
            let history = self.dir.join(STATE_DIR).join(HISTORY_DIR);
            fs::create_dir_all(&history).map_err(|e| Error::io(&history, e))?;
            let n = fs::read_dir(&history).map_err(|e| Error::io(&history, e))?.count() + 1;
            let old = history.join(format!("labels.v{n:03}.csv"));
            fs::copy(&path, &old).map_err(|e| Error::io(&old, e))?;
        }
        let csv = self.to_csv();
        fsutil::write_atomic(&path, |w| {
            w.write_all(csv.as_bytes()).map_err(|e| Error::io(&path, e))
        })?;
        self.saved = self.labels.clone();
        self.added = 0;
        self.log.clear();
        self.cursor = 0;
        let jp = self.journal_path();
        self.journal = Some(File::create(&jp).map_err(|e| Error::io(&jp, e))?);
        Ok(Some(path))
    }

    fn to_csv(&self) -> String {
        let mut header = vec!["file", self.config.text_column.trim(), "split", "status"];
        header.extend(self.extra_columns.iter().map(String::as_str));
        let mut out = csv_line(&header);
        for (row, label) in self.rows.iter().zip(&self.labels) {
            let mut fields =
                vec![row.file.as_str(), &label.text, row.split.map_or("", |s| s.name())];
            fields.push(label.status.name());
            for k in 0..self.extra_columns.len() {
                fields.push(row.extra.get(k).map_or("", String::as_str));
            }
            out.push_str(&csv_line(&fields));
        }
        out
    }

    /// Stores new OCR readings (merged over the previous ones) at once, like
    /// a model run. A rejection stays only when the reading is unchanged.
    pub fn merge_predictions(
        &mut self,
        model: String,
        texts: BTreeMap<String, TextPrediction>,
    ) -> Result<()> {
        let mut preds = self.predictions.clone();
        preds.model = model;
        for (file, mut p) in texts {
            let old = preds.texts.get(&file);
            p.rejected = old.is_some_and(|o| o.rejected && o.text == p.text);
            if let Some(&i) = self.index.get(&file)
                && let Some(o) = old
                && o.text != p.text
            {
                self.labels[i].rejected = false;
                self.saved[i].rejected = false;
            }
            preds.texts.insert(file, p);
        }
        fsutil::write_json_atomic(&self.dir.join(PREDICTIONS_FILE), &preds)?;
        self.predictions = preds;
        Ok(())
    }
}

/// Parses CSV (RFC 4180: quoted fields with `""`, CRLF or LF). Empty lines
/// are skipped.
pub fn parse_csv(text: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let (mut row, mut field) = (Vec::new(), String::new());
    let (mut quoted, mut chars) = (false, text.chars().peekable());
    while let Some(c) = chars.next() {
        match c {
            '"' if quoted && chars.peek() == Some(&'"') => {
                chars.next();
                field.push('"');
            }
            '"' if quoted => quoted = false,
            '"' if field.is_empty() => quoted = true,
            ',' if !quoted => row.push(std::mem::take(&mut field)),
            '\r' if !quoted => {}
            '\n' if !quoted => {
                row.push(std::mem::take(&mut field));
                if row.len() > 1 || !row[0].is_empty() {
                    rows.push(std::mem::take(&mut row));
                }
                row.clear();
            }
            c => field.push(c),
        }
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        rows.push(row);
    }
    rows
}

/// One CSV line, fields quoted when needed.
pub fn csv_line(fields: &[&str]) -> String {
    let mut out = String::new();
    for (k, f) in fields.iter().enumerate() {
        if k > 0 {
            out.push(',');
        }
        if f.contains([',', '"', '\n', '\r']) || f.trim() != *f {
            out.push('"');
            out.push_str(&f.replace('"', "\"\""));
            out.push('"');
        } else {
            out.push_str(f);
        }
    }
    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project() -> (tempfile::TempDir, TextProject) {
        let d = tempfile::tempdir().unwrap();
        fs::create_dir_all(d.path().join("images")).unwrap();
        for n in ["1", "2", "3"] {
            fs::write(d.path().join(format!("images/{n}.jpg")), b"x").unwrap();
        }
        fs::write(
            d.path().join(LABELS_FILE),
            "file,bib,split,status,ann_id\nimages/1.jpg,123,train,ok,7\nimages/2.jpg,,val,unreadable,8\n",
        )
        .unwrap();
        let cfg =
            ProjectConfig { kind: Kind::Text, text_column: "bib".into(), ..Default::default() };
        cfg.save(d.path()).unwrap();
        let p = TextProject::open(d.path()).unwrap();
        (d, p)
    }

    #[test]
    fn csv_round_trips_quotes_and_newlines() {
        let line = csv_line(&["a", "b,c", "say \"hi\"", " x", "l1\nl2"]);
        assert_eq!(parse_csv(&line), vec![vec!["a", "b,c", "say \"hi\"", " x", "l1\nl2"]]);
        assert_eq!(parse_csv("a,b\r\n\r\n,\n"), vec![vec!["a", "b"], vec!["", ""]]);
    }

    #[test]
    fn opens_adds_untracked_images_and_saves_with_history() {
        let (d, mut p) = project();
        assert_eq!(p.rows.len(), 3);
        assert_eq!(p.label(0), &Label::ok("123"));
        assert_eq!(p.label(1).status, Status::Unreadable);
        assert_eq!((p.rows[2].file.as_str(), p.label(2).status), ("images/3.jpg", Status::Todo));
        assert!(p.has_unsaved_changes(), "the added row is recorded by the next save");
        p.set(2, Label::ok("42")).unwrap();
        p.save().unwrap().unwrap();
        let csv = fs::read_to_string(d.path().join(LABELS_FILE)).unwrap();
        assert_eq!(
            csv,
            "file,bib,split,status,ann_id\nimages/1.jpg,123,train,ok,7\n\
             images/2.jpg,,val,unreadable,8\nimages/3.jpg,42,,ok,\n"
        );
        assert!(d.path().join(".fastlabel/history/labels.v001.csv").exists());
        assert!(p.save().unwrap().is_none());
    }

    #[test]
    fn undo_redo_and_crash_replay() {
        let (d, mut p) = project();
        p.set(0, Label::ok("124")).unwrap();
        p.set(1, Label::ok("9")).unwrap();
        assert_eq!(p.undo().unwrap(), Some(1));
        assert_eq!(p.label(1).status, Status::Unreadable);
        assert_eq!(p.redo().unwrap(), Some(1));
        p.undo().unwrap();
        drop(p);
        let p = TextProject::open(d.path()).unwrap();
        assert_eq!(p.label(0), &Label::ok("124"));
        assert_eq!(p.label(1).status, Status::Unreadable);
        assert!(p.can_redo());
    }

    #[test]
    fn rejections_are_saved_and_cleared_by_a_new_reading() {
        let (d, mut p) = project();
        let pred = |t: &str| TextPrediction { text: t.into(), score: 0.9, rejected: false };
        let texts = BTreeMap::from([("images/3.jpg".to_string(), pred("77"))]);
        p.merge_predictions("m".into(), texts.clone()).unwrap();
        p.set(2, Label { rejected: true, ..Default::default() }).unwrap();
        p.save().unwrap();
        let mut p = TextProject::open(d.path()).unwrap();
        assert!(p.label(2).rejected);
        p.merge_predictions("m".into(), texts).unwrap();
        assert!(p.label(2).rejected, "same reading: still rejected");
        p.merge_predictions("m2".into(), BTreeMap::from([("images/3.jpg".into(), pred("78"))]))
            .unwrap();
        assert!(!p.label(2).rejected);
        assert!(!p.has_unsaved_changes());
    }
}
