//! A dataset folder opened for review: annotations, pending edits (with crash
//! journal), review state and predictions.

use std::fs;
use std::path::{Path, PathBuf};

use crate::coco::CocoFile;
use crate::commit::{self, Summary, TrashReport};
use crate::dataset::Dataset;
use crate::edits::{AnnMark, EditLog, Op};
use crate::error::{Error, Result};
use crate::journal::{self, Entry, JOURNAL_FILE, Journal};
use crate::predictions::{PREDICTIONS_FILE, PredictionFile, Predictions};
use crate::review::{REVIEW_FILE, ReviewState};
use crate::versions::{self, VersionFile};
use crate::yolo;

pub const STATE_DIR: &str = ".fastlabel";

pub struct Project {
    pub dir: PathBuf,
    pub dataset: Dataset,
    /// Annotation file the dataset was loaded from; `None` when imported
    /// from YOLO labels or a plain image folder (not saved yet).
    pub source: Option<VersionFile>,
    pub log: EditLog,
    pub review: ReviewState,
    prediction_file: Option<PredictionFile>,
    pub predictions: Option<Predictions>,
    /// Non-fatal problems found while opening (shown to the user).
    pub warnings: Vec<String>,
    journal: Option<Journal>,
}

#[derive(Debug)]
pub struct SaveReport {
    pub path: PathBuf,
    pub summary: Summary,
    pub trash: TrashReport,
}

impl Project {
    /// Opens the latest annotation version in `dir`, or imports YOLO labels /
    /// plain images when there is none.
    pub fn open(dir: &Path) -> Result<Self> {
        let files = versions::discover(dir)?;
        match versions::latest(&files) {
            Some(v) => Self::open_version(dir, v.clone()),
            None => {
                let coco = yolo::import(dir)?;
                Self::from_coco(dir, coco, None)
            }
        }
    }

    /// Opens a specific annotation file inside `dir`.
    pub fn open_version(dir: &Path, version: VersionFile) -> Result<Self> {
        let coco = CocoFile::load(&version.path)?;
        Self::from_coco(dir, coco, Some(version))
    }

    fn from_coco(dir: &Path, coco: CocoFile, source: Option<VersionFile>) -> Result<Self> {
        let source_path = source.as_ref().map_or_else(|| dir.to_path_buf(), |v| v.path.clone());
        let (dataset, legacy) = Dataset::new(dir.to_path_buf(), coco, &source_path)?;
        let mut warnings = Vec::new();
        let mut log = EditLog::new(dataset.coco.next_annotation_id());
        for ann in &legacy.deleted_annotations {
            log.push(Op::AnnMark { ann: *ann, mark: AnnMark::Delete });
        }
        for image in &legacy.deleted_frames {
            log.push(Op::FrameDelete { image: *image });
        }
        let n_legacy = legacy.deleted_annotations.len() + legacy.deleted_frames.len();
        if n_legacy > 0 {
            warnings.push(format!(
                "{n_legacy} deletion marks from the previous tool loaded as pending edits"
            ));
        }

        let state_dir = dir.join(STATE_DIR);
        let review = match ReviewState::load(&state_dir.join(REVIEW_FILE)) {
            Ok(r) => r,
            Err(e) => {
                warnings.push(format!("review state ignored: {e}"));
                ReviewState::default()
            }
        };

        let mut project = Self {
            dir: dir.to_path_buf(),
            dataset,
            source,
            log,
            review,
            prediction_file: None,
            predictions: None,
            warnings,
            journal: None,
        };
        project.load_predictions();
        project.recover_journal()?;
        Ok(project)
    }

    pub fn state_dir(&self) -> PathBuf {
        self.dir.join(STATE_DIR)
    }

    fn journal_path(&self) -> PathBuf {
        self.state_dir().join(JOURNAL_FILE)
    }

    /// Name of the base annotation file, used to tie the journal to it.
    pub fn base_name(&self) -> String {
        self.source.as_ref().map_or_else(|| "<imported>".into(), VersionFile::file_name)
    }

    fn load_predictions(&mut self) {
        let path = self.dir.join(PREDICTIONS_FILE);
        if !path.exists() {
            return;
        }
        match PredictionFile::load(&path) {
            Ok(file) => {
                let preds = Predictions::from_file(&file, &self.dataset);
                self.warnings.extend(preds.warnings.iter().cloned());
                self.prediction_file = Some(file);
                self.predictions = Some(preds);
            }
            Err(e) => self.warnings.push(format!("predictions ignored: {e}")),
        }
    }

    fn recover_journal(&mut self) -> Result<()> {
        let path = self.journal_path();
        let Some(recorded) = journal::read(&path)? else {
            return Ok(());
        };
        if recorded.base == self.base_name() {
            let n = recorded.entries.len();
            journal::replay(&recorded.entries, &mut self.log);
            self.journal = Some(Journal::open_append(&path)?);
            if n > 0 {
                self.warnings.push(format!("recovered {n} unsaved actions from the last session"));
            }
        } else {
            let stale =
                path.with_file_name(format!("journal.{}.stale.jsonl", sanitize(&recorded.base)));
            fs::rename(&path, &stale).map_err(|e| Error::io(&path, e))?;
            self.warnings.push(format!(
                "unsaved actions made on {} were set aside in {}",
                recorded.base,
                stale.display()
            ));
        }
        Ok(())
    }

    fn journal(&mut self, entry: &Entry) -> Result<()> {
        if self.journal.is_none() {
            self.journal = Some(Journal::create(&self.journal_path(), &self.base_name())?);
        }
        self.journal.as_mut().expect("journal created").append(entry)
    }

    /// Records a user action. The edit is always applied in memory; an error
    /// only means it could not be journaled (crash protection is degraded).
    pub fn record(&mut self, op: Op) -> Result<()> {
        self.log.push(op.clone());
        self.journal(&Entry::Op { op })
    }

    pub fn undo(&mut self) -> Result<Option<Op>> {
        let Some(op) = self.log.undo().cloned() else {
            return Ok(None);
        };
        self.journal(&Entry::Undo)?;
        Ok(Some(op))
    }

    pub fn redo(&mut self) -> Result<Option<Op>> {
        let Some(op) = self.log.redo().cloned() else {
            return Ok(None);
        };
        self.journal(&Entry::Redo)?;
        Ok(Some(op))
    }

    /// True when saving would produce a new version.
    pub fn has_unsaved_changes(&self) -> bool {
        self.log.is_dirty() || self.source.is_none()
    }

    /// Writes the next `annotations.vNNN.json` with all pending edits applied,
    /// then the review state, then moves deleted frames to the trash. Returns
    /// `Ok(None)` when there is nothing to save.
    pub fn save(&mut self) -> Result<Option<SaveReport>> {
        if !self.has_unsaved_changes() {
            return Ok(None);
        }
        let applied = commit::apply(&self.dataset, self.log.state(), &self.review);
        let files = versions::discover(&self.dir)?;
        let path = versions::next_path(&self.dir, &files);
        applied.coco.save(&path)?;
        let state_dir = self.state_dir();
        applied.review.save(&state_dir.join(REVIEW_FILE))?;

        let version_stem =
            path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let trash = commit::move_to_trash(
            &self.dir,
            &state_dir.join("trash").join(version_stem),
            &applied.trashed_files,
            &applied.coco,
        );

        let version = versions::discover(&self.dir)?
            .into_iter()
            .find(|v| v.path == path)
            .ok_or_else(|| Error::invalid(&path, "saved file disappeared"))?;
        let (dataset, _) = Dataset::new(self.dir.clone(), applied.coco, &path)?;
        self.dataset = dataset;
        self.source = Some(version);
        self.review = applied.review;
        self.log = EditLog::new(self.dataset.coco.next_annotation_id());
        self.journal = None;
        let jp = self.journal_path();
        if jp.exists() {
            fs::remove_file(&jp).map_err(|e| Error::io(&jp, e))?;
        }

        if let Some(file) = &mut self.prediction_file {
            if !applied.trashed_files.is_empty() {
                let ds = &self.dataset;
                file.images.retain(|k, _| k.parse().ok().and_then(|id| ds.image(id)).is_some());
                file.save(&self.dir.join(PREDICTIONS_FILE))?;
            }
            self.predictions = Some(Predictions::from_file(file, &self.dataset));
        }
        Ok(Some(SaveReport { path, summary: applied.summary, trash }))
    }

    /// Replaces all predictions and writes `predictions.json`.
    pub fn set_predictions(&mut self, preds: Predictions) -> Result<()> {
        let file = preds.to_file(&self.dataset);
        file.save(&self.dir.join(PREDICTIONS_FILE))?;
        self.predictions = Some(Predictions::from_file(&file, &self.dataset));
        self.prediction_file = Some(file);
        Ok(())
    }

    /// Replaces the predictions of one image and writes `predictions.json`.
    pub fn set_image_predictions(
        &mut self,
        image_idx: usize,
        preds: Vec<crate::predictions::PredRef>,
        model: &str,
        confidence: f64,
    ) -> Result<()> {
        let mut all =
            self.predictions.clone().unwrap_or_else(|| Predictions::empty(model, confidence));
        all.set_image(image_idx, preds);
        self.set_predictions(all)
    }
}

fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '-' { c } else { '_' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::test_util::dataset;
    use crate::edits::PredMark;
    use crate::geom::BBox;
    use crate::predictions::PredRef;
    use crate::versions::VersionKind;

    fn write_dataset(dir: &Path) {
        let mut coco = dataset(
            2,
            &[(1, BBox::new(0.0, 0.0, 10.0, 10.0)), (2, BBox::new(5.0, 5.0, 10.0, 10.0))],
        )
        .coco;
        coco.annotations[1].extra.insert("to_delete".into(), true.into());
        coco.save(&dir.join("annotations.v001.json")).unwrap();
        fs::create_dir_all(dir.join("images/train")).unwrap();
        for i in 1..=2 {
            fs::write(dir.join(format!("images/train/{i:04}.jpg")), b"img").unwrap();
        }
    }

    #[test]
    fn open_record_crash_recover_save() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        write_dataset(dir);

        let mut p = Project::open(dir).unwrap();
        assert_eq!(p.log.state().ann_mark(2), Some(AnnMark::Delete), "legacy mark loaded");
        p.record(Op::FrameDelete { image: 1 }).unwrap();
        p.record(Op::AnnMark { ann: 2, mark: AnnMark::Delete }).unwrap(); // un-mark legacy
        p.undo().unwrap();
        drop(p); // "crash"

        let mut p = Project::open(dir).unwrap();
        assert!(p.log.state().is_frame_deleted(1));
        assert_eq!(p.log.state().ann_mark(2), Some(AnnMark::Delete));
        assert!(p.warnings.iter().any(|w| w.contains("recovered")));

        let report = p.save().unwrap().unwrap();
        assert_eq!(report.path, dir.join("annotations.v002.json"));
        assert_eq!(report.summary.deleted_frames, 1);
        assert_eq!(
            report.trash.moved,
            vec![dir.join(".fastlabel/trash/annotations.v002/images/train/0001.jpg")]
        );
        assert!(!dir.join("images/train/0001.jpg").exists());
        assert!(dir.join("images/train/0002.jpg").exists());
        assert!(p.dataset.annotations().is_empty());
        assert_eq!(p.dataset.images().len(), 1);
        assert!(!p.has_unsaved_changes());
        assert!(!dir.join(".fastlabel").join(JOURNAL_FILE).exists());
        assert!(p.save().unwrap().is_none());

        let saved = fs::read_to_string(dir.join("annotations.v002.json")).unwrap();
        assert!(!saved.contains("to_delete"));
        let p = Project::open(dir).unwrap();
        assert_eq!(p.source.unwrap().kind, VersionKind::Numbered(2));
        assert!(p.warnings.is_empty(), "{:?}", p.warnings);
    }

    #[test]
    fn journal_for_another_base_is_set_aside() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        write_dataset(dir);
        let mut p = Project::open(dir).unwrap();
        p.record(Op::FrameDelete { image: 2 }).unwrap();
        drop(p);
        // Someone saved a newer version meanwhile.
        fs::copy(dir.join("annotations.v001.json"), dir.join("annotations.v002.json")).unwrap();
        let p = Project::open(dir).unwrap();
        assert!(!p.log.state().is_frame_deleted(2));
        assert!(p.warnings.iter().any(|w| w.contains("set aside")));
    }

    #[test]
    fn predictions_persist_and_accept_creates_annotation() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        write_dataset(dir);
        let mut p = Project::open(dir).unwrap();
        let pred = PredRef {
            image_id: 1,
            category_id: 1,
            bbox: BBox::new(50.0, 50.0, 10.0, 10.0),
            score: 0.7,
        };
        p.set_image_predictions(0, vec![pred], "m.onnx", 0.1).unwrap();
        p.record(Op::PredMark { pred, mark: PredMark::Accept }).unwrap();
        p.save().unwrap();
        assert!(p.dataset.annotations().iter().any(|a| a.bbox == pred.bbox));

        let p = Project::open(dir).unwrap();
        let preds = p.predictions.unwrap();
        assert_eq!(preds.of(0).unwrap(), &[pred]);
        assert!(preds.of(1).is_none());
    }

    #[test]
    fn imports_plain_folder_and_first_save_writes_v001() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        fs::create_dir_all(dir.join("images")).unwrap();
        fs::write(dir.join("images/a.png"), crate::yolo::tests::png_header(20, 10)).unwrap();
        let mut p = Project::open(dir).unwrap();
        assert!(p.source.is_none());
        assert!(p.has_unsaved_changes());
        let report = p.save().unwrap().unwrap();
        assert_eq!(report.path, dir.join("annotations.v001.json"));
    }
}
