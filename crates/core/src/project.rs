//! A dataset folder opened for review: annotations, pending edits (with crash
//! journal), review state and predictions.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use crate::coco::{CocoFile, ImageId};
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
/// Trash moves of a save in progress, finished on the next open if the app
/// stopped after writing the annotation file (otherwise the frames would come
/// back as new images).
const MOVES_FILE: &str = "moves.json";

#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
struct PendingMoves {
    /// Annotation file (in the dataset folder) the moves belong to.
    version: String,
    /// Trash folder relative to the dataset folder.
    trash_dir: String,
    trashed: Vec<String>,
}

impl PendingMoves {
    fn run(&self, dir: &Path, retained: &CocoFile) -> TrashReport {
        commit::move_to_trash(dir, &dir.join(&self.trash_dir), &self.trashed, retained)
    }
}

pub struct Project {
    pub dir: PathBuf,
    pub dataset: Dataset,
    /// Annotation file the dataset was loaded from; `None` when imported
    /// from YOLO labels or a plain image folder (not saved yet).
    pub source: Option<VersionFile>,
    pub log: EditLog,
    /// Images found in `images/` but missing from the annotation file, added
    /// on open (id, file name). The next save records them.
    pub added_images: Vec<(ImageId, String)>,
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
                let mut coco = yolo::import(dir)?;
                let classes = crate::config::ProjectConfig::load(dir)?.classes;
                // A new project: its classes instead of the `object` default.
                if !classes.is_empty() && coco.annotations.is_empty() {
                    coco.categories = classes
                        .iter()
                        .enumerate()
                        .map(|(i, name)| crate::coco::CocoCategory {
                            id: i as i64 + 1,
                            name: name.clone(),
                            supercategory: String::new(),
                            extra: Default::default(),
                        })
                        .collect();
                }
                Self::from_coco(dir, coco, None)
            }
        }
    }

    /// Opens a specific annotation file inside `dir`.
    pub fn open_version(dir: &Path, version: VersionFile) -> Result<Self> {
        let coco = CocoFile::load(&version.path)?;
        Self::from_coco(dir, coco, Some(version))
    }

    fn from_coco(dir: &Path, mut coco: CocoFile, source: Option<VersionFile>) -> Result<Self> {
        let source_path = source.as_ref().map_or_else(|| dir.to_path_buf(), |v| v.path.clone());
        let mut warnings = Vec::new();
        // Before scanning images/: a half-done save would leave moved files
        // looking like new images.
        finish_moves(dir, &mut warnings);
        // A YOLO / folder import already lists every image.
        let added_images = match &source {
            Some(v) => {
                let hint = journal::read(&dir.join(STATE_DIR).join(JOURNAL_FILE))?
                    .filter(|r| r.base == v.file_name())
                    .map(|r| r.added_images)
                    .unwrap_or_default();
                let sources =
                    crate::add::read_sources(&dir.join(STATE_DIR).join(crate::add::SOURCES_FILE));
                let added = add_untracked_images(dir, &mut coco, &hint, &mut warnings);
                for img in &mut coco.images {
                    if let Some(p) = sources.get(&img.file_name) {
                        img.extra
                            .insert(crate::add::SOURCE_FIELD.into(), p.source_name.clone().into());
                        if !p.date_added.is_empty() {
                            img.extra
                                .insert(crate::add::DATE_FIELD.into(), p.date_added.clone().into());
                        }
                    }
                }
                added
            }
            None => Vec::new(),
        };
        let (dataset, legacy) = Dataset::new(dir.to_path_buf(), coco, &source_path)?;
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
            added_images,
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
            self.journal =
                Some(Journal::create(&self.journal_path(), &self.base_name(), &self.added_images)?);
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
        self.log.is_dirty() || self.source.is_none() || !self.added_images.is_empty()
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
        let state_dir = self.state_dir();
        let version_stem =
            path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let moves = PendingMoves {
            version: path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
            trash_dir: format!("{STATE_DIR}/trash/{version_stem}"),
            trashed: applied.trashed_files.clone(),
        };
        let moves_path = state_dir.join(MOVES_FILE);
        let has_moves = !moves.trashed.is_empty();
        if has_moves {
            crate::fsutil::write_json_atomic(&moves_path, &moves)?;
        }
        applied.coco.save(&path)?;
        applied.review.save(&state_dir.join(REVIEW_FILE))?;
        let trash = moves.run(&self.dir, &applied.coco);
        if has_moves {
            fs::remove_file(&moves_path).map_err(|e| Error::io(&moves_path, e))?;
        }

        let version = versions::discover(&self.dir)?
            .into_iter()
            .find(|v| v.path == path)
            .ok_or_else(|| Error::invalid(&path, "saved file disappeared"))?;
        let (dataset, _) = Dataset::new(self.dir.clone(), applied.coco, &path)?;
        self.dataset = dataset;
        self.source = Some(version);
        self.review = applied.review;
        self.added_images.clear();
        self.log = EditLog::new(self.dataset.coco.next_annotation_id());
        self.journal = None;
        for f in [self.journal_path(), state_dir.join(crate::add::SOURCES_FILE)] {
            if f.exists() {
                fs::remove_file(&f).map_err(|e| Error::io(&f, e))?;
            }
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

    /// Registers images copied into `images/` (see [`crate::add`]) as new
    /// frames, with their source names; saved with the next version. Returns
    /// their ids. Crash-safe: the source names go to `.fastlabel/sources.json`
    /// and the ids to the journal header, so a replay finds the same frames.
    pub fn register_added(&mut self, copied: &[crate::add::Copied]) -> Result<Vec<ImageId>> {
        let mut coco = self.dataset.coco.clone();
        let first = coco.images.iter().map(|i| i.id).max().unwrap_or(0) + 1;
        let tracked: HashSet<String> = coco.images.iter().map(|i| i.file_name.clone()).collect();
        let mut ids = Vec::new();
        let now = crate::dates::iso(crate::dates::now());
        let new = copied.iter().filter(|c| !tracked.contains(&c.file_name));
        for (next, c) in (first..).zip(new) {
            let path = self.dir.join(&c.file_name);
            let size = imagesize::size(&path).map_err(|e| Error::invalid(&path, e.to_string()))?;
            let mut extra = serde_json::Map::new();
            extra.insert(crate::add::SOURCE_FIELD.into(), c.source_name.clone().into());
            extra.insert(crate::add::DATE_FIELD.into(), now.clone().into());
            coco.images.push(crate::coco::CocoImage {
                id: next,
                file_name: c.file_name.clone(),
                width: size.width as u32,
                height: size.height as u32,
                tags: Vec::new(),
                extra,
            });
            self.added_images.push((next, c.file_name.clone()));
            ids.push(next);
        }
        if ids.is_empty() {
            return Ok(ids);
        }
        let sources_path = self.state_dir().join(crate::add::SOURCES_FILE);
        let mut sources = crate::add::read_sources(&sources_path);
        sources.extend(copied.iter().map(|c| {
            let p =
                crate::add::Pending { source_name: c.source_name.clone(), date_added: now.clone() };
            (c.file_name.clone(), p)
        }));
        crate::add::write_sources(&sources_path, &sources)?;
        let jp = self.journal_path();
        if self.journal.is_some() || jp.exists() {
            self.journal = Some(journal::rewrite(&jp, &self.base_name(), &self.added_images)?);
        }
        let source = self.source.as_ref().map_or_else(|| self.dir.clone(), |v| v.path.clone());
        self.dataset = Dataset::new(self.dir.clone(), coco, &source)?.0;
        if let Some(file) = &self.prediction_file {
            self.predictions = Some(Predictions::from_file(file, &self.dataset));
        }
        Ok(ids)
    }

    /// Replaces all predictions and writes `predictions.json`.
    pub fn set_predictions(&mut self, preds: Predictions) -> Result<()> {
        let file = preds.to_file(&self.dataset);
        file.save(&self.dir.join(PREDICTIONS_FILE))?;
        self.predictions = Some(Predictions::from_file(&file, &self.dataset));
        self.prediction_file = Some(file);
        Ok(())
    }

    /// Replaces the predictions of some images (by image index), keeping
    /// the others, and writes `predictions.json`.
    pub fn merge_predictions(
        &mut self,
        per_image: Vec<(usize, Vec<crate::predictions::PredRef>)>,
        model: &str,
        confidence: f64,
    ) -> Result<()> {
        let mut all =
            self.predictions.clone().unwrap_or_else(|| Predictions::empty(model, confidence));
        for (idx, preds) in per_image {
            all.set_image(idx, preds);
        }
        self.set_predictions(all)
    }

    /// Replaces the predictions of one image and writes `predictions.json`.
    pub fn set_image_predictions(
        &mut self,
        image_idx: usize,
        preds: Vec<crate::predictions::PredRef>,
        model: &str,
        confidence: f64,
    ) -> Result<()> {
        self.merge_predictions(vec![(image_idx, preds)], model, confidence)
    }
}

/// Finishes the file moves of a save interrupted after its annotation file
/// was written; drops them if that file was never written.
fn finish_moves(dir: &Path, warnings: &mut Vec<String>) {
    let path = dir.join(STATE_DIR).join(MOVES_FILE);
    if !path.exists() {
        return;
    }
    match crate::fsutil::read_json::<PendingMoves>(&path) {
        Ok(moves) => {
            // Without its annotation file, the save never happened.
            if let Ok(coco) = CocoFile::load(&dir.join(&moves.version)) {
                let report = moves.run(dir, &coco);
                warnings.push(format!(
                    "finished moving {} files of the interrupted save of {}",
                    report.moved.len(),
                    moves.version
                ));
                for (p, why) in report.failed {
                    warnings.push(format!("could not move {}: {why}", p.display()));
                }
            }
        }
        Err(e) => warnings.push(format!("ignored {}: {e}", path.display())),
    }
    if let Err(e) = fs::remove_file(&path) {
        warnings.push(format!("cannot remove {}: {e}", path.display()));
    }
}

/// Adds images present under `dir/images/` but missing from `coco`, with
/// ids after the existing ones. Ids recorded in the journal (`hint`) are
/// reused so replayed edits land on the same images.
fn add_untracked_images(
    dir: &Path,
    coco: &mut CocoFile,
    hint: &[(ImageId, String)],
    warnings: &mut Vec<String>,
) -> Vec<(ImageId, String)> {
    let images_dir = dir.join("images");
    if !images_dir.is_dir() {
        return Vec::new();
    }
    let files = match yolo::list_images(&images_dir, dir) {
        Ok(f) => f,
        Err(e) => {
            warnings.push(format!("cannot scan images/: {e}"));
            return Vec::new();
        }
    };
    let tracked: HashSet<&str> = coco.images.iter().map(|i| i.file_name.as_str()).collect();
    let untracked: Vec<String> = files
        .iter()
        .map(|p| p.to_string_lossy().replace('\\', "/"))
        .filter(|f| !tracked.contains(f.as_str()))
        .collect();
    if untracked.is_empty() {
        return Vec::new();
    }
    let mut next_id = coco.images.iter().map(|i| i.id).max().unwrap_or(0) + 1;
    let mut used: HashSet<ImageId> = coco.images.iter().map(|i| i.id).collect();
    let mut ordered: Vec<(Option<ImageId>, String)> = hint
        .iter()
        .filter(|(id, f)| untracked.contains(f) && !used.contains(id))
        .map(|(id, f)| (Some(*id), f.clone()))
        .collect();
    for f in &untracked {
        if !ordered.iter().any(|(_, g)| g == f) {
            ordered.push((None, f.clone()));
        }
    }
    let mut added = Vec::new();
    let mut unreadable = 0;
    for (hinted, file_name) in ordered {
        let Ok(size) = imagesize::size(dir.join(&file_name)) else {
            unreadable += 1;
            continue;
        };
        let id = hinted.unwrap_or_else(|| {
            while used.contains(&next_id) {
                next_id += 1;
            }
            next_id
        });
        used.insert(id);
        coco.images.push(crate::coco::CocoImage {
            id,
            file_name: file_name.clone(),
            width: size.width as u32,
            height: size.height as u32,
            tags: Vec::new(),
            extra: Default::default(),
        });
        added.push((id, file_name));
    }
    if !added.is_empty() {
        warnings.push(format!(
            "{} new images found in images/ and added (not annotated yet; saved with the next version)",
            added.len()
        ));
    }
    if unreadable > 0 {
        warnings.push(format!(
            "{unreadable} files in images/ are not readable images and were skipped"
        ));
    }
    added
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
            fs::write(
                dir.join(format!("images/train/{i:04}.jpg")),
                crate::yolo::tests::png_header(100, 100),
            )
            .unwrap();
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
    fn split_moves_are_saved_as_fields_and_files_stay() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        write_dataset(dir);
        let mut p = Project::open(dir).unwrap();
        p.record(Op::SetSplit { image: 2, split: Some(crate::Split::Val) }).unwrap();
        let report = p.save().unwrap().unwrap();
        assert_eq!(report.summary.moved_frames, 1);
        assert!(dir.join("images/train/0002.jpg").exists(), "files never move");
        assert_eq!(p.dataset.split_of(1), Some(crate::Split::Val));
        let p = Project::open(dir).unwrap();
        assert_eq!(p.dataset.split_of(1), Some(crate::Split::Val));
        assert_eq!(p.dataset.split_of(0), Some(crate::Split::Train));
        assert!(p.added_images.is_empty() && p.warnings.is_empty(), "{:?}", p.warnings);
    }

    #[test]
    fn interrupted_save_moves_are_finished_on_open() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        write_dataset(dir);
        // Crash after writing v002 (image 1 deleted) but before the trash move.
        let mut coco = CocoFile::load(&dir.join("annotations.v001.json")).unwrap();
        coco.images.retain(|i| i.id == 2);
        coco.annotations.retain(|a| a.image_id == 2);
        coco.save(&dir.join("annotations.v002.json")).unwrap();
        let moves = PendingMoves {
            version: "annotations.v002.json".into(),
            trash_dir: ".fastlabel/trash/annotations.v002".into(),
            trashed: vec!["images/train/0001.jpg".into()],
        };
        crate::fsutil::write_json_atomic(&dir.join(STATE_DIR).join(MOVES_FILE), &moves).unwrap();

        let p = Project::open(dir).unwrap();
        assert!(p.added_images.is_empty(), "the deleted frame does not come back");
        assert!(dir.join(".fastlabel/trash/annotations.v002/images/train/0001.jpg").exists());
        assert!(!dir.join(STATE_DIR).join(MOVES_FILE).exists());
        assert!(p.warnings.iter().any(|w| w.contains("interrupted save")), "{:?}", p.warnings);

        // Moves of a save whose annotation file was never written are dropped.
        let moves = PendingMoves {
            version: "annotations.v003.json".into(),
            trashed: vec!["images/train/0002.jpg".into()],
            ..moves
        };
        crate::fsutil::write_json_atomic(&dir.join(STATE_DIR).join(MOVES_FILE), &moves).unwrap();
        let p = Project::open(dir).unwrap();
        assert!(dir.join("images/train/0002.jpg").exists(), "never trashed");
        assert!(!dir.join(STATE_DIR).join(MOVES_FILE).exists());
        assert_eq!(p.dataset.images().len(), 1);
    }

    #[test]
    fn added_images_survive_a_crash_and_are_saved_with_sources() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        write_dataset(dir);
        let incoming = dir.join("incoming");
        fs::create_dir_all(&incoming).unwrap();
        fs::write(incoming.join("x.png"), crate::yolo::tests::png_header(30, 20)).unwrap();
        let mut p = Project::open(dir).unwrap();
        p.record(Op::FrameDelete { image: 1 }).unwrap(); // journal exists before the add
        let r = crate::add::copy_sources(
            dir,
            &crate::add::collect(std::slice::from_ref(&incoming)),
            |_, _| true,
        );
        let ids = p.register_added(&r.copied).unwrap();
        assert_eq!(ids, vec![3]);
        assert_eq!(p.dataset.image(3).unwrap().width, 30);
        p.record(Op::SetSplit { image: 3, split: Some(crate::Split::Val) }).unwrap();
        drop(p); // crash

        let mut p = Project::open(dir).unwrap();
        let img = p.dataset.image(3).expect("same id after replay");
        assert_eq!(img.extra[crate::add::SOURCE_FIELD], "incoming/x.png");
        let added = crate::dates::parse_iso(img.extra[crate::add::DATE_FIELD].as_str().unwrap());
        assert!(added.is_some_and(|t| (crate::dates::now() - t).abs() < 60));
        assert_eq!(p.log.state().splits.get(&3), Some(&crate::Split::Val));
        assert!(p.log.state().is_frame_deleted(1));
        p.save().unwrap().unwrap();
        assert!(!dir.join(STATE_DIR).join(crate::add::SOURCES_FILE).exists());
        let p = Project::open(dir).unwrap();
        let img = p.dataset.image(3).unwrap();
        assert_eq!(img.extra[crate::add::SOURCE_FIELD], "incoming/x.png");
        assert_eq!(p.dataset.split_of(p.dataset.image_idx(3).unwrap()), Some(crate::Split::Val));
        assert!(img.file_name.starts_with("images/") && img.file_name.len() == 27);
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
        let cfg = crate::config::ProjectConfig {
            classes: vec!["car".into(), "truck".into()],
            ..Default::default()
        };
        cfg.save(dir).unwrap();
        let mut p = Project::open(dir).unwrap();
        assert!(p.source.is_none());
        assert!(p.has_unsaved_changes());
        let names: Vec<_> = p.dataset.categories().iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["car", "truck"]);
        let report = p.save().unwrap().unwrap();
        assert_eq!(report.path, dir.join("annotations.v001.json"));
    }

    #[test]
    fn untracked_images_are_added_and_saved() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        write_dataset(dir); // images 1 and 2 (jpg stubs, tracked)
        fs::write(dir.join("images/train/new_b.png"), crate::yolo::tests::png_header(64, 32))
            .unwrap();
        fs::write(dir.join("images/train/broken.png"), b"nope").unwrap();
        let mut p = Project::open(dir).unwrap();
        assert_eq!(p.added_images, vec![(3, "images/train/new_b.png".to_string())]);
        let img = p.dataset.image(3).unwrap();
        assert_eq!((img.width, img.height), (64, 32));
        assert!(p.warnings.iter().any(|w| w.contains("1 new images")));
        assert!(p.warnings.iter().any(|w| w.contains("not readable")));
        assert!(p.has_unsaved_changes());

        // Edit the new image, crash, add another file that sorts first:
        // the replayed edit must still land on new_b.png.
        let ann = p.log.alloc_ann_id();
        p.record(Op::AddBox {
            ann,
            image: 3,
            category: 1,
            bbox: crate::geom::BBox::new(1.0, 1.0, 10.0, 10.0),
        })
        .unwrap();
        drop(p);
        fs::write(dir.join("images/train/new_a.png"), crate::yolo::tests::png_header(8, 8))
            .unwrap();
        let mut p = Project::open(dir).unwrap();
        assert_eq!(p.dataset.image(3).unwrap().file_name, "images/train/new_b.png");
        assert_eq!(p.dataset.image(4).unwrap().file_name, "images/train/new_a.png");
        p.save().unwrap();
        assert!(p.added_images.is_empty());
        let p = Project::open(dir).unwrap();
        assert!(p.added_images.is_empty(), "now tracked");
        assert_eq!(p.dataset.images().len(), 4);
        assert_eq!(p.dataset.annotations().iter().filter(|a| a.image_id == 3).count(), 1);
    }
}
