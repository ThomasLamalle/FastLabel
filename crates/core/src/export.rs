//! Export to training formats. The split is image data (see
//! [`crate::dataset::SPLIT_FIELD`]); exports lay files out by split:
//!
//! - YOLO (Ultralytics, YOLO26): `data.yaml`, `images/<split>/`, `labels/<split>/*.txt`.
//! - COCO per split (Roboflow layout, RF-DETR): `train/`, `valid/`, `test/`,
//!   each with `_annotations.coco.json` next to its images.
//! - COCO single file: one annotation file with `split` fields + `images/<split>/`.
//! - Ultralytics NDJSON: `<name>.ndjson` + `images/<split>/` (local `path`).
//!
//! Written to a zip (atomically) or to a new folder (built in a temporary
//! sibling, then renamed; images hardlinked when possible, else copied).
//! Optionally, images larger than a given long side are downscaled (labels
//! follow), so training does not decode and resize full frames every epoch.
//! A cancelled export leaves nothing behind.

use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipWriter};

use crate::coco::{CategoryId, CocoAnnotation, CocoFile, CocoImage};
use crate::dataset::{SPLIT_FIELD, Split};
use crate::error::{Error, Result};
use crate::fsutil;
use crate::geom::{self, BBox};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ExportFormat {
    #[default]
    Yolo,
    CocoPerSplit,
    CocoSingle,
    Ndjson,
}

impl ExportFormat {
    pub const ALL: [ExportFormat; 4] = [
        ExportFormat::Yolo,
        ExportFormat::CocoPerSplit,
        ExportFormat::CocoSingle,
        ExportFormat::Ndjson,
    ];

    pub fn label(self) -> &'static str {
        match self {
            ExportFormat::Yolo => "YOLO (Ultralytics / YOLO26)",
            ExportFormat::CocoPerSplit => "COCO per split (Roboflow layout, RF-DETR)",
            ExportFormat::CocoSingle => "COCO single file (split field)",
            ExportFormat::Ndjson => "Ultralytics NDJSON",
        }
    }

    /// Short name for default output names.
    pub fn slug(self) -> &'static str {
        match self {
            ExportFormat::Yolo => "yolo",
            ExportFormat::CocoPerSplit => "coco",
            ExportFormat::CocoSingle => "coco1",
            ExportFormat::Ndjson => "ndjson",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Target {
    Zip(PathBuf),
    /// Must not exist yet.
    Folder(PathBuf),
}

/// Batch image resizer: `(source, width, height)` → the image at exactly
/// that size, encoded in the source's format. Batches let the caller
/// work in parallel (core has no threads).
pub type ResizeFn = fn(&[(PathBuf, u32, u32)]) -> Vec<std::result::Result<Vec<u8>, String>>;

#[derive(Clone, Copy, Debug)]
pub struct Resize {
    /// Long side of the exported images; smaller ones are exported as is.
    pub max_side: u32,
    pub resize: ResizeFn,
}

/// Images per [`ResizeFn`] call.
const RESIZE_BATCH: usize = 32;

#[derive(Clone, Debug)]
pub struct ExportOptions {
    pub format: ExportFormat,
    /// Splits to export; frames of other splits (or without one) are left out.
    pub splits: Vec<Split>,
    /// Dataset name (NDJSON header and file name).
    pub name: String,
    pub resize: Option<Resize>,
    /// Boxes less tall than this (source pixels) are left out; 0 keeps all.
    pub min_box_height: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExportReport {
    /// Images written per split (train, val, test).
    pub per_split: [usize; 3],
    /// Image files not found (left out).
    pub missing: Vec<String>,
    /// Frames without a split (left out).
    pub no_split: usize,
    pub cancelled: bool,
}

impl ExportReport {
    pub fn images(&self) -> usize {
        self.per_split.iter().sum()
    }
}

/// One exported frame.
struct Entry<'a> {
    image: &'a CocoImage,
    split: Split,
    /// File name inside its split folder (unique per split).
    name: String,
    /// Exported (width, height).
    size: (u32, u32),
}

impl Entry<'_> {
    fn resized(&self) -> bool {
        self.size != (self.image.width, self.image.height)
    }

    /// The annotation in exported pixels.
    fn scaled(&self, a: &CocoAnnotation) -> CocoAnnotation {
        let mut a = a.clone();
        if self.resized() {
            let sx = f64::from(self.size.0) / f64::from(self.image.width);
            let sy = f64::from(self.size.1) / f64::from(self.image.height);
            let b = a.bbox;
            a.bbox = BBox::new(b.x * sx, b.y * sy, b.w * sx, b.h * sy);
            a.area *= sx * sy;
        }
        a
    }
}

fn coco_dir(split: Split) -> &'static str {
    match split {
        Split::Train => "train",
        Split::Val => "valid",
        Split::Test => "test",
    }
}

/// Frames to export with unique per-split file names (`<id>_<name>` when two
/// frames of a split share a base name).
fn entries<'a>(
    root: &Path,
    coco: &'a CocoFile,
    opts: &ExportOptions,
    report: &mut ExportReport,
) -> Vec<Entry<'a>> {
    let mut taken: HashSet<(Split, String)> = HashSet::new();
    let mut out = Vec::new();
    for image in &coco.images {
        let Some(split) = Split::of_image(image) else {
            report.no_split += 1;
            continue;
        };
        if !opts.splits.contains(&split) {
            continue;
        }
        if !root.join(&image.file_name).is_file() {
            report.missing.push(image.file_name.clone());
            continue;
        }
        let base = image.file_name.rsplit('/').next().unwrap_or(&image.file_name).to_string();
        let name = if taken.insert((split, base.clone())) {
            base
        } else {
            let n = format!("{}_{base}", image.id);
            taken.insert((split, n.clone()));
            n
        };
        let (w, h) = (image.width, image.height);
        let size = opts.resize.map_or((w, h), |r| geom::fit(w, h, r.max_side));
        out.push(Entry { image, split, name, size });
    }
    out
}

/// YOLO class index per category id (categories in id order).
fn class_indices(coco: &CocoFile) -> (BTreeMap<CategoryId, usize>, Vec<String>) {
    let mut cats: Vec<_> = coco.categories.iter().collect();
    cats.sort_by_key(|c| c.id);
    let index = cats.iter().enumerate().map(|(i, c)| (c.id, i)).collect();
    (index, cats.iter().map(|c| c.name.clone()).collect())
}

/// Normalized (cx, cy, w, h), clipped to the image; `None` for empty boxes.
fn normalized(b: &BBox, w: u32, h: u32) -> Option<[f64; 4]> {
    let b = b.clip(f64::from(w), f64::from(h))?;
    let (w, h) = (f64::from(w), f64::from(h));
    Some([(b.x + b.w / 2.0) / w, (b.y + b.h / 2.0) / h, b.w / w, b.h / h])
}

/// Destination files.
trait Sink {
    fn text(&mut self, rel: &str, bytes: &[u8]) -> Result<()>;
    fn image(&mut self, rel: &str, src: &Path) -> Result<()>;
    fn image_bytes(&mut self, rel: &str, bytes: &[u8]) -> Result<()>;
}

struct ZipSink<'w, W: Write + io::Seek> {
    zip: ZipWriter<&'w mut W>,
    out: PathBuf,
}

impl<W: Write + io::Seek> Sink for ZipSink<'_, W> {
    fn text(&mut self, rel: &str, bytes: &[u8]) -> Result<()> {
        let opts = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
        self.zip
            .start_file(rel, opts)
            .map_err(|e| Error::Zip { path: self.out.clone(), source: e })?;
        self.zip.write_all(bytes).map_err(|e| Error::io(&self.out, e))
    }

    fn image(&mut self, rel: &str, src: &Path) -> Result<()> {
        // JPEGs don't shrink: stored.
        let opts = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
        let mut file = File::open(src).map_err(|e| Error::io(src, e))?;
        self.zip
            .start_file(rel, opts)
            .map_err(|e| Error::Zip { path: self.out.clone(), source: e })?;
        io::copy(&mut file, &mut self.zip).map_err(|e| Error::io(src, e))?;
        Ok(())
    }

    fn image_bytes(&mut self, rel: &str, bytes: &[u8]) -> Result<()> {
        let opts = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
        self.zip
            .start_file(rel, opts)
            .map_err(|e| Error::Zip { path: self.out.clone(), source: e })?;
        self.zip.write_all(bytes).map_err(|e| Error::io(&self.out, e))
    }
}

struct FolderSink {
    dir: PathBuf,
}

impl FolderSink {
    fn path(&self, rel: &str) -> Result<PathBuf> {
        let p = self.dir.join(rel);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }
        Ok(p)
    }
}

impl Sink for FolderSink {
    fn text(&mut self, rel: &str, bytes: &[u8]) -> Result<()> {
        let p = self.path(rel)?;
        fs::write(&p, bytes).map_err(|e| Error::io(&p, e))
    }

    fn image(&mut self, rel: &str, src: &Path) -> Result<()> {
        let p = self.path(rel)?;
        if fs::hard_link(src, &p).is_ok() {
            return Ok(());
        }
        fs::copy(src, &p).map(|_| ()).map_err(|e| Error::io(&p, e))
    }

    fn image_bytes(&mut self, rel: &str, bytes: &[u8]) -> Result<()> {
        self.text(rel, bytes)
    }
}

/// Writes the export. `progress(done, total)` returns false to cancel.
pub fn export(
    root: &Path,
    coco: &CocoFile,
    opts: &ExportOptions,
    target: &Target,
    mut progress: impl FnMut(usize, usize) -> bool,
) -> Result<ExportReport> {
    let mut report = ExportReport::default();
    let entries = entries(root, coco, opts, &mut report);
    let mut write_all = |sink: &mut dyn Sink, report: &mut ExportReport| -> Result<bool> {
        write_texts(sink, coco, opts, &entries)?;
        let total = entries.len();
        for (c, chunk) in entries.chunks(RESIZE_BATCH).enumerate() {
            let jobs: Vec<_> = chunk
                .iter()
                .filter(|e| e.resized())
                .map(|e| (root.join(&e.image.file_name), e.size.0, e.size.1))
                .collect();
            let mut resized = match opts.resize {
                Some(r) if !jobs.is_empty() => (r.resize)(&jobs),
                _ => Vec::new(),
            }
            .into_iter();
            for (k, e) in chunk.iter().enumerate() {
                if !progress(c * RESIZE_BATCH + k, total) {
                    return Ok(false);
                }
                let (rel, src) = (image_path(opts.format, e), root.join(&e.image.file_name));
                if e.resized() {
                    let bytes = resized.next().unwrap_or_else(|| Err("not resized".into()));
                    sink.image_bytes(&rel, &bytes.map_err(|m| Error::invalid(&src, m))?)?;
                } else {
                    sink.image(&rel, &src)?;
                }
                report.per_split[e.split as usize] += 1;
            }
        }
        progress(total, total);
        Ok(true)
    };
    match target {
        Target::Zip(out) => {
            let mut cancelled = false;
            let result = fsutil::write_atomic(out, |w| {
                let mut sink = ZipSink { zip: ZipWriter::new(&mut *w), out: out.clone() };
                if !write_all(&mut sink, &mut report)? {
                    cancelled = true;
                    return Err(Error::invalid(out, "export cancelled"));
                }
                sink.zip.finish().map_err(|e| Error::Zip { path: out.clone(), source: e })?;
                Ok(())
            });
            match result {
                Err(_) if cancelled => {
                    report.cancelled = true;
                    report.per_split = [0; 3];
                    Ok(report)
                }
                r => r.map(|()| report),
            }
        }
        Target::Folder(out) => {
            if out.exists() {
                return Err(Error::invalid(out, "already exists"));
            }
            let name =
                out.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            let tmp = out.with_file_name(format!(".{name}.partial-{}", std::process::id()));
            let mut sink = FolderSink { dir: tmp.clone() };
            let result = write_all(&mut sink, &mut report);
            match result {
                Ok(true) => {
                    fs::rename(&tmp, out).map_err(|e| Error::io(out, e))?;
                    Ok(report)
                }
                Ok(false) => {
                    let _ = fs::remove_dir_all(&tmp);
                    report.cancelled = true;
                    report.per_split = [0; 3];
                    Ok(report)
                }
                Err(e) => {
                    let _ = fs::remove_dir_all(&tmp);
                    Err(e)
                }
            }
        }
    }
}

fn image_path(format: ExportFormat, e: &Entry) -> String {
    match format {
        ExportFormat::CocoPerSplit => format!("{}/{}", coco_dir(e.split), e.name),
        _ => format!("images/{}/{}", e.split.name(), e.name),
    }
}

fn json_bytes(v: &impl serde::Serialize) -> Vec<u8> {
    let mut out = serde_json::to_vec_pretty(v).expect("JSON serializes");
    out.push(b'\n');
    out
}

/// Annotations of the exported frames, by image id, without the boxes
/// below `min_height`.
fn annotations_of<'a>(
    coco: &'a CocoFile,
    entries: &[Entry],
    min_height: f64,
) -> BTreeMap<i64, Vec<&'a CocoAnnotation>> {
    let ids: HashSet<i64> = entries.iter().map(|e| e.image.id).collect();
    let mut by_image: BTreeMap<i64, Vec<&CocoAnnotation>> = BTreeMap::new();
    for a in &coco.annotations {
        if ids.contains(&a.image_id) && a.bbox.h >= min_height {
            by_image.entry(a.image_id).or_default().push(a);
        }
    }
    by_image
}

fn write_texts(
    sink: &mut dyn Sink,
    coco: &CocoFile,
    opts: &ExportOptions,
    entries: &[Entry],
) -> Result<()> {
    let anns = annotations_of(coco, entries, opts.min_box_height);
    let (index, names) = class_indices(coco);
    let yolo_lines = |e: &Entry, decimals: usize| -> Vec<(usize, [f64; 4])> {
        anns.get(&e.image.id)
            .into_iter()
            .flatten()
            .filter_map(|a| {
                let n = normalized(&a.bbox, e.image.width, e.image.height)?;
                let r = |v: f64| {
                    (v * 10f64.powi(decimals as i32)).round() / 10f64.powi(decimals as i32)
                };
                Some((index[&a.category_id], n.map(r)))
            })
            .collect()
    };
    let splits: Vec<Split> = Split::ALL.into_iter().filter(|s| opts.splits.contains(s)).collect();
    match opts.format {
        ExportFormat::Yolo => {
            let mut yaml =
                String::from("# Exported by FastLabel. Paths are relative to this file.\n");
            for s in &splits {
                yaml.push_str(&format!("{}: images/{}\n", s.name(), s.name()));
            }
            yaml.push_str("names:\n");
            for (i, n) in names.iter().enumerate() {
                yaml.push_str(&format!("  {i}: {}\n", yaml_str(n)));
            }
            sink.text("data.yaml", yaml.as_bytes())?;
            for e in entries {
                let stem = e.name.rsplit_once('.').map_or(e.name.as_str(), |(s, _)| s);
                let body: String = yolo_lines(e, 6)
                    .iter()
                    .map(|(c, [x, y, w, h])| format!("{c} {x:.6} {y:.6} {w:.6} {h:.6}\n"))
                    .collect();
                sink.text(&format!("labels/{}/{stem}.txt", e.split.name()), body.as_bytes())?;
            }
        }
        ExportFormat::Ndjson => {
            let class_names: Map<String, Value> =
                names.iter().enumerate().map(|(i, n)| (i.to_string(), n.clone().into())).collect();
            let header = json!({
                "type": "dataset", "task": "detect", "name": opts.name,
                "class_names": class_names, "path": ".",
            });
            let mut out = serde_json::to_string(&header).expect("JSON serializes");
            out.push('\n');
            for e in entries {
                let boxes: Vec<Value> = yolo_lines(e, 5)
                    .iter()
                    .map(|(c, [x, y, w, h])| json!([c, x, y, w, h]))
                    .collect();
                let line = json!({
                    "type": "image", "file": e.name, "width": e.size.0,
                    "height": e.size.1, "split": e.split.name(),
                    "annotations": {"boxes": boxes},
                });
                out.push_str(&serde_json::to_string(&line).expect("JSON serializes"));
                out.push('\n');
            }
            let file =
                format!("{}.ndjson", if opts.name.is_empty() { "dataset" } else { &opts.name });
            sink.text(&file, out.as_bytes())?;
        }
        ExportFormat::CocoSingle => {
            let mut out = CocoFile {
                categories: coco.categories.clone(),
                extra: coco.extra.clone(),
                ..Default::default()
            };
            for e in entries {
                let mut img = e.image.clone();
                img.file_name = image_path(opts.format, e);
                (img.width, img.height) = e.size;
                img.extra.insert(SPLIT_FIELD.into(), e.split.name().into());
                out.images.push(img);
                out.annotations
                    .extend(anns.get(&e.image.id).into_iter().flatten().map(|a| e.scaled(a)));
            }
            sink.text("annotations.json", &json_bytes(&out))?;
        }
        ExportFormat::CocoPerSplit => {
            for s in &splits {
                let mut out =
                    CocoFile { categories: coco.categories.clone(), ..Default::default() };
                for e in entries.iter().filter(|e| e.split == *s) {
                    out.images.push(CocoImage {
                        id: e.image.id,
                        file_name: e.name.clone(),
                        width: e.size.0,
                        height: e.size.1,
                        tags: Vec::new(),
                        extra: Map::new(),
                    });
                    out.annotations.extend(anns.get(&e.image.id).into_iter().flatten().map(|a| {
                        let mut a = e.scaled(a);
                        a.extra.clear();
                        a
                    }));
                }
                sink.text(&format!("{}/_annotations.coco.json", coco_dir(*s)), &json_bytes(&out))?;
            }
        }
    }
    Ok(())
}

/// Quotes YAML scalars that need it.
fn yaml_str(s: &str) -> String {
    let plain = !s.is_empty()
        && s.chars().all(|c| c.is_alphanumeric() || matches!(c, ' ' | '_' | '-' | '.'))
        && !s.starts_with(' ')
        && !s.ends_with(' ');
    if plain { s.to_string() } else { serde_json::to_string(s).expect("string serializes") }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::test_util::dataset;
    use std::io::Read;

    /// 4 images: 1 train, 2 val (same base name as 1), 3 test, 4 no split.
    fn fixture() -> (tempfile::TempDir, CocoFile) {
        let dir = tempfile::tempdir().unwrap();
        let mut coco = dataset(
            4,
            &[(1, BBox::new(10.0, 20.0, 20.0, 10.0)), (2, BBox::new(90.0, 90.0, 20.0, 20.0))],
        )
        .coco;
        coco.images[1].file_name = "images/x/0001.jpg".into();
        coco.images[1].extra.insert(SPLIT_FIELD.into(), "val".into());
        coco.images[2].extra.insert(SPLIT_FIELD.into(), "test".into());
        coco.images[3].file_name = "images/0004.jpg".into();
        for img in &coco.images {
            let p = dir.path().join(&img.file_name);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(p, format!("jpeg{}", img.id)).unwrap();
        }
        (dir, coco)
    }

    fn opts(format: ExportFormat) -> ExportOptions {
        ExportOptions {
            format,
            splits: Split::ALL.to_vec(),
            name: "Bibs".into(),
            resize: None,
            min_box_height: 0.0,
        }
    }

    fn read(dir: &Path, rel: &str) -> String {
        fs::read_to_string(dir.join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"))
    }

    #[test]
    fn yolo_folder_layout_labels_and_yaml() {
        let (d, coco) = fixture();
        let out = d.path().join("export");
        let r = export(
            d.path(),
            &coco,
            &opts(ExportFormat::Yolo),
            &Target::Folder(out.clone()),
            |_, _| true,
        )
        .unwrap();
        assert_eq!(r.per_split, [1, 1, 1]);
        assert_eq!(r.no_split, 1);
        assert_eq!(
            read(&out, "data.yaml").lines().filter(|l| !l.starts_with('#')).collect::<Vec<_>>(),
            ["train: images/train", "val: images/val", "test: images/test", "names:", "  0: bib"]
        );
        assert_eq!(read(&out, "labels/train/0001.txt"), "0 0.200000 0.250000 0.200000 0.100000\n");
        // Clipped to the 100x100 image: x 90..100, y 90..100.
        assert_eq!(read(&out, "labels/val/0001.txt"), "0 0.950000 0.950000 0.100000 0.100000\n");
        assert_eq!(read(&out, "labels/test/0003.txt"), "", "background image: empty label file");
        assert_eq!(read(&out, "images/val/0001.jpg"), "jpeg2");
        assert!(
            !d.path().read_dir().unwrap().any(|e| e
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains("partial"))
        );
        let err =
            export(d.path(), &coco, &opts(ExportFormat::Yolo), &Target::Folder(out), |_, _| true);
        assert!(err.is_err(), "never writes into an existing folder");
    }

    #[test]
    fn coco_per_split_and_single_file() {
        let (d, coco) = fixture();
        let out = d.path().join("coco");
        export(
            d.path(),
            &coco,
            &opts(ExportFormat::CocoPerSplit),
            &Target::Folder(out.clone()),
            |_, _| true,
        )
        .unwrap();
        let val: CocoFile =
            serde_json::from_str(&read(&out, "valid/_annotations.coco.json")).unwrap();
        assert_eq!(val.images.len(), 1);
        assert_eq!(val.images[0].file_name, "0001.jpg");
        assert_eq!(val.annotations.len(), 1);
        assert_eq!(read(&out, "valid/0001.jpg"), "jpeg2");

        let out = d.path().join("coco1");
        let o = ExportOptions {
            splits: vec![Split::Train, Split::Test],
            ..opts(ExportFormat::CocoSingle)
        };
        export(d.path(), &coco, &o, &Target::Folder(out.clone()), |_, _| true).unwrap();
        let all: CocoFile = serde_json::from_str(&read(&out, "annotations.json")).unwrap();
        let names: Vec<_> = all.images.iter().map(|i| i.file_name.as_str()).collect();
        assert_eq!(names, ["images/train/0001.jpg", "images/test/0003.jpg"]);
        assert_eq!(all.images[1].extra[SPLIT_FIELD], "test");
        assert_eq!(all.annotations.len(), 1);
    }

    #[test]
    fn ndjson_zip_matches_the_ultralytics_local_layout() {
        let (d, coco) = fixture();
        let out = d.path().join("bibs.zip");
        let r = export(
            d.path(),
            &coco,
            &opts(ExportFormat::Ndjson),
            &Target::Zip(out.clone()),
            |_, _| true,
        )
        .unwrap();
        assert_eq!(r.images(), 3);
        let mut zip = zip::ZipArchive::new(File::open(&out).unwrap()).unwrap();
        let mut s = String::new();
        zip.by_name("Bibs.ndjson").unwrap().read_to_string(&mut s).unwrap();
        let lines: Vec<Value> = s.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(lines[0]["type"], "dataset");
        assert_eq!(lines[0]["path"], ".");
        assert_eq!(lines[0]["class_names"]["0"], "bib");
        assert_eq!(lines[1]["file"], "0001.jpg");
        assert_eq!(lines[1]["split"], "train");
        assert_eq!(lines[1]["annotations"]["boxes"], json!([[0, 0.2, 0.25, 0.2, 0.1]]));
        let mut img = String::new();
        zip.by_name("images/test/0003.jpg").unwrap().read_to_string(&mut img).unwrap();
        assert_eq!(img, "jpeg3");
    }

    #[test]
    fn duplicate_names_missing_files_and_cancel() {
        let (d, mut coco) = fixture();
        coco.images[1].extra.insert(SPLIT_FIELD.into(), "train".into()); // same base name in train
        fs::remove_file(d.path().join(&coco.images[2].file_name)).unwrap();
        let out = d.path().join("y");
        let r = export(
            d.path(),
            &coco,
            &opts(ExportFormat::Yolo),
            &Target::Folder(out.clone()),
            |_, _| true,
        )
        .unwrap();
        assert_eq!(r.missing, vec!["images/train/0003.jpg"]);
        assert_eq!(read(&out, "images/train/2_0001.jpg"), "jpeg2");
        assert!(out.join("labels/train/2_0001.txt").exists());

        for target in [Target::Zip(d.path().join("c.zip")), Target::Folder(d.path().join("c"))] {
            let r =
                export(d.path(), &coco, &opts(ExportFormat::Yolo), &target, |_, _| false).unwrap();
            assert!(r.cancelled);
            let (Target::Zip(p) | Target::Folder(p)) = &target;
            assert!(!p.exists());
        }
        let leftovers: Vec<_> = d
            .path()
            .read_dir()
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .filter(|n| n.to_string_lossy().starts_with('.'))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    #[test]
    fn write_errors_are_reported_not_taken_for_a_cancel() {
        let (d, coco) = fixture();
        let blocker = d.path().join("file");
        fs::write(&blocker, b"x").unwrap();
        for target in [Target::Zip(blocker.join("out.zip")), Target::Folder(blocker.join("out"))] {
            let r = export(d.path(), &coco, &opts(ExportFormat::Yolo), &target, |_, _| true);
            assert!(r.is_err(), "{target:?}: {r:?}");
        }
    }

    #[test]
    fn resize_scales_coco_boxes_and_keeps_small_images() {
        let (d, mut coco) = fixture();
        coco.images[0].width = 200; // 200x100, box (10, 20, 20, 10)
        let o = ExportOptions {
            resize: Some(Resize {
                max_side: 100,
                resize: |jobs| jobs.iter().map(|(_, w, h)| Ok(format!("{w}x{h}").into())).collect(),
            }),
            ..opts(ExportFormat::CocoPerSplit)
        };
        let out = d.path().join("small");
        export(d.path(), &coco, &o, &Target::Folder(out.clone()), |_, _| true).unwrap();
        assert_eq!(read(&out, "train/0001.jpg"), "100x50");
        assert_eq!(read(&out, "valid/0001.jpg"), "jpeg2", "100x100: not resized");
        let train: CocoFile =
            serde_json::from_str(&read(&out, "train/_annotations.coco.json")).unwrap();
        assert_eq!((train.images[0].width, train.images[0].height), (100, 50));
        assert_eq!(train.annotations[0].bbox, BBox::new(5.0, 10.0, 10.0, 5.0));
    }

    #[test]
    fn short_boxes_are_left_out() {
        let (d, coco) = fixture(); // box heights: 10 (train), 20 (val)
        let o = ExportOptions { min_box_height: 15.0, ..opts(ExportFormat::Yolo) };
        let out = d.path().join("min");
        export(d.path(), &coco, &o, &Target::Folder(out.clone()), |_, _| true).unwrap();
        assert_eq!(read(&out, "labels/train/0001.txt"), "", "image kept, box dropped");
        assert_eq!(read(&out, "labels/val/0001.txt").lines().count(), 1);
    }

    #[test]
    fn yaml_quotes_odd_names() {
        assert_eq!(yaml_str("bib"), "bib");
        assert_eq!(yaml_str("race bib"), "race bib");
        assert_eq!(yaml_str("a: b"), "\"a: b\"");
    }
}
