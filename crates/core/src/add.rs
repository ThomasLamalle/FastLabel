//! Adding images to a dataset: files are copied into `images/` under a
//! content-hash name (`images/<first 16 hex of SHA-256>.<ext>`), so names are
//! unique without a counter and adding the same image twice is detected.
//! The original path is kept as the image's `source_name`.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::error::Result;
use crate::fsutil;
use crate::yolo;

/// Image field with the path the image was added from.
pub const SOURCE_FIELD: &str = "source_name";
/// Image field with when the image was added (ISO 8601 UTC), used by the
/// image date sort: file dates do not survive copies.
pub const DATE_FIELD: &str = "date_added";
/// `.fastlabel/sources.json`: `source_name` and `date_added` of images added
/// but not saved yet (by file name), so a crash before the save keeps them.
pub const SOURCES_FILE: &str = "sources.json";

/// A file to add and the name it came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Source {
    pub path: PathBuf,
    /// File name for a single file, `<folder>/<relative path>` for a folder.
    pub source_name: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Copied {
    /// `images/<hash>.<ext>`, relative to the dataset folder.
    pub file_name: String,
    pub source_name: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AddReport {
    pub copied: Vec<Copied>,
    /// Sources whose content is already in the dataset (or given twice).
    pub duplicates: Vec<String>,
    /// Sources that could not be read or are not images, with the reason.
    pub failed: Vec<(String, String)>,
    pub cancelled: bool,
}

/// Images of the given files and folders (recursive, hidden folders
/// skipped), in a stable order.
pub fn collect(paths: &[PathBuf]) -> Vec<Source> {
    let mut out = Vec::new();
    for p in paths {
        if p.is_dir() {
            let folder =
                p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            if let Ok(files) = yolo::list_images(p, p) {
                out.extend(files.into_iter().map(|rel| Source {
                    path: p.join(&rel),
                    source_name: format!("{folder}/{}", rel.to_string_lossy().replace('\\', "/")),
                }));
            }
        } else if yolo::is_image(p) {
            let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            out.push(Source { path: p.clone(), source_name: name });
        }
    }
    out
}

/// `images/<first 16 hex of SHA-256>.<lowercase extension>`.
pub fn hashed_name(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    let hex: String = digest[..8].iter().map(|b| format!("{b:02x}")).collect();
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("jpg").to_ascii_lowercase();
    Ok(format!("images/{hex}.{ext}"))
}

/// Copies `sources` into `dir/images/` (temporary file + rename, so a crash
/// never leaves a partial image under its final name). `progress(done,
/// total)` returns false to cancel; files copied so far are kept and reported.
pub fn copy_sources(
    dir: &Path,
    sources: &[Source],
    mut progress: impl FnMut(usize, usize) -> bool,
) -> AddReport {
    let mut report = AddReport::default();
    for (i, src) in sources.iter().enumerate() {
        if !progress(i, sources.len()) {
            report.cancelled = true;
            break;
        }
        if let Err(e) = imagesize::size(&src.path) {
            report.failed.push((src.source_name.clone(), format!("not a readable image: {e}")));
            continue;
        }
        let name = match hashed_name(&src.path) {
            Ok(n) => n,
            Err(e) => {
                report.failed.push((src.source_name.clone(), e.to_string()));
                continue;
            }
        };
        let dst = dir.join(&name);
        if dst.exists() {
            report.duplicates.push(src.source_name.clone());
            continue;
        }
        let tmp = dst.with_file_name(format!(
            ".{}.tmp-{}",
            name.rsplit('/').next().unwrap_or(&name),
            std::process::id()
        ));
        let copied = dst
            .parent()
            .map_or(Ok(()), fs::create_dir_all)
            .and_then(|()| fs::copy(&src.path, &tmp))
            .and_then(|_| fs::rename(&tmp, &dst));
        match copied {
            Ok(()) => {
                report.copied.push(Copied { file_name: name, source_name: src.source_name.clone() })
            }
            Err(e) => {
                let _ = fs::remove_file(&tmp);
                report.failed.push((src.source_name.clone(), e.to_string()));
            }
        }
    }
    progress(sources.len(), sources.len());
    report
}

/// Fields of an added image kept until it is saved.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Pending {
    pub source_name: String,
    #[serde(default)]
    pub date_added: String,
}

/// Pending fields of added images (by file name).
pub fn read_sources(path: &Path) -> BTreeMap<String, Pending> {
    fsutil::read_json(path).unwrap_or_default()
}

pub fn write_sources(path: &Path, sources: &BTreeMap<String, Pending>) -> Result<()> {
    fsutil::write_json_atomic(path, sources)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_under_hash_names_and_skips_duplicates_and_non_images() {
        let tmp = tempfile::tempdir().unwrap();
        let (incoming, dataset) = (tmp.path().join("race 2026"), tmp.path().join("ds"));
        fs::create_dir_all(incoming.join("cam1")).unwrap();
        fs::create_dir_all(incoming.join(".hidden")).unwrap();
        fs::create_dir_all(&dataset).unwrap();
        let png = yolo::tests::png_header(64, 32);
        fs::write(incoming.join("cam1/a.PNG"), &png).unwrap();
        fs::write(incoming.join("b.png"), &png).unwrap(); // same content as a
        fs::write(incoming.join(".hidden/c.png"), &png).unwrap();
        fs::write(incoming.join("broken.jpg"), b"nope").unwrap();
        fs::write(incoming.join("notes.txt"), b"x").unwrap();
        let mut other = png.clone();
        other.push(0);
        let single = tmp.path().join("single.png");
        fs::write(&single, &other).unwrap();

        let sources = collect(&[incoming.clone(), single.clone(), tmp.path().join("notes.txt")]);
        let names: Vec<_> = sources.iter().map(|s| s.source_name.as_str()).collect();
        assert_eq!(
            names,
            ["race 2026/b.png", "race 2026/broken.jpg", "race 2026/cam1/a.PNG", "single.png"]
        );

        let r = copy_sources(&dataset, &sources, |_, _| true);
        assert_eq!(r.copied.len(), 2);
        assert_eq!(r.duplicates, ["race 2026/cam1/a.PNG"]);
        assert_eq!(r.failed.len(), 1);
        let name = &r.copied[0].file_name;
        assert!(
            name.starts_with("images/")
                && name.ends_with(".png")
                && name.len() == "images/".len() + 16 + 4
        );
        assert_eq!(fs::read(dataset.join(name)).unwrap(), png);
        assert_eq!(
            hashed_name(&incoming.join("cam1/a.PNG")).unwrap(),
            *name,
            "extension lowercased"
        );

        let again = copy_sources(&dataset, &sources, |_, _| true);
        assert!(again.copied.is_empty(), "everything already there");
        let leftovers = fs::read_dir(dataset.join("images"))
            .unwrap()
            .filter(|e| e.as_ref().unwrap().file_name().to_string_lossy().starts_with('.'));
        assert_eq!(leftovers.count(), 0);
        let cancelled = copy_sources(&tmp.path().join("ds2"), &sources, |_, _| false);
        assert!(cancelled.cancelled && cancelled.copied.is_empty());
    }
}
