//! Zip export (annotation file + images tree), ready for Ultralytics import.

use std::fs::{self, File};
use std::io;
use std::path::Path;

use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipWriter};

use crate::coco::CocoFile;
use crate::error::{Error, Result};
use crate::fsutil;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExportReport {
    pub images: usize,
    pub missing: Vec<String>,
    pub cancelled: bool,
}

/// Writes `out` with `annotation_name` at the root and every image of `coco`
/// at its `file_name`, read from `root`. Images are stored uncompressed (JPEGs
/// don't shrink). `progress(done, total)` returns false to cancel; a cancelled
/// export leaves no zip behind.
pub fn export_zip(
    root: &Path,
    coco: &CocoFile,
    annotation_name: &str,
    out: &Path,
    mut progress: impl FnMut(usize, usize) -> bool,
) -> Result<ExportReport> {
    let mut report = ExportReport::default();
    let total = coco.images.len();
    let zip_err = |e| Error::Zip {
        path: out.to_path_buf(),
        source: e,
    };
    let result = fsutil::write_atomic(out, |w| {
        let mut zip = ZipWriter::new(&mut *w);
        let deflate = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
        let stored = SimpleFileOptions::default()
            .compression_method(CompressionMethod::Stored)
            .large_file(false);
        zip.start_file(annotation_name, deflate).map_err(zip_err)?;
        let fmt = serde_json::ser::PrettyFormatter::with_indent(b"    ");
        serde::Serialize::serialize(
            coco,
            &mut serde_json::Serializer::with_formatter(&mut zip, fmt),
        )
        .map_err(|e| Error::json(out, e))?;
        for (i, img) in coco.images.iter().enumerate() {
            if !progress(i, total) {
                report.cancelled = true;
                return Err(Error::invalid(out, "export cancelled"));
            }
            let src = root.join(&img.file_name);
            let mut file = match File::open(&src) {
                Ok(f) => f,
                Err(e) if e.kind() == io::ErrorKind::NotFound => {
                    report.missing.push(img.file_name.clone());
                    continue;
                }
                Err(e) => return Err(Error::io(&src, e)),
            };
            zip.start_file(&img.file_name, stored).map_err(zip_err)?;
            io::copy(&mut file, &mut zip).map_err(|e| Error::io(&src, e))?;
            report.images += 1;
        }
        progress(total, total);
        zip.finish().map_err(zip_err)?;
        Ok(())
    });
    match result {
        Ok(()) => Ok(report),
        Err(_) if report.cancelled => {
            let _ = fs::remove_file(out);
            Ok(report)
        }
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::test_util::dataset;
    use std::io::Read;

    #[test]
    fn exports_annotations_and_images() {
        let dir = tempfile::tempdir().unwrap();
        let coco = dataset(2, &[]).coco;
        fs::create_dir_all(dir.path().join("images/train")).unwrap();
        fs::write(dir.path().join("images/train/0001.jpg"), b"jpegdata").unwrap();
        let out = dir.path().join("out.zip");
        let report = export_zip(dir.path(), &coco, "annotations.v004.json", &out, |_, _| {
            true
        })
        .unwrap();
        assert_eq!(report.images, 1);
        assert_eq!(report.missing, vec!["images/train/0002.jpg"]);

        let mut zip = zip::ZipArchive::new(File::open(&out).unwrap()).unwrap();
        let mut s = String::new();
        zip.by_name("annotations.v004.json")
            .unwrap()
            .read_to_string(&mut s)
            .unwrap();
        assert_eq!(serde_json::from_str::<CocoFile>(&s).unwrap(), coco);
        let mut img = Vec::new();
        zip.by_name("images/train/0001.jpg")
            .unwrap()
            .read_to_end(&mut img)
            .unwrap();
        assert_eq!(img, b"jpegdata");
    }

    #[test]
    fn cancel_leaves_no_file() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.zip");
        let report = export_zip(dir.path(), &dataset(2, &[]).coco, "a.json", &out, |_, _| {
            false
        })
        .unwrap();
        assert!(report.cancelled);
        assert!(!out.exists());
    }
}
