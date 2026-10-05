//! Import of YOLO / Ultralytics detection datasets (`data.yaml`, `images/`,
//! `labels/`) and of plain image folders without labels.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Map;
use yaml_rust2::{Yaml, YamlLoader};

use crate::coco::{CocoAnnotation, CocoCategory, CocoFile, CocoImage};
use crate::error::{Error, Result};
use crate::geom::BBox;

pub const IMAGE_EXTENSIONS: [&str; 5] = ["jpg", "jpeg", "png", "bmp", "webp"];

pub fn is_image(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| IMAGE_EXTENSIONS.iter().any(|x| x.eq_ignore_ascii_case(e)))
}

/// Class names from `data.yaml` (`names:` as a list or an `index: name` map).
pub fn read_class_names(yaml_path: &Path) -> Result<Vec<String>> {
    let text = fs::read_to_string(yaml_path).map_err(|e| Error::io(yaml_path, e))?;
    let docs =
        YamlLoader::load_from_str(&text).map_err(|e| Error::invalid(yaml_path, e.to_string()))?;
    let Some(doc) = docs.first() else {
        return Ok(Vec::new());
    };
    let as_name = |y: &Yaml| match y {
        Yaml::String(s) => Some(s.clone()),
        Yaml::Integer(i) => Some(i.to_string()),
        _ => None,
    };
    match &doc["names"] {
        Yaml::Array(items) => Ok(items.iter().filter_map(as_name).collect()),
        Yaml::Hash(map) => {
            let mut pairs = Vec::new();
            for (k, v) in map {
                let (Yaml::Integer(i), Some(name)) = (k, as_name(v)) else {
                    return Err(Error::invalid(
                        yaml_path,
                        "names keys must be class indices",
                    ));
                };
                pairs.push((*i, name));
            }
            pairs.sort();
            if pairs.iter().enumerate().any(|(n, (i, _))| *i != n as i64) {
                return Err(Error::invalid(yaml_path, "class indices must be 0..n"));
            }
            Ok(pairs.into_iter().map(|(_, n)| n).collect())
        }
        _ => Ok(Vec::new()),
    }
}

/// All images under `dir`, recursively, sorted, as paths relative to `base`.
pub fn list_images(dir: &Path, base: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in fs::read_dir(&d).map_err(|e| Error::io(&d, e))? {
            let entry = entry.map_err(|e| Error::io(&d, e))?;
            let path = entry.path();
            let file_type = entry.file_type().map_err(|e| Error::io(&path, e))?;
            if file_type.is_dir() {
                if !entry.file_name().to_string_lossy().starts_with('.') {
                    stack.push(path);
                }
            } else if is_image(&path) {
                out.push(path.strip_prefix(base).unwrap_or(&path).to_path_buf());
            }
        }
    }
    out.sort();
    Ok(out)
}

/// Builds a COCO file from `dir/images/**` and the matching
/// `dir/labels/**/*.txt`. Missing or empty label files mean no boxes; boxes
/// are clipped to the image. Without `data.yaml` a single `object` class is used.
pub fn import(dir: &Path) -> Result<CocoFile> {
    let yaml = dir.join("data.yaml");
    let mut names = if yaml.exists() {
        read_class_names(&yaml)?
    } else {
        Vec::new()
    };
    if names.is_empty() {
        names.push("object".into());
    }
    let categories = names
        .iter()
        .enumerate()
        .map(|(i, n)| CocoCategory {
            id: i as i64 + 1,
            name: n.clone(),
            supercategory: String::new(),
            extra: Map::new(),
        })
        .collect();

    let images_dir = dir.join("images");
    let image_root = if images_dir.is_dir() {
        images_dir
    } else {
        dir.to_path_buf()
    };
    let mut coco = CocoFile {
        categories,
        ..CocoFile::default()
    };
    for (i, rel) in list_images(&image_root, dir)?.into_iter().enumerate() {
        let path = dir.join(&rel);
        let size = imagesize::size(&path)
            .map_err(|e| Error::invalid(&path, format!("unreadable image size: {e}")))?;
        let (w, h) = (size.width as u32, size.height as u32);
        let id = i as i64 + 1;
        let file_name = rel.to_string_lossy().replace('\\', "/");
        let label_path = label_path_for(dir, &rel);
        if let Ok(text) = fs::read_to_string(&label_path) {
            for (line_no, line) in text.lines().enumerate() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let bad = || {
                    Error::invalid(
                        &label_path,
                        format!(
                            "line {}: invalid YOLO detection label `{line}`",
                            line_no + 1
                        ),
                    )
                };
                let v: Vec<f64> = line
                    .split_whitespace()
                    .map(str::parse)
                    .collect::<std::result::Result<_, _>>()
                    .map_err(|_| bad())?;
                let [cls, cx, cy, bw, bh] = v[..] else {
                    return Err(bad());
                };
                if cls < 0.0 || cls.fract() != 0.0 || cls as usize >= names.len() {
                    return Err(Error::invalid(
                        &label_path,
                        format!("line {}: unknown class {cls}", line_no + 1),
                    ));
                }
                let (fw, fh) = (f64::from(w), f64::from(h));
                let b = BBox::new((cx - bw / 2.0) * fw, (cy - bh / 2.0) * fh, bw * fw, bh * fh);
                if let Some(b) = b.clip(fw, fh) {
                    let ann_id = coco.annotations.len() as i64 + 1;
                    coco.annotations
                        .push(CocoAnnotation::new(ann_id, id, cls as i64 + 1, b));
                }
            }
        }
        coco.images.push(CocoImage {
            id,
            file_name,
            width: w,
            height: h,
            tags: Vec::new(),
            extra: Map::new(),
        });
    }
    Ok(coco)
}

/// `images/train/a.jpg` → `labels/train/a.txt`.
fn label_path_for(dir: &Path, rel_image: &Path) -> PathBuf {
    let mut comps = rel_image.components();
    let rest: PathBuf = match comps.next() {
        Some(first) if first.as_os_str() == "images" => comps.collect(),
        _ => rel_image.to_path_buf(),
    };
    dir.join("labels").join(rest).with_extension("txt")
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Minimal valid PNG of the given size (header is enough for `imagesize`).
    pub fn png_header(w: u32, h: u32) -> Vec<u8> {
        let mut v = vec![
            0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 13, b'I', b'H', b'D', b'R',
        ];
        v.extend_from_slice(&w.to_be_bytes());
        v.extend_from_slice(&h.to_be_bytes());
        v.extend_from_slice(&[8, 2, 0, 0, 0, 0, 0, 0, 0]);
        v
    }

    #[test]
    fn imports_yolo_layout() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        fs::write(
            d.join("data.yaml"),
            "path: x\nnames:\n  0: bib\n  1: face\n",
        )
        .unwrap();
        for split in ["train", "val"] {
            fs::create_dir_all(d.join("images").join(split)).unwrap();
            fs::create_dir_all(d.join("labels").join(split)).unwrap();
        }
        fs::write(d.join("images/train/a.png"), png_header(200, 100)).unwrap();
        fs::write(d.join("images/val/b.png"), png_header(200, 100)).unwrap();
        fs::write(d.join("images/val/c.png"), png_header(200, 100)).unwrap();
        fs::write(
            d.join("labels/train/a.txt"),
            "0 0.5 0.5 0.1 0.2\n1 0.99 0.5 0.1 0.2\n\n",
        )
        .unwrap();
        fs::write(d.join("labels/val/b.txt"), "").unwrap();

        let coco = import(d).unwrap();
        assert_eq!(
            coco.categories
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>(),
            vec!["bib", "face"]
        );
        assert_eq!(
            coco.images
                .iter()
                .map(|i| i.file_name.as_str())
                .collect::<Vec<_>>(),
            vec!["images/train/a.png", "images/val/b.png", "images/val/c.png"]
        );
        assert_eq!(coco.annotations.len(), 2);
        assert_eq!(coco.annotations[0].bbox, BBox::new(90.0, 40.0, 20.0, 20.0));
        let clipped = coco.annotations[1].bbox;
        assert!((clipped.x - 188.0).abs() < 1e-9 && (clipped.x2() - 200.0).abs() < 1e-9);
        assert_eq!(coco.annotations[1].category_id, 2);
    }

    #[test]
    fn rejects_malformed_labels_and_reads_list_names() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        fs::write(d.join("data.yaml"), "names: [bib]\n").unwrap();
        assert_eq!(read_class_names(&d.join("data.yaml")).unwrap(), vec!["bib"]);
        fs::create_dir_all(d.join("images")).unwrap();
        fs::create_dir_all(d.join("labels")).unwrap();
        fs::write(d.join("images/a.png"), png_header(10, 10)).unwrap();
        fs::write(d.join("labels/a.txt"), "0 0.5 0.5\n").unwrap();
        assert!(import(d).is_err());
        fs::write(d.join("labels/a.txt"), "3 0.5 0.5 0.1 0.1\n").unwrap();
        assert!(import(d).is_err());
    }

    #[test]
    fn plain_image_folder_gets_default_class() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("x.png"), png_header(10, 10)).unwrap();
        let coco = import(dir.path()).unwrap();
        assert_eq!(coco.images.len(), 1);
        assert_eq!(coco.categories[0].name, "object");
    }
}
