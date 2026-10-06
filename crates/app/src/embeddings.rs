//! Per-frame embedding vectors (`.fastlabel/embeddings.bin`) and the last
//! computed cluster layout (`.fastlabel/clusters.json`).

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use fastlabel_core::{ImageId, fsutil};
use fastlabel_media::cache::Fnv;
use serde::{Deserialize, Serialize};

pub const EMBEDDINGS_FILE: &str = "embeddings.bin";
pub const LAYOUT_FILE: &str = "clusters.json";
const MAGIC: &[u8; 8] = b"FLEMB001";

/// Stable key of a frame: its id and file name without folders, so moving
/// it to another split keeps its vector.
pub fn key(image_id: ImageId, file_name: &str) -> u64 {
    let base = file_name.rsplit('/').next().unwrap_or(file_name);
    Fnv::default().u64(image_id as u64).bytes(base.as_bytes()).finish()
}

/// Identifies the model that produced the vectors (file name and size).
pub fn model_id(path: &Path) -> String {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    format!("{name}:{size}")
}

#[derive(Default)]
pub struct Embeddings {
    pub model: String,
    dim: usize,
    by_key: HashMap<u64, Vec<f32>>,
}

impl Embeddings {
    /// Missing or unreadable files give an empty store.
    pub fn load(path: &Path) -> Self {
        std::fs::read(path).ok().and_then(|b| Self::decode(&b)).unwrap_or_default()
    }

    fn decode(bytes: &[u8]) -> Option<Self> {
        let mut r = bytes.strip_prefix(MAGIC)?;
        let mut take = |n: usize| -> Option<&[u8]> {
            let (head, tail) = r.split_at_checked(n)?;
            r = tail;
            Some(head)
        };
        let model_len = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
        let model = String::from_utf8(take(model_len)?.to_vec()).ok()?;
        let dim = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
        let count = u64::from_le_bytes(take(8)?.try_into().ok()?) as usize;
        let mut by_key = HashMap::with_capacity(count.min(1 << 20));
        for _ in 0..count {
            let k = u64::from_le_bytes(take(8)?.try_into().ok()?);
            let v =
                take(dim * 4)?.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
            by_key.insert(k, v);
        }
        Some(Self { model, dim, by_key })
    }

    pub fn save(&self, path: &Path) -> fastlabel_core::Result<()> {
        fsutil::write_atomic(path, |w| {
            let io = |e| fastlabel_core::Error::io(path, e);
            w.write_all(MAGIC).map_err(io)?;
            w.write_all(&(self.model.len() as u32).to_le_bytes()).map_err(io)?;
            w.write_all(self.model.as_bytes()).map_err(io)?;
            w.write_all(&(self.dim as u32).to_le_bytes()).map_err(io)?;
            w.write_all(&(self.by_key.len() as u64).to_le_bytes()).map_err(io)?;
            for (k, v) in &self.by_key {
                w.write_all(&k.to_le_bytes()).map_err(io)?;
                for x in v {
                    w.write_all(&x.to_le_bytes()).map_err(io)?;
                }
            }
            Ok(())
        })
    }

    /// Vectors of another model are dropped.
    pub fn use_model(&mut self, model: &str) {
        if self.model != model {
            *self = Self { model: model.to_string(), ..Self::default() };
        }
    }

    pub fn get(&self, k: u64) -> Option<&[f32]> {
        self.by_key.get(&k).map(Vec::as_slice)
    }

    /// Adds vectors; ones of another size than the stored ones are ignored.
    pub fn extend(&mut self, values: impl IntoIterator<Item = (u64, Vec<f32>)>) {
        for (k, v) in values {
            if self.by_key.is_empty() {
                self.dim = v.len();
            }
            if v.len() == self.dim && !v.is_empty() {
                self.by_key.insert(k, v);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.by_key.len()
    }
}

/// A computed layout, valid for the frames (`keys`) it was computed on.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SavedLayout {
    pub model: String,
    pub min_cluster_size: usize,
    pub keys: Vec<u64>,
    pub points: Vec<[f32; 2]>,
    pub labels: Vec<i32>,
}

impl SavedLayout {
    pub fn load(path: &Path) -> Option<Self> {
        let l: Self = fsutil::read_json(path).ok()?;
        let n = l.keys.len();
        (l.points.len() == n && l.labels.len() == n).then_some(l)
    }

    pub fn save(&self, path: &Path) -> fastlabel_core::Result<()> {
        fsutil::write_json_atomic(path, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_ignore_the_split_folder() {
        assert_eq!(key(3, "images/train/a.jpg"), key(3, "images/val/a.jpg"));
        assert_ne!(key(3, "images/train/a.jpg"), key(4, "images/train/a.jpg"));
        assert_ne!(key(3, "images/train/a.jpg"), key(3, "images/train/b.jpg"));
    }

    #[test]
    fn round_trip_model_change_and_bad_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(EMBEDDINGS_FILE);
        let mut e = Embeddings::default();
        e.use_model("m:1");
        e.extend([(1, vec![0.5, -1.0]), (2, vec![1.0, 2.0]), (3, vec![1.0])]);
        assert_eq!(e.len(), 2, "wrong size ignored");
        e.save(&path).unwrap();

        let mut back = Embeddings::load(&path);
        assert_eq!(back.model, "m:1");
        assert_eq!(back.get(1), Some(&[0.5, -1.0][..]));
        assert_eq!(back.len(), 2);
        back.use_model("m:1");
        assert_eq!(back.len(), 2);
        back.use_model("other:5");
        assert_eq!(back.len(), 0);

        std::fs::write(&path, b"FLEMB001\x05\0\0\0ab").unwrap();
        assert_eq!(Embeddings::load(&path).len(), 0, "truncated file");
        assert_eq!(Embeddings::load(&dir.path().join("missing")).len(), 0);

        let layout = SavedLayout {
            keys: vec![1, 2],
            points: vec![[0.0, 1.0], [2.0, 3.0]],
            labels: vec![0, -1],
            ..SavedLayout::default()
        };
        let lp = dir.path().join(LAYOUT_FILE);
        layout.save(&lp).unwrap();
        assert_eq!(SavedLayout::load(&lp), Some(layout));
    }
}
