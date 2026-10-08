//! Persistent thumbnail cache: small JPEG files keyed by a stable hash of
//! (source path, size, mtime, region, thumbnail size).

use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use image::RgbImage;

use crate::crop::Region;
use crate::decode::{decode_bytes, encode_jpeg};

/// Bump when the thumbnail format or resampling changes.
const CACHE_VERSION: u64 = 1;
const QUALITY: u8 = 90;

/// FNV-1a: stable across runs and Rust versions (unlike `DefaultHasher`).
#[derive(Clone, Copy)]
pub struct Fnv(u64);

impl Default for Fnv {
    fn default() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
}

impl Fnv {
    pub fn bytes(mut self, bytes: &[u8]) -> Self {
        for b in bytes {
            self.0 ^= u64::from(*b);
            self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
        }
        self
    }

    pub fn u64(self, v: u64) -> Self {
        self.bytes(&v.to_le_bytes())
    }

    pub fn finish(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Debug)]
pub struct DiskCache {
    dir: PathBuf,
}

impl DiskCache {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// Cache key; `None` when the source file cannot be stat'ed.
    pub fn key(source: &Path, region: Region, max_side: u32) -> Option<u64> {
        let meta = fs::metadata(source).ok()?;
        let mtime = meta.modified().ok()?.duration_since(UNIX_EPOCH).ok()?.as_nanos() as u64;
        Some(
            Fnv::default()
                .u64(CACHE_VERSION)
                .bytes(source.as_os_str().as_encoded_bytes())
                .u64(meta.len())
                .u64(mtime)
                .u64(u64::from(region.x))
                .u64(u64::from(region.y))
                .u64(u64::from(region.w))
                .u64(u64::from(region.h))
                .u64(u64::from(max_side))
                .finish(),
        )
    }

    fn path(&self, key: u64) -> PathBuf {
        let hex = format!("{key:016x}");
        self.dir.join(&hex[..2]).join(format!("{hex}.jpg"))
    }

    pub fn get(&self, key: u64) -> Option<RgbImage> {
        let bytes = fs::read(self.path(key)).ok()?;
        decode_bytes(&bytes).ok()
    }

    /// Best effort: a failed write only costs a re-decode later.
    pub fn put(&self, key: u64, img: &RgbImage) {
        let path = self.path(key);
        let Ok(bytes) = encode_jpeg(img, QUALITY) else {
            return;
        };
        let Some(dir) = path.parent() else { return };
        if fs::create_dir_all(dir).is_err() {
            return;
        }
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        if fs::write(&tmp, bytes).is_ok() && fs::rename(&tmp, &path).is_err() {
            let _ = fs::remove_file(&tmp);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::tests::gradient;

    #[test]
    fn put_get_and_key_changes_with_inputs() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src.png");
        gradient(20, 20).save(&src).unwrap();
        let region = Region { x: 0, y: 0, w: 10, h: 10 };
        let k = DiskCache::key(&src, region, 256).unwrap();
        assert_ne!(k, DiskCache::key(&src, Region { x: 1, ..region }, 256).unwrap());
        assert_ne!(k, DiskCache::key(&src, region, 128).unwrap());
        assert!(DiskCache::key(&dir.path().join("nope.jpg"), region, 256).is_none());

        let cache = DiskCache::new(dir.path().join("cache"));
        assert!(cache.get(k).is_none());
        cache.put(k, &gradient(10, 10));
        assert_eq!(cache.get(k).unwrap().dimensions(), (10, 10));
    }
}
