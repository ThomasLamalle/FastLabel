//! Last-session state, restored on launch (`~/.config/fastlabel/session.json`).

use std::path::{Path, PathBuf};

use fastlabel_core::fsutil;
use serde::{Deserialize, Serialize};

use crate::crops::ViewOptions;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Session {
    pub last_dataset: Option<PathBuf>,
    /// Selected crop index in `last_dataset`.
    pub selected: usize,
    pub columns: usize,
    pub view: ViewOptions,
    pub model_path: Option<PathBuf>,
    pub confidence: f64,
    /// ONNX image embedder for cluster mode.
    pub embed_model_path: Option<PathBuf>,
    pub min_cluster_size: usize,
    pub export: ExportSettings,
    /// Split given to added images (`None`: no split).
    pub add_split: Option<fastlabel_core::Split>,
    pub auto_split: crate::cluster::AutoSplit,
    /// GPU memory budget for crop thumbnails.
    pub texture_budget_mb: usize,
    /// Boxes less tall than this (source pixels) are tagged and left out
    /// of exports; 0 keeps all.
    pub min_box_height: f64,
}

impl Default for Session {
    fn default() -> Self {
        Self {
            last_dataset: None,
            selected: 0,
            columns: 5,
            view: ViewOptions::default(),
            model_path: None,
            confidence: 0.1,
            embed_model_path: None,
            min_cluster_size: 20,
            export: ExportSettings::default(),
            add_split: None,
            auto_split: crate::cluster::AutoSplit::default(),
            texture_budget_mb: 256,
            min_box_height: 10.0,
        }
    }
}

/// Last choices of the export dialog.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExportSettings {
    pub format: fastlabel_core::export::ExportFormat,
    /// Train, val, test included.
    pub splits: [bool; 3],
    /// Zip file, else a folder (hardlinked images).
    pub zip: bool,
    /// Downscale images to a long side of `imgsz`.
    pub resize: bool,
    /// `None`: the report's suggested input size.
    pub imgsz: Option<u32>,
}

impl Default for ExportSettings {
    fn default() -> Self {
        Self {
            format: Default::default(),
            splits: [true; 3],
            zip: true,
            resize: false,
            imgsz: None,
        }
    }
}

impl Session {
    pub fn default_path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("fastlabel").join("session.json"))
    }

    /// Missing or unreadable files give the defaults.
    pub fn load(path: &Path) -> Self {
        fsutil::read_json(path).unwrap_or_default()
    }

    pub fn save(&self, path: &Path) -> fastlabel_core::Result<()> {
        fsutil::write_json_atomic(path, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_partial_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.json");
        assert_eq!(Session::load(&path), Session::default());
        let s = Session { columns: 8, last_dataset: Some("/x".into()), ..Session::default() };
        s.save(&path).unwrap();
        assert_eq!(Session::load(&path), s);
        std::fs::write(&path, r#"{"columns": 3}"#).unwrap();
        assert_eq!(Session::load(&path).columns, 3);
        assert_eq!(Session::load(&path).confidence, 0.1);
    }
}
