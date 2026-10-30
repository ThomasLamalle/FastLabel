//! What a project annotates: `.fastlabel/project.json`. Folders without it
//! are box projects (COCO, YOLO or plain images), as before the file existed.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::fsutil;

pub const CONFIG_FILE: &str = "project.json";

/// Annotation kind; each kind has its own data files and views.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// Bounding boxes in COCO files (`annotations.vNNN.json`).
    #[default]
    Boxes,
    /// One text per image (crop) in `labels.csv`, see [`crate::text`].
    Text,
}

impl Kind {
    pub const ALL: [Kind; 2] = [Kind::Boxes, Kind::Text];

    pub fn label(self) -> &'static str {
        match self {
            Kind::Boxes => "Boxes (detection)",
            Kind::Text => "Text (OCR on crops)",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProjectConfig {
    pub kind: Kind,
    pub name: String,
    /// Box classes of a new project without annotations; afterwards the
    /// COCO categories are the reference.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub classes: Vec<String>,
    /// Text projects: the `labels.csv` column holding the text.
    pub text_column: String,
    /// Text projects: characters a text may hold; empty allows any. Typing
    /// ignores the others and an OCR reading keeps its longest run of them.
    pub charset: String,
}

impl Default for ProjectConfig {
    fn default() -> Self {
        Self {
            kind: Kind::Boxes,
            name: String::new(),
            classes: Vec::new(),
            text_column: "text".into(),
            charset: String::new(),
        }
    }
}

impl ProjectConfig {
    /// The config of `dir`; the default box project when the file is missing.
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join(crate::project::STATE_DIR).join(CONFIG_FILE);
        if !path.exists() {
            return Ok(Self::default());
        }
        let cfg: Self = fsutil::read_json(&path)?;
        if cfg.kind == Kind::Text && reserved(&cfg.text_column) {
            return Err(Error::invalid(&path, format!("bad text column `{}`", cfg.text_column)));
        }
        Ok(cfg)
    }

    pub fn save(&self, dir: &Path) -> Result<()> {
        fsutil::write_json_atomic(&dir.join(crate::project::STATE_DIR).join(CONFIG_FILE), self)
    }

    /// Keeps the characters of `text` allowed by [`Self::charset`].
    pub fn filter(&self, text: &str) -> String {
        text.chars().filter(|c| self.charset.is_empty() || self.charset.contains(*c)).collect()
    }
}

/// Column names a text column cannot take (empty or used by `labels.csv`).
pub fn reserved(column: &str) -> bool {
    let c = column.trim();
    c.is_empty() || ["file", "split", "status"].contains(&c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_file_is_a_box_project_and_round_trips() {
        let d = tempfile::tempdir().unwrap();
        assert_eq!(ProjectConfig::load(d.path()).unwrap().kind, Kind::Boxes);
        let cfg = ProjectConfig {
            kind: Kind::Text,
            name: "Bibs".into(),
            text_column: "bib".into(),
            charset: "0123456789".into(),
            ..Default::default()
        };
        cfg.save(d.path()).unwrap();
        assert_eq!(ProjectConfig::load(d.path()).unwrap(), cfg);
        assert_eq!(cfg.filter("a12-3"), "123");
    }
}
