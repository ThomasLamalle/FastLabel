//! Text recognition (OCR) on crops with a PaddleOCR CTC recognizer, such as
//! PP-OCRv6 small as shipped by RapidOCR (the bib reading pipeline's model):
//! input `x` `[N, 3, 48, W]`, output `[N, columns, classes]` of
//! probabilities, alphabet in the `character` metadata (one per line; index 0
//! is "no character", the last one a space).

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use fastlabel_media::decode::decode_rgb;
use image::RgbImage;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

use crate::batch::{BatchItem, Progress};
use crate::detector::{InferError, Result};

/// Input height when the model does not fix it: PP-OCR reads 48 px lines.
const HEIGHT: u32 = 48;
/// PP-OCR was trained on lines padded to at least 320 px; narrower padding
/// changed 8 of 40 bib readings in the pipeline.
const MIN_WIDTH: u32 = 320;
// ponytail: fixed stretch tuned on bibs (the pipeline's OCR_STRETCH); make it
// a setting if another project reads long text lines worse with it.
/// Crops are widened by this factor before reading: a bib's digits are short
/// next to the crop, and two equal digits side by side merged into one. On 40
/// labelled bib crops, 1.5 read 27 of 28 readable bibs against 24 unstretched.
pub const STRETCH: f32 = 1.5;

pub struct TextReader {
    session: Session,
    pub path: PathBuf,
    alphabet: Vec<char>,
    height: u32,
    /// Fixed input width, if the model has one.
    width: Option<u32>,
}

/// A reading: the text kept and the mean probability of its characters.
#[derive(Clone, Debug, PartialEq)]
pub struct Reading {
    pub text: String,
    pub score: f32,
}

#[derive(Debug, Default)]
pub struct OcrResult {
    pub readings: Vec<(usize, Reading)>,
    pub failed: Vec<(usize, String)>,
    pub cancelled: bool,
}

impl TextReader {
    /// `threads = 0` lets ONNX Runtime decide.
    pub fn load(path: &Path, threads: usize) -> Result<Self> {
        let mut builder =
            Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3)?;
        if threads > 0 {
            builder = builder.with_intra_threads(threads)?;
        }
        let session = builder.commit_from_file(path)?;
        let characters = session.metadata()?.custom("character").ok_or_else(|| {
            InferError::Unsupported(
                "no `character` metadata: a PaddleOCR text recognizer (CTC) is expected".into(),
            )
        })?;
        let alphabet =
            [' '].into_iter().chain(characters.lines().flat_map(str::chars)).chain([' ']);
        let input =
            session.inputs().first().ok_or_else(|| InferError::Unsupported("no input".into()))?;
        let shape = input.dtype().tensor_shape().map(|s| s.to_vec()).unwrap_or_default();
        let [_, 3, h, w] = shape[..] else {
            return Err(InferError::Unsupported(format!("input shape {shape:?}")));
        };
        Ok(Self {
            session,
            path: path.to_path_buf(),
            alphabet: alphabet.collect(),
            height: if h > 0 { h as u32 } else { HEIGHT },
            width: (w > 0).then_some(w as u32),
        })
    }

    /// Model input for a crop: stretched to the input height, blue, green,
    /// red planes in [-1, 1], zero padding on the right. Returns (data, width).
    fn preprocess(&self, img: &RgbImage) -> (Vec<f32>, usize) {
        let h = self.height;
        let w = (h as f32 * img.width() as f32 / img.height().max(1) as f32 * STRETCH).ceil();
        let mut w = (w as u32).max(1);
        let padded = match self.width {
            Some(fixed) => {
                w = w.min(fixed);
                fixed
            }
            None => w.max(MIN_WIDTH),
        };
        let resized = crate::prepost::resize(img, w, h);
        let (h, padded) = (h as usize, padded as usize);
        let mut data = vec![0.0f32; 3 * h * padded];
        for (x, y, p) in resized.enumerate_pixels() {
            for c in 0..3 {
                let v = f32::from(p[2 - c]) / 127.5 - 1.0;
                data[(c * h + y as usize) * padded + x as usize] = v;
            }
        }
        (data, padded)
    }

    /// Each character read with its probability: the most likely class per
    /// column, repeats merged, "no character" dropped (CTC greedy decoding).
    pub fn read(&mut self, img: &RgbImage) -> Result<Vec<(char, f32)>> {
        let (data, width) = self.preprocess(img);
        let input = Tensor::from_array(([1, 3, self.height as usize, width], data))?;
        let outputs = self.session.run(ort::inputs![input])?;
        let (shape, probs) = outputs[0].try_extract_tensor::<f32>()?;
        let classes = shape.get(2).copied().unwrap_or(0) as usize;
        if classes != self.alphabet.len() {
            return Err(InferError::Unsupported(format!(
                "{classes} output classes for an alphabet of {}",
                self.alphabet.len()
            )));
        }
        Ok(decode_ctc(probs, &self.alphabet))
    }
}

fn decode_ctc(probs: &[f32], alphabet: &[char]) -> Vec<(char, f32)> {
    let mut text = Vec::new();
    let mut previous = 0;
    for column in probs.chunks(alphabet.len()) {
        let (index, p) = column
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or((0, 0.0), |(i, p)| (i, *p));
        if index != 0 && index != previous {
            text.push((alphabet[index], p));
        }
        previous = index;
    }
    text
}

/// The reading to keep: with a charset, its longest run of allowed
/// characters (the first of equal runs), so words printed next to a bib
/// number are dropped; else the whole text, trimmed.
pub fn keep(text: &[(char, f32)], charset: &str) -> Reading {
    let run: &[(char, f32)] = if charset.is_empty() {
        let start = text.iter().position(|c| !c.0.is_whitespace()).unwrap_or(text.len());
        let end = text.iter().rposition(|c| !c.0.is_whitespace()).map_or(start, |e| e + 1);
        &text[start..end]
    } else {
        let allowed = |c: &(char, f32)| charset.contains(c.0);
        text.chunk_by(|a, b| allowed(a) == allowed(b))
            .filter(|run| allowed(&run[0]))
            .rev()
            .max_by_key(|run| run.len())
            .unwrap_or(&[])
    };
    let score =
        if run.is_empty() { 0.0 } else { run.iter().map(|c| c.1).sum::<f32>() / run.len() as f32 };
    Reading { text: run.iter().map(|c| c.0).collect(), score }
}

/// Reads every item in turn (crops are small: decoding is cheap next to the
/// model). Progress and cancellation through `progress`.
pub fn run_ocr(
    reader: &mut TextReader,
    items: Vec<BatchItem>,
    charset: &str,
    progress: &Progress,
) -> OcrResult {
    progress.total.store(items.len(), Ordering::Relaxed);
    progress.done.store(0, Ordering::Relaxed);
    let mut result = OcrResult::default();
    for (n, item) in items.into_iter().enumerate() {
        if progress.cancel.load(Ordering::Relaxed) {
            result.cancelled = true;
            break;
        }
        let read = decode_rgb(&item.path)
            .map_err(|e| e.to_string())
            .and_then(|img| reader.read(&img).map_err(|e| e.to_string()));
        match read {
            Ok(text) => result.readings.push((item.key, keep(&text, charset))),
            Err(e) => result.failed.push((item.key, e)),
        }
        progress.done.store(n + 1, Ordering::Relaxed);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chars(s: &str) -> Vec<(char, f32)> {
        s.chars().map(|c| (c, 0.5)).collect()
    }

    #[test]
    fn keep_takes_the_first_longest_allowed_run() {
        let digits = "0123456789";
        assert_eq!(keep(&chars("A12-3456 7890"), digits).text, "3456");
        assert_eq!(keep(&chars("ABC"), digits), Reading { text: String::new(), score: 0.0 });
        assert_eq!(keep(&chars("  hello world "), "").text, "hello world");
        assert_eq!(keep(&chars("12"), digits).score, 0.5);
    }

    #[test]
    fn ctc_merges_repeats_and_drops_blanks() {
        let alphabet = [' ', '1', '2', ' '];
        // Columns: 1, 1, blank, 1, 2 → "112".
        let col = |i: usize| (0..4).map(|k| if k == i { 0.9 } else { 0.0 }).collect::<Vec<_>>();
        let probs: Vec<f32> = [1, 1, 0, 1, 2].iter().flat_map(|&i| col(i)).collect();
        let text: String = decode_ctc(&probs, &alphabet).iter().map(|c| c.0).collect();
        assert_eq!(text, "112");
    }

    /// `FASTLABEL_OCR_MODEL=<rec.onnx> cargo test -p fastlabel-infer reads_digits -- --ignored`
    #[test]
    #[ignore]
    fn reads_digits() {
        let path = std::env::var("FASTLABEL_OCR_MODEL").expect("FASTLABEL_OCR_MODEL");
        let mut r = TextReader::load(Path::new(&path), 0).unwrap();
        // Black "7" strokes on white: the model must return a short text.
        let img = RgbImage::from_fn(60, 40, |x, y| {
            let ink =
                (y < 8 && x > 15 && x < 45) || (x > 25 + (40 - y) / 3 && x < 32 + (40 - y) / 3);
            image::Rgb(if ink { [0; 3] } else { [255; 3] })
        });
        let text = r.read(&img).unwrap();
        println!("{text:?}");
        assert_eq!(keep(&text, "0123456789").text, "7");
    }
}
