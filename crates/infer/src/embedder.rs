//! Image embeddings from an ONNX model (e.g. DINOv2 exported by
//! `scripts/export_dinov2.py`): one vector per frame, for clustering.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use fastlabel_media::decode::decode_rgb;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

use crate::batch::{BatchItem, Progress};
use crate::detector::{InferError, Result};
use crate::prepost;

/// Input side when the model does not fix it.
const DEFAULT_SIZE: u32 = 224;
/// Frames per model call.
const BATCH: usize = 8;
const DECODERS: usize = 2;

pub struct Embedder {
    session: Session,
    pub path: PathBuf,
    pub input_w: u32,
    pub input_h: u32,
    /// The model takes a batch dimension other than 1.
    batched: bool,
}

#[derive(Debug, Default)]
pub struct EmbedResult {
    /// (key, L2-normalized vector).
    pub vectors: Vec<(usize, Vec<f32>)>,
    pub failed: Vec<(usize, String)>,
    pub cancelled: bool,
}

impl Embedder {
    /// Loads a model with one `[N, 3, H, W]` input (ImageNet-normalized RGB)
    /// and a `[N, D]` output; `threads = 0` lets ONNX Runtime decide.
    pub fn load(path: &Path, threads: usize) -> Result<Self> {
        let mut builder =
            Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3)?;
        if threads > 0 {
            builder = builder.with_intra_threads(threads)?;
        }
        let session = builder.commit_from_file(path)?;
        let input =
            session.inputs().first().ok_or_else(|| InferError::Unsupported("no input".into()))?;
        let shape = input.dtype().tensor_shape().map(|s| s.to_vec()).unwrap_or_default();
        let (batched, input_h, input_w) = match shape.as_slice() {
            [n, 3, h, w] => {
                let side = |v: i64| if v > 0 { v as u32 } else { DEFAULT_SIZE };
                (*n != 1, side(*h), side(*w))
            }
            _ => return Err(InferError::Unsupported(format!("input shape {shape:?}"))),
        };
        let out =
            session.outputs().first().ok_or_else(|| InferError::Unsupported("no output".into()))?;
        let out_shape = out.dtype().tensor_shape().map(|s| s.to_vec()).unwrap_or_default();
        if out_shape.len() != 2 {
            return Err(InferError::Unsupported(format!(
                "output shape {out_shape:?}, expected [N, D] (an embedding model)"
            )));
        }
        Ok(Self { session, path: path.to_path_buf(), input_w, input_h, batched })
    }

    /// Model input for one frame.
    pub fn preprocess(&self, img: &image::RgbImage) -> Vec<f32> {
        prepost::imagenet_chw_smooth(img, self.input_w, self.input_h)
    }

    /// Embeds preprocessed frames; vectors are L2-normalized.
    pub fn embed(&mut self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(inputs.len());
        let chunk = if self.batched { BATCH } else { 1 };
        for batch in inputs.chunks(chunk) {
            let shape = [batch.len(), 3, self.input_h as usize, self.input_w as usize];
            let data: Vec<f32> = batch.concat();
            let outputs = self.session.run(ort::inputs![Tensor::from_array((shape, data))?])?;
            let (oshape, values) = outputs[0].try_extract_tensor::<f32>()?;
            let dim = oshape.get(1).copied().unwrap_or(0) as usize;
            if dim == 0 || values.len() != dim * batch.len() {
                return Err(InferError::Unsupported(format!("output shape {oshape:?}")));
            }
            out.extend(values.chunks(dim).map(normalized));
        }
        Ok(out)
    }
}

fn normalized(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 && norm.is_finite() { v.iter().map(|x| x / norm).collect() } else { v.to_vec() }
}

/// Embeds many frames: decoder threads decode and preprocess while the model
/// runs batches. Progress and cancellation through `progress`.
pub fn run_embed(emb: &mut Embedder, items: Vec<BatchItem>, progress: &Progress) -> EmbedResult {
    progress.total.store(items.len(), Ordering::Relaxed);
    progress.done.store(0, Ordering::Relaxed);
    let next = AtomicUsize::new(0);
    let (tx, rx) = crossbeam_channel::bounded(2 * BATCH);
    let mut result = EmbedResult::default();
    let emb_ref = &*emb;
    let size = (emb_ref.input_w, emb_ref.input_h);
    std::thread::scope(|s| {
        for _ in 0..DECODERS {
            let tx = tx.clone();
            let (items, next) = (&items, &next);
            s.spawn(move || {
                loop {
                    if progress.cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(item) = items.get(i) else { return };
                    let input = decode_rgb(&item.path)
                        .map(|img| prepost::imagenet_chw_smooth(&img, size.0, size.1))
                        .map_err(|e| e.to_string());
                    if tx.send((item.key, input)).is_err() {
                        return;
                    }
                }
            });
        }
        drop(tx);
        let mut keys = Vec::with_capacity(BATCH);
        let mut inputs = Vec::with_capacity(BATCH);
        let mut flush = |keys: &mut Vec<usize>, inputs: &mut Vec<Vec<f32>>, r: &mut EmbedResult| {
            if keys.is_empty() {
                return;
            }
            match emb.embed(inputs) {
                Ok(v) => r.vectors.extend(keys.iter().copied().zip(v)),
                Err(e) => r.failed.extend(keys.iter().map(|k| (*k, e.to_string()))),
            }
            progress.done.fetch_add(keys.len(), Ordering::Relaxed);
            keys.clear();
            inputs.clear();
        };
        for (key, input) in rx.iter() {
            if progress.cancel.load(Ordering::Relaxed) {
                result.cancelled = true;
                break;
            }
            match input {
                Ok(data) => {
                    keys.push(key);
                    inputs.push(data);
                }
                Err(e) => {
                    result.failed.push((key, e));
                    progress.done.fetch_add(1, Ordering::Relaxed);
                }
            }
            if keys.len() == BATCH {
                flush(&mut keys, &mut inputs, &mut result);
            }
        }
        if !result.cancelled {
            flush(&mut keys, &mut inputs, &mut result);
        }
        // Unblock decoders waiting on a full channel.
        drop(rx);
    });
    result.cancelled |= progress.cancel.load(Ordering::Relaxed);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vectors_are_unit_length() {
        let v = normalized(&[3.0, 4.0]);
        assert_eq!(v, vec![0.6, 0.8]);
        assert_eq!(normalized(&[0.0, 0.0]), vec![0.0, 0.0]);
    }
}
