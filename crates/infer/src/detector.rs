//! ONNX Runtime detector with automatic output-layout detection.

use std::path::{Path, PathBuf};

use image::RgbImage;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

use crate::prepost::{self, Detection, Letterbox};

#[derive(Debug, thiserror::Error)]
pub enum InferError {
    #[error("ONNX Runtime: {0}")]
    Ort(String),
    #[error("unsupported model: {0}")]
    Unsupported(String),
}

impl<T> From<ort::Error<T>> for InferError {
    fn from(e: ort::Error<T>) -> Self {
        Self::Ort(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, InferError>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    /// YOLO raw head `[1, 4 + nc, anchors]`, needs NMS.
    YoloRaw,
    /// YOLO end-to-end `[1, n, 6]`.
    YoloEnd2End,
    /// DETR-style `boxes` + `logits` outputs (RF-DETR).
    Detr,
}

/// NMS IoU and detection cap, Ultralytics defaults.
const NMS_IOU: f64 = 0.7;
const MAX_DET: usize = 300;

pub struct Detector {
    session: Session,
    pub path: PathBuf,
    pub arch: Arch,
    pub input_w: u32,
    pub input_h: u32,
    /// Dynamic input shape: frames are letterboxed to the smallest
    /// stride-aligned size instead of a square.
    pub dynamic: bool,
    stride: u32,
    names: Vec<String>,
    /// Output indices: (boxes, logits) for DETR, (0, _) otherwise.
    outputs: (usize, usize),
}

impl Detector {
    /// Loads a model; `threads = 0` lets ONNX Runtime decide.
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
        let (meta_imgsz, meta_end2end, meta_names, meta_stride) = {
            let meta = session.metadata()?;
            (
                meta.custom("imgsz"),
                meta.custom("end2end"),
                meta.custom("names"),
                meta.custom("stride"),
            )
        };
        let stride = meta_stride.and_then(|s| s.trim().parse().ok()).unwrap_or(32);
        let dynamic = !matches!(shape.as_slice(), [_, 3, h, w] if *h > 0 && *w > 0);
        let (input_h, input_w) = match shape.as_slice() {
            [_, 3, h, w] if *h > 0 && *w > 0 => (*h as u32, *w as u32),
            _ => {
                let size = meta_imgsz
                    .and_then(|s| s.trim_matches(['[', ']']).split(',').next()?.trim().parse().ok())
                    .unwrap_or(640);
                (size, size)
            }
        };

        let out_names: Vec<String> =
            session.outputs().iter().map(|o| o.name().to_string()).collect();
        let out_shapes: Vec<Vec<i64>> = session
            .outputs()
            .iter()
            .map(|o| o.dtype().tensor_shape().map(|s| s.to_vec()).unwrap_or_default())
            .collect();
        let find = |cands: &[&str]| out_names.iter().position(|n| cands.contains(&n.as_str()));
        let (arch, outputs) = if let (Some(b), Some(l)) =
            (find(&["dets", "boxes", "pred_boxes"]), find(&["labels", "logits", "pred_logits"]))
        {
            (Arch::Detr, (b, l))
        } else if meta_end2end.as_deref() == Some("True")
            || matches!(out_shapes.first().map(Vec::as_slice), Some([_, _, 6]))
        {
            (Arch::YoloEnd2End, (0, 0))
        } else if out_names.len() == 1 {
            (Arch::YoloRaw, (0, 0))
        } else {
            return Err(InferError::Unsupported(format!("unrecognised outputs {out_names:?}")));
        };

        let mut names = meta_names.map(|n| prepost::parse_names(&n)).unwrap_or_default();
        if names.is_empty()
            && arch == Arch::YoloRaw
            && let Some([_, c, _]) = out_shapes.first().map(Vec::as_slice)
        {
            names = (0..(*c - 4).max(0)).map(|i| format!("class{i}")).collect();
        }
        Ok(Self {
            session,
            path: path.to_path_buf(),
            arch,
            input_w,
            input_h,
            dynamic,
            stride,
            names,
            outputs,
        })
    }

    pub fn class_names(&self) -> &[String] {
        &self.names
    }

    pub fn detect(&mut self, img: &RgbImage, conf: f32) -> Result<Vec<Detection>> {
        let (w, h) = img.dimensions();
        let (in_w, in_h) = if self.dynamic && self.arch != Arch::Detr {
            Letterbox::rect_input(w, h, self.input_w.max(self.input_h), self.stride)
        } else {
            (self.input_w, self.input_h)
        };
        let shape = [1usize, 3, in_h as usize, in_w as usize];
        match self.arch {
            Arch::YoloRaw | Arch::YoloEnd2End => {
                let (data, lb) = prepost::letterbox_chw(img, in_w, in_h);
                let arch = self.arch;
                let outputs = self.session.run(ort::inputs![Tensor::from_array((shape, data))?])?;
                let (oshape, out) = outputs[0].try_extract_tensor::<f32>()?;
                let dims: Vec<usize> = oshape.iter().map(|&d| d as usize).collect();
                Ok(decode_yolo(arch, out, &dims, conf, &lb, w, h)?)
            }
            Arch::Detr => {
                let data = prepost::imagenet_chw(img, in_w, in_h);
                let (bi, li) = self.outputs;
                let outputs = self.session.run(ort::inputs![Tensor::from_array((shape, data))?])?;
                let (bshape, boxes) = outputs[bi].try_extract_tensor::<f32>()?;
                let (lshape, logits) = outputs[li].try_extract_tensor::<f32>()?;
                let queries = bshape.get(1).copied().unwrap_or(0) as usize;
                let nc = lshape.get(2).copied().unwrap_or(0) as usize;
                if boxes.len() < queries * 4 || logits.len() < queries * nc {
                    return Err(InferError::Unsupported("DETR output shapes".into()));
                }
                Ok(prepost::decode_detr(boxes, logits, queries, nc, conf, w, h))
            }
        }
    }
}

fn decode_yolo(
    arch: Arch,
    out: &[f32],
    dims: &[usize],
    conf: f32,
    lb: &Letterbox,
    w: u32,
    h: u32,
) -> Result<Vec<Detection>> {
    match (arch, dims) {
        (Arch::YoloRaw, [1, c, a]) if *c > 4 => {
            Ok(prepost::decode_yolo_raw(out, (*c, *a), conf, lb, w, h, (NMS_IOU, MAX_DET)))
        }
        (Arch::YoloEnd2End, [1, n, 6]) => Ok(prepost::decode_yolo_end2end(out, *n, conf, lb, w, h)),
        _ => Err(InferError::Unsupported(format!("output shape {dims:?} for {arch:?}"))),
    }
}
