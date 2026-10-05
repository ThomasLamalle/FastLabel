
# FastLabel functionalities

FastLabel aims to reinvent annotation tooling by focusing on fast workflows. It focuses on fast annotation triage, since a human with good shortcuts is faster at accepting/denying annotations than at drawing them.

Native desktop app written in Rust, built for speed and stability (Linux/Wayland first, CPU-only must be fully usable).

Supports only bounding box annotations for now, but the data model supports N categories.
First use case: annotating bibs to train a bib detector model. See examples of dataset and labels:
[annotations](../Datasets/Ultralytics/v1_exp0/annotations.v022.json) [images](../Datasets/Ultralytics/v1_exp0/images) [predictions](../Datasets/Ultralytics/v1_exp0/predictions.json)

Two possible workflows:
- The preferred one: mosaic of crops, displaying ground truth (GT) annotations and the model prediction if it exists. Allow running a model on all frames or on a single frame, given a confidence threshold, and display its predictions. Display 'pred' or 'GT' on annotations. Predictions in red and GT in green.
- Classic frame-by-frame annotation/edition.

This file is the source of truth. The functional spec of the previous Python tool (`../BibDataset/REVIEW_FRAMES_FUNCTIONAL_SPEC.md`) is a reference for behavior not described here; where they conflict, this file wins.


## Data

- **Format: COCO** (`images`, `annotations`, `categories`). Existing files open unchanged.
- Image paths encode the split: `images/{train,val,test}/...`.
- YOLO/Ultralytics datasets (`data.yaml`, `images/`, `labels/`) can be imported; reviewed data is saved as COCO.
- **Versioning**: each save writes a new `annotations.vNNN.json` (never overwrites). On open, the most recent of `annotations.vNNN.json` / legacy `annotations.draft.json` is loaded.
- **Predictions** are stored separately in `predictions.json` (same schema as today: `version`, `model`, `confidence`, `images{id → {file_name, width, height, boxes[{id, category_id, bbox, score}]}}`). Loaded automatically on open.
- **Tool state** lives in `<dataset>/.fastlabel/`, never inside the COCO files:
  - `journal.jsonl`: append-only log of pending (unsaved) actions, replayed after a crash.
  - `review.json`: persistent review state (reviewed annotations, rejected predictions).
  - `cache/`: thumbnail cache.
  - `trash/`: deleted frames.
- **Frame deletion moves images to `.fastlabel/trash/vNNN/<original path>`**, never a hard delete. Metadata is always written (atomically) before any image is moved.
- **Adding images**: copy them anywhere under `images/` (e.g. `images/train/`). On open, images not referenced by the annotation file are added (no annotations yet) and recorded by the next save. Run the model on them with *Model → Run on frames without predictions*, then review their predictions in the mosaic, sorted by image date.
- **Export**: one action produces a zip with the current annotation file + `images/` tree (same layout as today's `v19.zip`), ready to import into Ultralytics.


## Mosaic mode

- Display a mosaic of crops as a **continuous, virtualized scroll** (only visible rows are loaded), with quick navigation using arrow keys or vim hjkl shortcuts.
- Number of crops per line should be adjustable.
- Crop margin in absolute height and width (original image pixels, each side) should be adjustable to allow more frame context in the crop.
- Crop content:
  - one crop per GT annotation; when it is matched to a prediction (TP) the crop shows both boxes: GT in green labelled `GT`, prediction in red labelled `pred` (+ confidence);
  - one crop per unmatched prediction (red, `pred`).
- Each crop shows its split (train/val/test) as a small badge. A split filter (All/train/val/test) is available.

### Shortcuts

All D/F/G marks are **toggles** (press again to undo the mark) and **auto-advance** the selection to the next crop still waiting for a decision (no D/G/R mark, frame not marked F).

- 'D' : on a GT crop, mark the annotation for deletion. On a prediction crop, reject the prediction (persisted, so it does not come back after the next model run). Display a red 'D' on the top right of the crop.
- 'G' : on a prediction crop, accept the prediction into the dataset. On a GT crop, confirm it as good. Display a green 'G' on the top right corner of the crop.
- 'F' : delete the frame associated with the crop. Display 'F' on all crops of this frame. Advances to the next undecided crop of a different frame. F is exclusive with D/G/R: marking a frame clears the D/G/R marks of its crops (same undo step), and D/G/R are refused on a frame marked F.
- 'R' : on a TP crop, replace the GT box geometry with the prediction's box (tightens sloppy boxes). Display an orange 'R'.
- Shift + D/F/G : toggle the shortcut for each crop from the selected one to the end of the row (each frame once for F), then select the first crop of the next row.
- D/F/G/R do not trigger instant deletion/acceptance, otherwise it would be laggy and constantly change the mosaic layout. The actual change is applied when saving.
- 'U' / Ctrl+Z : undo the last action (Ctrl+Shift+Z / Ctrl+Y : redo). Undo history covers everything since the last save.
- Enter or E : open the selected crop's full frame in the editor popup.
- Hold Space : peek at the full frame of the selected crop, overlaid on the mosaic.
- Tab or M : switch between mosaic mode and frame mode.
- 'S' : save the annotations to a new version, applying all changes. Simple incremental versioning; each save generates a new annotation file.
- 'Q' : save and quit the app.

### Review state

- Every crop on which an action was taken (D/G/R/F) is marked **reviewed**, persisted across sessions.
- Toggle filter "hide reviewed" so a review pass never shows the same crop twice.
- Rejected predictions are remembered by geometry: after a new model run, a prediction overlapping a previously rejected one (same image, same class, IoU ≥ 0.9) is shown as already rejected.

### Sorting

Multiple sortings are allowed in mosaic mode to make reviewing faster, each with an asc/desc option. Some are always available, others only when the information exists (with a button to trigger the computation):

Always available:
- by id
- by height
- by width

Others:
- by IoU with prediction
- by prediction confidence
- by color uniformity: grayscale standard deviation inside the bbox. Ascending surfaces suspiciously uniform crops (blank/occluded), descending surfaces very busy crops. Computed in the background and cached.
- by image date: file creation time (a copied file gets a new one; falls back to modification time), newest first, to review recently added images.

### Filters

Similar to sorting, toggle buttons allow to:
- display only FP: predictions with no intersection with a GT annotation, or with best IoU below a tunable threshold (default **0.3**)
- display only FN: GT with no intersection with a prediction, or with best IoU below a tunable threshold (default **0.3**)
- hide reviewed crops
- filter by split

Thresholds (all adjustable in the UI, remembered between sessions):

| Threshold | Default | Role |
|---|---:|---|
| Match IoU | 0.5 | One-to-one greedy GT↔prediction matching (TP / IoU sort) |
| FP filter IoU | 0.3 | Prediction whose best IoU with any GT is below → FP filter |
| FN filter IoU | 0.3 | GT whose best IoU with any prediction is below → FN filter |
| Inference confidence | 0.10 | Minimum score kept when running the model |

The 0.3 filter defaults isolate real misses; loose-but-overlapping boxes (0.3–0.5) are found with the IoU ascending sort.


## Frame mode

- Navigate frame by frame (h/l, arrows; Ctrl to jump to the next annotated frame).
- Draw new boxes, select, move, resize with 8 handles (corners + edges), delete with D.
- Zoom (wheel) and pan (middle-drag / Space-drag).
- Change the category of the selected box with keys 1–9.
- Free-form frame tags (comma separated).
- F marks the frame for deletion.
- Run the model on the current frame only.
- The same editor is used in a popup when pressing Enter on a crop in the mosaic. "Run on this frame" is also available from the mosaic.
- Frame-mode edits are pending actions like mosaic marks: they are visible immediately in frame view, applied to the dataset (and the mosaic layout) on save.


## Model inference

- ONNX models run through ONNX Runtime, no Python at runtime. CPU only must be fast enough (no GPU on the dev machine); GPU execution providers are optional.
- Supported architectures: **YOLO26** (first), **RF-DETR** (next). The model's class names are mapped to dataset categories by name.
- Run on all frames (in the background, with progress and cancel) or on a single frame.
- Results are written to `predictions.json`; a failed or cancelled run never replaces the previous predictions.


## Session

The last session is restored on launch: dataset, mode, scroll position, sort, filters, thresholds, crops per line, margins, model path, confidence.


# TODO: ideas for later (don't implement yet)

- Predict once at a very low confidence, then filter predictions live with a confidence slider (no re-run).
- More sorts: aspect ratio, area, blur (Laplacian variance), boxes touching the frame border.
- Duplicate GT filter: two GT boxes on the same frame with IoU > 0.7.
- Headless CLI: `fastlabel predict | stats | export | diff vN vM`.
- Version diff view: what changed between two annotation versions.
- Export rejected predictions as hard negatives.
- Trash browser to restore deleted frames.
- GPU execution providers (CUDA / TensorRT) for inference.
- Stats panel: counts per split/class, box size histograms.


# Future functionality to consider but don't implement yet

## Manage splitting: clustering

Easily manage and rearrange the train/val/test split. One functionality from the Ultralytics platform that is very good is clustering:
compute image "embeddings" and display clusters of the images as a point cloud next to the mosaic of images. You can then select points/clusters by drawing around them and apply modifications to them, for example moving a cluster to a different split. There is also a global toggle button that displays All/train/val/test images, and it applies to the point cloud.
One feature it misses is that you can't easily apply modifications to a cluster: you have to manually select the corresponding images and apply the action. It would be better to right-click the selected cluster and apply an action on it directly.
We could also apply this clustering feature to the annotations/crops instead of only to the images.

## Cleaning dataset

- If we could spot annotations/images that are too similar, we could spot possible wrong labels and also delete duplicated images/annotations.
