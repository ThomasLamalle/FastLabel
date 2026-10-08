
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
- **The split is data**: each image has a `split` field (`train` / `val` / `test`) in the annotation file, so changing a split never moves files and every version records the split. Files without the field (older versions, YOLO imports) take the split from the path (`images/{train,val,test}/...`); the next save writes the field. Images can sit anywhere under `images/`.
- YOLO/Ultralytics datasets (`data.yaml`, `images/`, `labels/`) can be imported; reviewed data is saved as COCO.
- **Versioning**: each save writes a new `annotations.vNNN.json` (never overwrites). On open, the most recent of `annotations.vNNN.json` / legacy `annotations.draft.json` is loaded.
- **Predictions** are stored separately in `predictions.json` (same schema as today: `version`, `model`, `confidence`, `images{id → {file_name, width, height, boxes[{id, category_id, bbox, score}]}}`). Loaded automatically on open.
- **Tool state** lives in `<dataset>/.fastlabel/`, never inside the COCO files:
  - `journal.jsonl`: append-only log of pending (unsaved) actions, replayed after a crash.
  - `review.json`: persistent review state (reviewed annotations, rejected predictions).
  - `cache/`: thumbnail cache.
  - `trash/`: deleted frames.
- **Frame deletion moves images to `.fastlabel/trash/vNNN/<original path>`**, never a hard delete. Metadata is always written (atomically) before any image is moved.
- **Adding images**: *Add → Images… / Folders…* copies them into `images/` under a unique content-hash name (`images/<first 16 hex of SHA-256>.<ext>`); the original path is kept in the image's `source_name` field. An image whose content is already in the dataset is skipped (reported). Copying runs in the background (progress, cancel); the new frames appear at once, optionally with a split (*Split of the added frames*, pending like any split change), and are recorded by the next save (crash-safe: source names in `.fastlabel/sources.json`, ids in the journal header). Each added image gets a `date_added` field (ISO 8601 UTC), which the image date sort uses (file dates do not survive copies). Annotate them in the **To do** tab, or run the model on them with *Model → Run on frames without predictions* and review the predictions in the mosaic, sorted by image date. Images copied by hand anywhere under `images/` are still picked up on open.
- **Export** (*Export…*, or `fastlabel --export OUT --format yolo|coco|coco1|ndjson <dataset>`): pending edits are saved first, then the dataset is written laid out by split, to a zip or to a new folder (images hardlinked, so it is instant and takes no space; a cancelled or failed export leaves nothing):
  - **YOLO** (Ultralytics, YOLO26): `data.yaml`, `images/<split>/`, `labels/<split>/*.txt` (normalized, clipped to the image; empty file for frames without boxes).
  - **COCO per split** (Roboflow layout, RF-DETR): `train/`, `valid/`, `test/`, each with `_annotations.coco.json` next to its images.
  - **COCO single file**: `annotations.json` with `split` fields + `images/<split>/`.
  - **Ultralytics NDJSON**: `<dataset>.ndjson` (local `path`) + `images/<split>/`, readable by Ultralytics' NDJSON converter.
  - Splits to include are chosen in the dialog; frames without a split are left out and counted. Two frames of a split with the same file name get an `<id>_` prefix.
- **Tabs** (toolbar): *Mosaic (M)*, *Frame (M)*, *To do (T)*, *Clusters (C)*, *Rules*; the shortcuts still work.
- **Labeling rules** (*Rules* tab): the dataset's labeling rules as free text, editable in place (shortcuts are off while typing; Esc goes back to the mosaic). Stored in `.fastlabel/rules.md`, written when the field loses focus, on leaving the tab and on exit. A new dataset starts with: *Label the whole bib. If occulted, box only the visible part.*

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
- 'F' : delete the frame associated with the crop. Display 'F' on all crops of this frame. Advances to the next undecided crop of a different frame. F is exclusive with D/G/R: marking a frame clears the D/G/R marks of its crops, and D/G/R on a crop of a frame marked F applies the mark and unmarks the frame (each in one undo step).
- 'R' : on a paired crop (GT + prediction), replace the GT box geometry with the prediction's box (tightens sloppy boxes). Display an orange 'R'.
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
- by aspect ratio (width / height)
- by area

Others:
- by IoU with prediction
- by prediction confidence
- by color uniformity: grayscale standard deviation inside the bbox. Ascending surfaces suspiciously uniform crops (blank/occluded), descending surfaces very busy crops. Computed in the background and cached.
- by image date: the image's `date_added` field, else the file creation time (falls back to modification time), newest first, to review recently added images.

### Filters

Similar to sorting, toggle buttons allow to:
- display only FP: predictions overlapping no GT box (IoU 0 with all of them)
- display only FN: GT boxes overlapping no prediction
- hide reviewed crops
- filter by split (All / train / val / test / no split)
- *No boxes*: frames without GT boxes, i.e. the crops of their predictions (the frames themselves are in the To do tab). Also applies to the cluster view.

No thresholds to tune. Each GT box is paired with its best-overlapping prediction (any overlap); the pair is shown in one crop with its IoU, and loose boxes are found with the IoU ascending sort. An unpaired prediction overlapping an already paired GT (a duplicate) is labelled "extra".


## Frame mode

- Navigate frame by frame (h/l, arrows; Ctrl to jump to the next annotated frame; right click on the frame: next frame).
- Draw new boxes, select, move, resize with 8 handles (corners + edges), delete with D.
- Zoom (wheel) and pan (middle-drag / Space-drag).
- Change the category of the selected box with keys 1–9.
- Free-form frame tags (comma separated).
- F marks the frame for deletion.
- Run the model on the current frame only.
- The same editor is used in a popup when pressing Enter on a crop in the mosaic. "Run on this frame" is also available from the mosaic.
- Frame-mode edits are pending actions like mosaic marks: they are visible immediately in frame view, applied to the dataset (and the mosaic layout) on save.


## Cleaning workflows

Toolbar menu *Cleaning*, each workflow is a dedicated review mode.

### Overlapping GT boxes

Double annotations: GT boxes of the same category that overlap each other (IoU > 0; on v1_exp0 almost all such pairs cover most of the smaller box). Groups are reviewed one at a time, zoomed onto the group, every box with its own color and number (left to right).

- 1–9: keep that box (confirmed), queue the others of the group for deletion, go to the next group. Groups with 10 or more boxes: type the number, then Enter.
- A: keep all boxes (e.g. after fixing them by hand), next group.
- D: delete all boxes of the group, next group.
- Progress (group n / total) is shown in the bottom status line.
- Boxes stay editable (move, resize, draw) before deciding.
- F: delete the frame, next group. Right / Left: next / previous group. Esc (or M / Tab / Q): back to the mosaic.
- Decisions are pending edits like any other (undo with U, applied on save). After a save the remaining groups are rescanned; resolved groups (deleted boxes, all boxes kept or reviewed) do not come back.

## To do tab

Toolbar *To do* or `T` in the mosaic: frames without any GT box (typically just added) that are not confirmed as background, as whole-frame tiles, newest first (image date), with their source name, split, date and whether the model was run on them. The split filter applies (*no split* finds frames added without a split). *Show background* also lists the frames already saved as background (labelled *background*; G / F auto-advance and skipping forward in frame mode pass over them).

- Enter / E / double click: frame mode restricted to the to-do frames (*To do only* in its header, on by default): h / l, arrows and right click go to the next / previous to-do frame in the same order, with the full editor (draw, zoom, categories, tags, run the model on the frame). Moving forward (l, →, right click) past a frame left untouched (no box drawn, not marked F) marks it as background, so skipping is enough for frames without bibs; going back marks nothing, and G on a marked frame takes the mark back. G with no box selected marks the frame as background and moves on; F deletes it and moves on. Tab (or Esc with no box selected) goes back to the grid, on the frame reached. The tile of a frame with new boxes shows "boxes drawn".
- G: background, nothing to annotate (toggle; persisted in `review.json`, so the frame leaves the list for good). Background means no boxes: G is refused on a frame with boxes; drawing a box or accepting a prediction on a frame marked background takes the mark back (same undo step); a save records a frame as background only if it has no boxes, and a background frame that gets boxes later stops being one. Its predictions are left as they are. In frame mode, G with no box selected does the same on any frame without boxes.
- F: delete the frame (toggle). G and F advance to the next undecided frame; both are pending until saved, U undoes.
- Arrows / hjkl, PageUp / PageDown, Home / End: move. Esc, T, M or Tab: back to the mosaic.
- After a save, frames with boxes or confirmed as background leave the list.

## Cluster mode: managing the splits

Toolbar *Clusters* or `C` in the mosaic. Every frame is a point of a 2D map built from image embeddings, so similar frames (same scene, same camera, consecutive video frames) sit together. This shows split leakage (near-identical frames in train and val) and lets whole groups be moved at once.

- **Embeddings**: an ONNX image embedder (input `[N, 3, H, W]` ImageNet-normalized RGB, output `[N, D]`), DINOv2-small by default (`scripts/export_dinov2.py`). Computed in the background with progress and cancel, cached per frame in `.fastlabel/embeddings.bin` (frames moved to another split keep their vector; a different model recomputes everything). Only frames without a vector are computed.
- **Layout**: PCA (50 dims) → UMAP 2D → HDBSCAN clusters (noise in grey), computed in the background and cached in `.fastlabel/clusters.json`. New frames: *Update layout*. *min cluster size* + *Recluster* only recomputes the clusters.
- **Display**: points colored by split (pending moves included) or by cluster; frames marked F in dark grey. The global split filter (All/train/val/test) and *No boxes* apply to the cloud. Wheel zooms around the cursor, middle drag (or Space + drag) pans, *Fit view* resets. Hovering a point shows the frame, its split and cluster. Counts per split are shown for the visible frames and for the selection.
- **Selection**: drag a lasso around points (Shift adds, Ctrl removes), click a point (Shift/Ctrl toggles), click empty space to clear, Ctrl+A selects all visible frames. The selected frames are listed as thumbnails below the cloud; double click a point or a thumbnail to open the frame in the editor popup.
- **Actions** (right click on the cloud, or keys), one undo step each:
  - Move to train / val / test (`1` / `2` / `3`): pending until saved (the `split` field changes; files stay where they are).
  - Show in mosaic (`Enter`): the mosaic shows only the crops of the selected frames until the *Cluster selection ✕* chip is cleared.
  - Delete frames (`F`): F on every selected frame (or un-F when they all are marked).
- **Auto-split…** (cluster mode toolbar): target shares for val and test (default 10 % each); whole clusters go to one split so near-identical frames never straddle two splits, and a cluster may fill at most a set share of the val / test target (default 25 %, so val and test span several scenes; bigger clusters stay in train). Unclustered frames are placed one by one, frames marked F are left alone. Groups are drawn in a seeded random order: *Reroll* tries another draw. One undo step, pending until saved; check it with the report.
- Esc, C, M or Tab: back to the mosaic.
- Split shares (frames and % per split, pending moves included) are always shown in the status line, and in cluster mode for the visible frames and the selection.

## Reports

*Report* (toolbar, all frames), or in cluster mode *Report: all frames*, *Report on selection*, *Report on cluster #n* (right click on a point, which also offers *Select cluster #n*). Headless: `fastlabel --report out.html <dataset>`. A self-contained HTML page (plots in SVG, light/dark) is written to `.fastlabel/reports/` and opened in the browser. Pending split moves are included; frames marked F and boxes marked D are left out.

- Split balance: images, boxes, images without boxes, boxes per image (mean, median, max) per split, with shares; boxes per category when there are several.
- Distributions per split, normalized so splits of different sizes compare: boxes per image, box center heatmaps (with mean center and spread), box short side in pixels, box height relative to the image, aspect ratio.
- Image resolutions per split.
- Model input size guide: for input sizes 320–1920, the box short side once letterboxed (p5, median) and the share of boxes under 10 px; suggests the smallest size keeping at most 2 % under 10 px.
- Box quality flags: tiny boxes, boxes touching the image border, extreme aspect ratios.
- Model results per split (from `predictions.json`): recall, precision, mean IoU, loose pairs.
- Similarity between splits (needs the embeddings): for each val / test frame, the most similar frame of another split; near duplicates (cosine ≥ 0.95) leak.
- Clusters × splits (needs the layout): composition of each cluster, share of train frames in scenes val / test never see.

## Model inference

- ONNX models run through ONNX Runtime, no Python at runtime. CPU only must be fast enough (no GPU on the dev machine); GPU execution providers are optional.
- Ultralytics `.pt` weights can be chosen too: the first run exports them to `<name>.onnx` next to the weights (`uvx`, CPU torch, training image size) and reuses that file until the `.pt` changes.
- Supported architectures: **YOLO26** (first), **RF-DETR** (next). The model's class names are mapped to dataset categories by name.
- Run on all frames (in the background, with progress and cancel) or on a single frame.
- Results are written to `predictions.json`; a failed or cancelled run never replaces the previous predictions.


## Session

The last session is restored on launch: dataset, mode, scroll position, sort, filters, thresholds, crops per line, margins, model path, confidence, embedding model path, minimum cluster size.


# TODO: ideas for later (don't implement yet)

- Predict once at a very low confidence, then filter predictions live with a confidence slider (no re-run).
- More sorts: blur (Laplacian variance), boxes touching the frame border.
- Duplicate GT filter: two GT boxes on the same frame with IoU > 0.7.
- Headless CLI: `fastlabel predict | stats | export | diff vN vM`.
- Version diff view: what changed between two annotation versions.
- Export rejected predictions as hard negatives.
- Trash browser to restore deleted frames.
- GPU execution providers (CUDA / TensorRT) for inference.
- Stats panel: counts per split/class, box size histograms.


# Future functionality to consider but don't implement yet

## Clustering crops

Apply cluster mode to the annotations/crops instead of only to the images (same pipeline, embedding crops instead of frames).

## Cleaning dataset

- If we could spot annotations/images that are too similar, we could spot possible wrong labels and also delete duplicated images/annotations.
