# Online Bangla Handwriting Word Detection and Incremental Learning

This repository is the scaffold for a full-stack Bangla handwriting recognition web app.

## Current phase

The initial scaffold is in place. The word viewer also includes a Segmentation section that
splits the entered Bangla word into grapheme clusters and lets you remove or add character chips.
CSV/TXT upload now parses pen traces and displays the writing in the canvas. Labeled samples
persist in SQLite and Train rebuilds the DTW word recognizer. An experimental YOLOv11
character detector is trained from the isolated character trace dataset by composing
synthetic multi-character lines with known character boxes. Character detections are grouped
by their horizontal gaps into word boxes. Corrected boxes for the exact same uploaded trace
persist and are restored on repeat uploads.

## Structure

- `backend/app/` – FastAPI application, configuration, database models, routers, and ML skeleton modules.
- `frontend/` – static HTML/CSS/JS layout for the web UI.
- `data/` – upload, database, and checkpoint storage.
- `scripts/` – reproducible data generation utilities.

## Run

This project targets Python 3.10.

```bash
cd bangla-ohr
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000/

## Synthetic data

```bash
python scripts/make_synthetic_data.py --output data/synthetic_pen_data.txt --words 8 --lines 2
```

The generator keeps word spacing clear and distributes the exact requested word count
across lines, including when the count does not divide evenly.

Upload CSV/TXT files with `x,y,pen` rows. A header is optional; comma, tab, and whitespace
separators are accepted. Pen state `1` means down and `0` means up. The upload endpoint
currently accepts UTF-8 files up to 10 MB.
The handwriting viewer opens each upload focused on the selected word so its strokes are
large and clear. Choose **Full page** to see all detected words together. Use the mouse wheel
or `+`/`−` buttons to zoom, drag to pan horizontally or vertically, and **Fit** to reset zoom.
For uncertain handwritten boundaries, use **Correct word boxes** in full-page view: select a
box, use **Re-detect boxes** to run the current word segmentation again on the upload, draw a
replacement around the word's strokes, or delete a false box. Corrections save to the current
upload session. Check the detected boundaries before labeling words so labels remain attached
to the intended handwriting sample.
After uploading a script, choose **Show full prediction** to recognize every detected word
without manually reviewing or labeling them one at a time. The separate results view shows the
full handwriting plot, the supplied Bangla passage, and a numbered list of detected words with
their confidence; words the current model cannot recognize appear as **Unknown**. Recognition
quality depends on saved training samples, so train the word model before expecting useful
predictions.
Manually corrected boxes are saved with the uploaded trace. Uploading the exact same file again
restores its saved strokes and boxes. After correcting complete word boundaries, press **Train**
to train the YOLOv11 character detector from the available character dataset. Training runs in
the background; its status and synthetic validation mAP50 are shown after completion. The
detector weights persist in `data/checkpoints/yolo11-character.pt` and are loaded at backend
startup. On CPU the app uses a smaller training run; a CUDA GPU enables the larger run. The
first training run downloads the public `yolo11n.pt` starting weights.
The manually corrected word boxes also continue to train the separate spacing-threshold model.

Set `BANGLA_CHARACTER_DATASET_DIR` before starting the backend if the
`OHR_Bangla_Characters_Checked` folder is not at the default sibling `Canvas` location.

## Notes

The available dataset contains isolated character traces, not real word/line images or word
box labels. Synthetic layouts provide character-box training examples, but the reported mAP50
only measures synthetic validation images and does not establish accuracy on real handwritten
sentences. YOLO detects character classes; a spacing rule groups those detections into words.
For a real-world word-box accuracy measurement, collect and manually label full-line traces
from representative writers. The synthetic character mAP is not a real handwriting accuracy
score. When no trained YOLO weights are available, the app falls back to its height-and-gap
heuristic segmenter. Each retained stroke is assigned to one detected box.
Adjacent word boxes on the same line are clipped at their shared midpoint so their padded
rectangles cannot overlap. Segmentation remains heuristic, so unusual or tightly joined
handwriting can still need manual correction.
The segmentation chips can be edited before saving and are stored with each sample.
The current trainer is a CPU-friendly DTW template baseline, not the later neural CTC model.
