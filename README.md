# antiTranscription

A Python pipeline that takes a phone photo of printed sheet music and outputs a playable MIDI file.

## Pipeline stages

1. **Page rectification** (`rectify.py`) detects the page boundary via Otsu thresholding and a morphological close, fits a rotated bounding rectangle, and warps the page to a fixed 850 × 1100 px canvas.
2. **Staff line detection** (`detect_staves.py`) binarizes the rectified image, runs the probabilistic Hough transform, filters near-horizontal segments, clusters y-coordinates, and groups lines into 5-line staves.
3. Staff line removal and symbol segmentation (`segment_symbols.py`).
4. Symbol classification (CNN trained on PrIMuS, `symbol_classifier.py`).
5. MIDI reconstruction (`pitch_midi.py`).

### Reading paths (three ways to get notes off a page)

The classical segment-then-classify path (stages 3–5) cascades segmentation
errors and shows a large clean-to-real domain gap (22.7% on real photos). Two
stronger readers were added:

- **Geometry-only** (`pitch_geometry.py`) — template-matched noteheads + staff
  geometry. Segmentation-free, pitch-only. 60–86% exact pitch on the test photos.
- **End-to-end CRNN+CTC** (`crnn_omr.py`, `train_crnn.py`, `read_crnn.py`) — a
  CNN→BiLSTM→CTC sequence model that reads a whole staff image directly into a
  semantic token sequence (pitch **and** duration), with no segmentation stage.
  Trained on PrIMuS treble staves with synthetic phone-photo augmentation
  (`camera_aug.py`) to close the domain gap.

## CRNN usage

```bash
# 1. download + extract PrIMuS into data/primus/, then build the sequence manifest + vocab
python omr_dataset.py                 # writes data/seq_manifest.csv, data/vocab.json

# 2. train (MPS/GPU recommended; falls back to CPU)
python train_crnn.py --epochs 25 --augment camera --device mps --out data/crnn_camera.pt
python train_crnn.py --epochs 25 --augment none   --device mps --out data/crnn_clean.pt   # ablation baseline

# 3. evaluate
python test_crnn.py --ckpt data/crnn_camera.pt        # three-way pitch/duration/SER on real photos
python eval_ablation.py --ckpt data/crnn_camera.pt    # clean vs camera-augmented val SER (domain gap)

# 4. transcribe a photo to MIDI
python read_crnn.py yankeeDoodle                       # -> results/yankeeDoodle_crnn.mid
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# rectify a single image
python test_rectify.py "sheet music/yankeeDoodle.jpg"

# rectify all test images
python batch_test.py

# detect staves on all test images
python test_staves.py

# regenerate milestone figures
python make_figures.py
```

## Test images

Five monophonic beginner scores in `sheet music/`: cs131, londonBridgeIsFalling, maryHadLittleLamb, twinkleTwinkleLittleStar, yankeeDoodle. Ground-truth note sequences (pitch + duration) are in `sheet music/sheetNote/`.

## Results

Run `python make_figures.py` to regenerate figures in `results/`. Current status: 5/5 images rectify successfully, 12 staves detected across all 5 images.

## Course

Stanford CS 131 — Computer Vision, Spring 2026
# antiTranscription
