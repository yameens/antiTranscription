# antiTranscription

A Python pipeline that takes a phone photo of printed sheet music and outputs a playable MIDI file.

## Pipeline stages

1. **Page rectification** (`rectify.py`) detects the page boundary via Otsu thresholding and a morphological close, fits a rotated bounding rectangle, and warps the page to a fixed 850 × 1100 px canvas.
2. **Staff line detection** (`detect_staves.py`) binarizes the rectified image, runs the probabilistic Hough transform, filters near-horizontal segments, clusters y-coordinates, and groups lines into 5-line staves.
3. Staff line removal and symbol segmentation *in progress*
4. Symbol classification (CNN trained on PrIMuS) *in progress*
5. MIDI reconstruction *in progress*

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
