# Sheet-Music-to-MIDI Pipeline — Progress Report

**Author:** Yameen Sekandari
**Course:** CS131, Spring 2026
**Scope:** monophonic printed scores, treble clef, durations whole through eighth, no accidentals/key signatures.

This report summarizes the full state of the project: a five-stage computer-vision
pipeline that takes a phone photo of printed sheet music and produces a playable
MIDI file. It documents what was built at each stage, the quantitative results,
where the supporting code lives, and which figures back each claim.

---

## 1. Executive summary

The complete end-to-end pipeline runs: **photo -> rectify -> detect staves ->
remove staves + segment -> classify (CNN) -> infer pitch -> MIDI**. All five
stages described in the proposal are implemented and tested on real phone photos.

Headline numbers:

| Metric | Value |
| --- | --- |
| Rectification quality (5 images) | 5/5 PASS, staff slant < 0.25 deg, border background < 1% |
| CNN PrIMuS validation accuracy (clean -> clean) | **58.9%** |
| CNN local accuracy (clean -> real phone photo) | **22.7%** |
| Domain gap | **-36.2 percentage points** |
| Clean labeled training crops mined from PrIMuS | **35,933** (from 87,678 staves) |
| Best end-to-end pitch (yankeeDoodle, +-1 step) | **50%** |

The domain gap (clean training vs messy phone-photo testing) is the central
result the proposal set out to measure.

---

## 2. Technical Progress (by stage)

### Stage 1 — Page rectification

The rectification stage was overhauled twice. The original proposal version used
grayscale -> Gaussian blur -> Canny (50/150) -> `findContours` -> `approxPolyDP`
(2% epsilon) to find four page corners. This worked on well-framed images but
failed on angled/uneven-lit photos: it would lock onto a single stave as the
largest closed contour and warp that into the 850x1100 canvas, producing
zoomed-in, distorted output.

The current implementation replaces the brittle edge-based detector with a
**robust corner-detection cascade backed by lighting-robust segmentation**:

1. **Segmentation cascade** (handles uneven lighting / shadows):
   - Pass 1: raw Otsu + small morphological close (well-lit pages).
   - Pass 2: CLAHE-equalized + Otsu + height-scaled close.
   - Pass 3: CLAHE + fixed bright-pixel threshold + scaled close.
   - Pass 4: CLAHE + adaptive threshold + scaled close (last resort).
2. **Corner-detection cascade** on the sealed page mask:
   - Primary: `_corners_via_hough` — detect the four page edges as lines, then
     intersect them. This survives a partially occluded corner because it only
     needs the edges, not the full boundary.
   - Fallback 1: `_corners_via_polydp` — adaptive-epsilon `approxPolyDP` to a
     4-vertex convex quad.
   - Fallback 2: `_corners_via_minrect` — `minAreaRect` (original behavior).
3. `_validate_quad` rejects non-convex, too-small, or wrong-aspect quads.
4. Corners ordered (TL, TR, BR, BL) by sum/difference, homography via
   `getPerspectiveTransform`, warp to an 850x1100 canvas via `warpPerspective`.

**Code:** `rectify.py`
- Segmentation cascade + kernel scaling: `_find_page_corners` (lines 321-389)
- CLAHE helper: `_clahe_equalize` (lines 94-104)
- Edge-intersection corner detector: `_corners_via_hough` (lines 199-289)
- Quad validation: `_validate_quad` (lines 150-198)
- Public entry points: `rectify_page` (line 390), `rectify_page_debug` (line 401)

**Figures:** `01_rectification/` (final outputs), `07_progress_and_diagnostics/rectification_before_fix/` (the broken edge-based outputs, for a before/after comparison).

### Stage 2 — Staff line detection

Adaptive threshold (block size 25, C=10) to handle varying light, inverted so
staff lines are white, then `HoughLinesP` with a tight angle filter to keep only
near-horizontal segments. Detected y-values are clustered (`_cluster_ys`) and
grouped into 5-line staves using a gap-split plus a sliding-window validator that
checks each candidate stave has 5 lines with consistent spacing (coefficient of
variation < 0.4). Mean intra-stave spacing sets the vertical scale for every
later stage.

**Code:** `detect_staves.py`
- Main detector: `detect_staves` (lines 37-216)
- y-clustering: `_cluster_ys` (lines 23-34)
- Stave validity check (5 lines, even spacing): `_is_valid_stave` (lines 131-139)

**Figures:** `02_staff_detection/` (`*_staves_compare.png`, one colored line per stave).

### Stage 3 — Staff removal and symbol segmentation

Adaptive threshold + invert, then a **mask-based staff-line erasure** at the
known staff y-coordinates (more precise than a global horizontal opening, which
would also erase ledger lines and beams). A vertical morphological close bridges
note stems that were severed by the mask strips, then a small open removes stub
specks. Connected-component analysis extracts symbol bounding boxes, filtered by
size, and returns left-to-right / top-to-bottom sorted crops.

**Code:** `segment_symbols.py`
- Main routine: `remove_staves_and_segment` (lines 20-136)
- Stem-bridging close (key fix for fragmentation): lines 92-100

**Figures:** `03_segmentation/` (`*_segments_compare.png` boxes, `*_cleaned.png` staff-removed binaries).

### Stage 4 — Symbol classification (CNN)

A 4-block convolutional network (`SymbolCNN`) classifies 64x64 binary crops into
a 9-class vocabulary (note/rest whole-half-quarter-eighth, plus "other").

**Training data via "strict-discard" pairing.** PrIMuS provides clean rendered
staves with a semantic token list. The hard problem is pairing each segmented
crop with the correct label. Naive positional pairing drifts because beamed notes
merge into one blob, stems fragment, and non-note symbols (rests, accidentals,
barlines) look note-sized. The solution: extract note tokens only, segment blobs,
filter to note-shaped components, and **if the filtered blob count does not
exactly equal the note-token count, discard the entire stave** rather than pair
mismatched items. This trades quantity for label correctness.

Over the full 87,678-stave corpus this kept 2,920 staves (3.3%) yielding 35,933
clean crops. A diagnostic confirmed this strictness is necessary: only ~3% of
staves have blob counts that match any token sequence, because connected-
component segmentation simultaneously merges, fragments, and mixes in non-note
symbols.

**Class-weighting bug fix.** The first full-corpus training run achieved 0%
accuracy for all 30 epochs. Root cause: the five empty classes (no training
samples) received weights ~133x larger than the dominant note class via a
`counts.get(i, 1)` fallback, so the optimizer pushed all logits toward classes
that never appear as targets. Fix: compute sqrt-inverse weights over populated
classes only, normalize among them, and assign empty classes a neutral 1.0. After
the fix the model learned immediately (55.6% val accuracy at epoch 1).

**Code:**
- Architecture: `symbol_classifier.py`, `SymbolCNN` (lines 69-131), vocabulary
  and `semantic_token_to_class` (lines 19-62)
- Strict-discard crop extraction: `train_classifier.py`, `_extract_crops_from_stave`
  (lines 182-296), strict count gate (lines 298-307)
- Full-corpus multi-package walk: `build_manifest` (lines 321-430)
- Fixed class weighting: `train` (lines 594-612)
- Augmentation (affine, blur, morph, noise, erasing): `_train_transform` (lines 509-540)
- Domain-transfer test: `test_on_local` (lines 716-790)

**Figures:** `04_cnn_dataset/crop_montage_primus.png` vs `crop_montage_local.png`
(this single comparison visually *is* the domain gap); label-alignment audits in
`07_progress_and_diagnostics/crop_label_audit/`.

### Stage 5 — Pitch inference and MIDI export

Pitch is geometric, not learned: the notehead y-coordinate (found from the
horizontal pixel-density profile of the crop, which handles stems-up and
stems-down) is converted to a staff position in half-spaces relative to the
bottom line, then mapped to a treble-clef MIDI note and pitch string. Duration
comes from the CNN class. The (pitch, duration) sequence is written to a type-0
MIDI file with `mido`; rests accumulate silence.

**Code:** `pitch_midi.py`
- Notehead localization: `_find_notehead_y` (lines 128-161)
- Staff-position -> MIDI: `_staff_position_to_midi` (lines 63-88)
- Pitch inference: `infer_pitch` (lines 164-215)
- MIDI writer: `write_midi` (lines 252-322)
- Pipeline integration: `build_midi_sequence` (lines 329-396)

**Figures:** `05_pitch_and_midi/` (`*_pitch_eval.png` — green/red correctness
overlay plus ground-truth vs inferred pitch contour).

### Pipeline glue

`sheet_utils.py` ties the stages together (`run_pipeline`, lines 175-202),
parses ground truth (`parse_ground_truth`, lines 73-112), and filters note-like
components (`filter_note_components`, lines 119-168).

---

## 3. Results

### Rectification (5/5 PASS)
All five photos rectify correctly with the new cascade. Staff slant < 0.25 deg,
border-background fraction < 1%. The hardest case (londonBridgeIsFalling, steep
angle + deep corner shadow) rectifies to 0.0 deg slant; its remaining limitation
is physical (one page corner is lost in shadow and not captured on sensor), not
algorithmic.

### Staff detection and spacing

| Image | Staves detected | Mean spacing |
| --- | --- | --- |
| yankeeDoodle | 4 | 15.3 px |
| twinkleTwinkleLittleStar | 3 | 11.0 px |
| maryHadLittleLamb | 2 | 11.8 px |
| cs131 | 3 | 6.5 px |
| londonBridgeIsFalling | 1 (relaxed params) | — |

The straighter pages from the rectification fix let Hough recover cs131's
previously-missed third stave.

### CNN (the domain gap — the central result)

| Setting | Accuracy |
| --- | --- |
| PrIMuS validation (clean -> clean) | 58.9% |
| Local phone-photo test (clean -> real) | 22.7% |
| **Domain gap** | **-36.2 pp** |

Per-class transfer: note_eighth transfers best (80% local) because its flag shape
survives noise; note_half worst (0% local) because its hollow notehead — the only
feature distinguishing it from a quarter — is destroyed by binarization. The
"other" class (21 local crops, 0%) drags local accuracy down because the strict
gate produced zero "other" training examples; excluding it, note-only local
accuracy is 22/76 = 28.9%.

### End-to-end pitch (per image)

yankeeDoodle is the strongest: 28 notes compared, 14% exact, 50% within +-1
diatonic step. Accuracy is bounded by segmentation (beamed groups merging,
over-filtering), not by rectification.

---

## 4. Diagnostics and dead-ends investigated (rigor evidence)

These show the project explored alternatives rigorously rather than reporting
only the happy path. Figures live in `07_progress_and_diagnostics/`.

- **Beam-splitting investigation** (`beam_splitting_investigation/`): tested
  whether splitting wide merged note-blobs at stem positions would recover
  discarded staves. Vertical projection profiles are clean on textbook PrIMuS
  eighth groups but sparse/edge-only on phone photos. Quantified result:
  splitting fixes < 1% of count-mismatched staves, so it was not worth building.
- **Fragmentation investigation** (`fragmentation_investigation/`): annotated
  staves proving the dominant "too many blobs" problem is non-note symbols
  (rests, accidentals, digits, barlines) plus simultaneous merge/fragment noise,
  not pure beam-merging.
- **Crop-label audit** (`crop_label_audit/`): contact sheets that exposed the
  positional-pairing drift and motivated the strict-discard approach.

---

## 5. Updated timeline and plan

Completed:
- Stage 1 rectification (overhauled, robust to angle/lighting) — done.
- Stage 2 staff detection — done.
- Stage 3 staff removal + segmentation — done.
- Stage 4 CNN trained on 35,933 clean PrIMuS crops — done (58.9% val).
- Stage 5 pitch inference + MIDI export — done.
- Domain-transfer evaluation (the proposal's key measurement) — done (22.7%).

Remaining before the final report (due Sat 6/6):
- Write the 4-page CVPR-format report using the numbers and figures here.
- Produce one combined five-stage pipeline figure for the overview.
- Optional stretch: segmentation improvements to lift end-to-end pitch accuracy;
  Camera-PrIMuS augmentation to narrow the domain gap.

---

## 6. Where to find everything (code map)

| Stage / concern | File | Key symbols (lines) |
| --- | --- | --- |
| Rectification | `rectify.py` | `_find_page_corners` (321), `_corners_via_hough` (199), `_clahe_equalize` (94), `rectify_page` (390) |
| Staff detection | `detect_staves.py` | `detect_staves` (37), `_cluster_ys` (23) |
| Staff removal + segment | `segment_symbols.py` | `remove_staves_and_segment` (20) |
| CNN architecture | `symbol_classifier.py` | `SymbolCNN` (69), `semantic_token_to_class` (39) |
| Dataset + training | `train_classifier.py` | `_extract_crops_from_stave` (182), `build_manifest` (321), `train` (551), `test_on_local` (716) |
| Pitch + MIDI | `pitch_midi.py` | `infer_pitch` (164), `_staff_position_to_midi` (63), `write_midi` (252) |
| Pipeline glue | `sheet_utils.py` | `run_pipeline` (175), `parse_ground_truth` (73) |
| End-to-end test | `test_pitch_midi.py` | whole-file harness |

**Reproduce key results:**
- Rectification figures: `python3 test_rectify.py`
- End-to-end pitch + MIDI: `python3 test_pitch_midi.py`
- Build PrIMuS crops: `python3 train_classifier.py --build --primusdir . --max_samples 90000`
- Train CNN: `python3 train_classifier.py --epochs 30 --batch 64`
- Domain-transfer test: `python3 train_classifier.py --buildlocal && python3 train_classifier.py --testlocal`
