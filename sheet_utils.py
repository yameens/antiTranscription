"""
sheet_utils.py
shared utilities for the sheet-music pipeline that do not depend on PyTorch
or mido.  imported by both train_classifier.py and test_pitch_midi.py.

contents
--------
LOCAL_IMAGES              list of (image_stem, gt_filename) pairs
NOTE_DIR / SHEET_DIR      canonical path constants
_DEFAULT_STAVE_PARAMS     hough parameters for detect_staves
_RELAXED_STAVE_PARAMS     fallback hough parameters

parse_ground_truth(txt_path)        -> list[(pitch|None, duration)]
filter_note_components(...)         -> filtered symbol list
run_pipeline(jpg_path)              -> (rectified, staves, spacing, symbols, cleaned)
"""

from __future__ import annotations

import os
import warnings

import cv2
import numpy as np

from rectify import rectify_page
from detect_staves import detect_staves
from segment_symbols import remove_staves_and_segment


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
BASE      = os.path.dirname(os.path.abspath(__file__))
SHEET_DIR = os.path.join(BASE, "sheet music")
NOTE_DIR  = os.path.join(SHEET_DIR, "sheetNote")

# images and their matching ground-truth filenames
# (marryHadLittleLamb.txt has the "marry" typo in the filename — kept as-is)
LOCAL_IMAGES: list[tuple[str, str]] = [
    ("yankeeDoodle",             "yankeeDoodle.txt"),
    ("twinkleTwinkleLittleStar", "twinkleTwinkleLittleStar.txt"),
    ("maryHadLittleLamb",        "marryHadLittleLamb.txt"),
    ("cs131",                    "cs131.txt"),
]

# stave detection parameters
_DEFAULT_STAVE_PARAMS = dict(
    hough_threshold=100, min_line_length=300, max_line_gap=40,
    angle_tolerance_deg=2.0, cluster_tol=3, morph_close_width=0,
)
_RELAXED_STAVE_PARAMS = dict(
    hough_threshold=60, min_line_length=150, max_line_gap=60,
    angle_tolerance_deg=3.0, cluster_tol=5, morph_close_width=60,
)


# ---------------------------------------------------------------------------
# ground-truth parsing
# ---------------------------------------------------------------------------

_DURATION_ALIASES: dict[str, str] = {
    "whole":   "whole",
    "half":    "half",
    "quarter": "quarter",
    "eighth":  "eighth",
    "eigth":   "eighth",   # typo in londonBridgeIsFalling.txt
    "eight":   "eighth",   # typo in marryHadLittleLamb.txt
}


def parse_ground_truth(txt_path: str) -> list[tuple[str | None, str]]:
    """
    parse a sheetNote .txt file into an ordered list of (pitch, duration).

    pitch is a string like "C4", "G5", or None for rests.
    duration is one of "whole" / "half" / "quarter" / "eighth".

    handled variants:
        "B4 quarter"    -> ("B4", "quarter")
        "C5 eigth"      -> ("C5", "eighth")   [spelling normalised]
        "quarterRest"   -> (None, "quarter")
        blank lines     -> skipped
    """
    sequence: list[tuple[str | None, str]] = []
    with open(txt_path, "r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            # rest shorthand: "quarterRest", "halfRest", etc.
            lower = line.lower()
            if lower.endswith("rest"):
                raw_dur = lower[: lower.index("rest")]
                dur = _DURATION_ALIASES.get(raw_dur)
                if dur:
                    sequence.append((None, dur))
                continue

            parts = line.split()
            if len(parts) != 2:
                continue
            pitch_tok, dur_tok = parts
            dur = _DURATION_ALIASES.get(dur_tok.lower())
            if dur is None:
                continue
            if len(pitch_tok) >= 2 and pitch_tok[0].upper() in "ABCDEFG":
                sequence.append((pitch_tok, dur))

    return sequence


# ---------------------------------------------------------------------------
# note-like component filter
# ---------------------------------------------------------------------------

def filter_note_components(
    symbols: list[tuple[tuple[int, int, int, int], np.ndarray]],
    staff_spacing: float,
    image_width: int,
    staves: list[list[float]] | None = None,
    clef_fraction: float = 0.09,
) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    """
    keep only components whose shape is consistent with a printed note.

    rejects:
    - components in the leftmost clef/time-sig region (x < clef_fraction * w)
    - barlines: very tall, narrow  (aspect ratio w/h < 0.08)
    - page-spanning blobs          (w > 0.5 * image_width)
    - too small to be a note head  (h < 0.8 * staff_spacing)
    - taller than a full bar       (h > 5 * staff_spacing)
    - components outside the stave vertical range when staves are provided
      (y_centre must be within 4 * staff_spacing of at least one stave centre)
    """
    clef_x_limit = clef_fraction * image_width

    # precompute stave vertical bounds if provided
    stave_bounds: list[tuple[float, float]] = []
    if staves:
        margin = 4.0 * staff_spacing
        for stave in staves:
            top    = min(stave) - margin
            bottom = max(stave) + margin
            stave_bounds.append((top, bottom))

    kept = []
    for bbox, crop in symbols:
        x, y, w, h = bbox
        if x < clef_x_limit:
            continue
        if w > 0.5 * image_width:
            continue
        # raised minimum from 0.4 to 0.8 — whole notes are roughly 0.8-1×
        # spacing tall; anything smaller is noise or a serif speck
        if h < 0.8 * staff_spacing or h > 5.0 * staff_spacing:
            continue
        if w == 0 or (w / h) < 0.08:
            continue
        # y-range gate: component centre must fall within one stave's range
        if stave_bounds:
            cy = y + h / 2.0
            if not any(top <= cy <= bottom for top, bottom in stave_bounds):
                continue
        kept.append((bbox, crop))
    return kept


# ---------------------------------------------------------------------------
# pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(jpg_path: str):
    """
    run rectify -> detect_staves -> remove_staves_and_segment on one image.

    returns (rectified, staves, spacing, symbols, cleaned).
    tries default stave params first; falls back to relaxed on failure.
    raises RuntimeError if both parameter sets fail.
    """
    rectified = rectify_page(jpg_path)
    staves, spacing = None, None

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        try:
            staves, spacing = detect_staves(rectified, **_DEFAULT_STAVE_PARAMS)
        except ValueError:
            pass

    if staves is None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                staves, spacing = detect_staves(rectified, **_RELAXED_STAVE_PARAMS)
            except ValueError as exc:
                raise RuntimeError(f"stave detection failed: {exc}") from exc

    symbols, cleaned = remove_staves_and_segment(rectified, staves, spacing)
    return rectified, staves, spacing, symbols, cleaned
