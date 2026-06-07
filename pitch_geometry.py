"""
pitch_geometry.py
geometry-only pitch reader for a rectified treble-clef page.

motivation
----------
pitch is a purely *vertical* quantity: a notehead's centre y-coordinate
relative to the five staff lines fully determines its letter name and octave.
duration, stems, beams and staff-line removal are irrelevant to pitch.  so
instead of leaning on the (lossy) staff-removal + connected-component
segmentation, this module finds notehead centres directly with template
matching and reads the pitch straight off the staff geometry.

pipeline
--------
1. rectify the page and detect the staves (reused from the main pipeline).
2. binarize, then fill enclosed holes so that *hollow* half-note heads become
   solid ellipses identical in shape to *filled* quarter-note heads.
3. correlate a synthetic filled-ellipse template (sized from the measured
   staff spacing) against the page, restricted to each stave's vertical band.
4. greedily pick correlation peaks with non-maximum suppression -> notehead
   centres (cx, cy).
5. map each centre's y to an integer staff position and then to a pitch using
   treble-clef rules (bottom line = E4).

the public entry point is `read_pitches(jpg_path)`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

from rectify import rectify_page
from detect_staves import detect_staves
from pitch_midi import _staff_position_to_midi, _midi_to_pitch_str


# ---------------------------------------------------------------------------
# stave detection parameters (same as the rest of the pipeline)
# ---------------------------------------------------------------------------
_DEFAULT_STAVE_PARAMS = dict(
    hough_threshold=100, min_line_length=300, max_line_gap=40,
    angle_tolerance_deg=2.0, cluster_tol=3, morph_close_width=0,
)


@dataclass
class Notehead:
    cx: float          # centre x on the rectified page
    cy: float          # centre y on the rectified page
    stave_idx: int     # which stave it was assigned to
    staff_pos: int     # half-steps above the bottom line (E4 = 0)
    pitch: str         # e.g. "C5"
    score: float       # template-match correlation


# ---------------------------------------------------------------------------
# binary preparation
# ---------------------------------------------------------------------------

def _binarize_filled(gray: np.ndarray, spacing: float) -> np.ndarray:
    """
    adaptive-threshold the page (ink -> white) and fill the small hollow centre
    of half-note heads so they look identical to filled quarter-note heads.

    rather than a morphological close (which would also bridge a notehead to the
    staff line running through it), we fill *enclosed* background blobs only:
    label the background, then solidify any background component that is both
    small (smaller than ~1.5x a staff cell) and does not touch the image
    border.  the hollow centre of a half note qualifies; the page background
    and the gaps between staff lines do not.  returns uint8 0/255 (ink = 255).
    """
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=25, C=10,
    )
    ink = cv2.bitwise_not(binary)          # ink = 255, paper = 0

    h_img, w_img = ink.shape
    bg = cv2.bitwise_not(ink)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bg, connectivity=8)

    out = ink.copy()
    max_hole = 1.5 * spacing * spacing
    for lbl in range(1, n):
        x = stats[lbl, cv2.CC_STAT_LEFT]
        y = stats[lbl, cv2.CC_STAT_TOP]
        w = stats[lbl, cv2.CC_STAT_WIDTH]
        h = stats[lbl, cv2.CC_STAT_HEIGHT]
        area = stats[lbl, cv2.CC_STAT_AREA]
        touches_border = (x == 0 or y == 0 or x + w >= w_img or y + h >= h_img)
        if area <= max_hole and not touches_border:
            out[labels == lbl] = 255
    return out


# ---------------------------------------------------------------------------
# notehead detection (elliptical template matching + peak picking)
# ---------------------------------------------------------------------------

def _ellipse_template(spacing: float) -> np.ndarray:
    """synthetic filled notehead: an ellipse ~1.25x spacing wide, ~1x tall."""
    nh = max(5, int(round(spacing * 1.00)))
    nw = max(6, int(round(spacing * 1.25)))
    if nh % 2 == 0:
        nh += 1
    if nw % 2 == 0:
        nw += 1
    tmpl = np.zeros((nh, nw), np.uint8)
    cv2.ellipse(tmpl, (nw // 2, nh // 2),
                (nw // 2 - 1, nh // 2 - 1), 0, 0, 360, 255, -1)
    return tmpl


def _greedy_peaks(
    resp: np.ndarray, threshold: float, min_dx: int, min_dy: int,
) -> list[tuple[int, int, float]]:
    """greedy non-maximum suppression: return (x, y, score) best-first."""
    ys, xs = np.where(resp >= threshold)
    if len(xs) == 0:
        return []
    scores = resp[ys, xs]
    order = np.argsort(scores)[::-1]
    xs, ys, scores = xs[order], ys[order], scores[order]
    kept: list[tuple[int, int, float]] = []
    for x, y, s in zip(xs, ys, scores):
        if all(abs(x - kx) >= min_dx or abs(y - ky) >= min_dy
               for kx, ky, _ in kept):
            kept.append((int(x), int(y), float(s)))
    return kept


def detect_noteheads(
    rectified: np.ndarray,
    staves: list[list[float]],
    spacing: float,
    score_threshold: float = 0.42,
    band_halfspaces: float = 6.5,
    clef_frac: float = 0.17,
    x_merge_spacings: float = 2.8,
) -> list[Notehead]:
    """
    locate notehead centres by elliptical template matching and read pitches.

    1. binarize + fill half-note holes so filled and open heads look identical.
    2. correlate a filled-ellipse template; pick peaks (one per head) inside
       each stave's vertical band, so the title/lyrics never produce hits.
    3. drop the clef (left margin) and the time signature (two vertically
       stacked blobs at the same x) — neither is a notehead.
    4. snap each centre's y to a treble-clef staff position -> pitch.
    """
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    filled_u8 = _binarize_filled(gray, spacing)
    filled = filled_u8.astype(np.float32)
    h_img, w_img = filled.shape

    # solidity gate: opening with a notehead-sized ellipse keeps only blobs as
    # big as a head and erases thin staff lines, stems, and rest glyphs.  a
    # template peak is a true notehead only if this mask has ink under it.
    ow = max(3, int(round(spacing * 0.70)))
    oh = max(3, int(round(spacing * 0.55)))
    solid = cv2.morphologyEx(
        filled_u8, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ow, oh)),
    )
    gate_r = max(2, int(round(spacing * 0.6)))

    tmpl = _ellipse_template(spacing).astype(np.float32)
    th, tw = tmpl.shape
    resp = cv2.matchTemplate(filled, tmpl, cv2.TM_CCOEFF_NORMED)
    cy_off, cx_off = th // 2, tw // 2

    half_space = spacing / 2.0
    band = band_halfspaces * half_space
    clef_x = clef_frac * w_img
    min_dx = max(3, int(round(spacing * 1.4)))
    min_dy = max(3, int(round(spacing * 1.3)))

    raw: list[Notehead] = []
    for s_idx, stave in enumerate(staves):
        top_y, bottom_y = min(stave) - band, max(stave) + band
        masked = np.full_like(resp, -1.0)
        r_lo = max(0, int(top_y - cy_off))
        r_hi = min(resp.shape[0], int(bottom_y - cy_off))
        masked[r_lo:r_hi, :] = resp[r_lo:r_hi, :]

        bottom_line_y = stave[-1]
        for px, py, score in _greedy_peaks(masked, score_threshold,
                                           min_dx, min_dy):
            cx, cy = px + cx_off, py + cy_off
            if cx < clef_x:                      # treble clef sits here
                continue
            # solidity gate: reject peaks with no notehead-sized ink under them
            # (rest glyphs, stem/ledger artefacts) — survives staff removal
            y0, y1 = max(0, cy - gate_r), min(h_img, cy + gate_r + 1)
            x0, x1 = max(0, cx - gate_r), min(w_img, cx + gate_r + 1)
            if solid[y0:y1, x0:x1].max() == 0:
                continue
            staff_pos = round((bottom_line_y - cy) / half_space)
            midi = _staff_position_to_midi(staff_pos)
            raw.append(Notehead(cx=cx, cy=cy, stave_idx=s_idx,
                                staff_pos=staff_pos,
                                pitch=_midi_to_pitch_str(midi), score=score))

    # drop time signatures: two detections at (nearly) the same x stacked
    # vertically.  a monophonic melody never has two heads at one x.
    stacked: set[int] = set()
    for i, a in enumerate(raw):
        for j, b in enumerate(raw):
            if i < j and a.stave_idx == b.stave_idx \
                    and abs(a.cx - b.cx) < 0.7 * spacing \
                    and abs(a.cy - b.cy) > 1.3 * spacing:
                stacked.add(i)
                stacked.add(j)
    kept_after_stack = [n for k, n in enumerate(raw) if k not in stacked]

    # monophonic melody: at most one notehead per horizontal position.  collapse
    # near-duplicate detections (notehead fragments, stem/ledger artefacts beside
    # a real head) by keeping the highest-scoring one within a min x-separation.
    min_sep = x_merge_spacings * spacing
    noteheads: list[Notehead] = []
    for s_idx in range(len(staves)):
        group = sorted((n for n in kept_after_stack if n.stave_idx == s_idx),
                       key=lambda n: -n.score)
        kept: list[Notehead] = []
        for n in group:
            if all(abs(n.cx - k.cx) >= min_sep for k in kept):
                kept.append(n)
        noteheads.extend(kept)

    noteheads.sort(key=lambda n: (n.stave_idx, n.cx))
    return noteheads


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

_NOTE_NAMES = ["C", "D", "E", "F", "G", "A", "B"]


def pitch_step(pitch: str | None) -> int | None:
    """diatonic step index of a pitch string ("C4"->28, "D4"->29, ...)."""
    if not pitch:
        return None
    try:
        return int(pitch[-1]) * 7 + _NOTE_NAMES.index(pitch[:-1].upper())
    except (ValueError, IndexError):
        return None


def align_pitches(
    detected: list[str], truth: list[str],
) -> list[tuple[int | None, int | None]]:
    """
    Needleman-Wunsch alignment of two pitch sequences (the standard OMR
    evaluation, robust to a single miss/insertion instead of cascading drift).
    scoring: +2 exact, +1 within one diatonic step, -1 mismatch, -1 gap.
    returns (det_idx, truth_idx) pairs; None marks an inserted/deleted slot.
    """
    n, m = len(detected), len(truth)
    score = np.zeros((n + 1, m + 1))
    for i in range(n + 1):
        score[i][0] = -i
    for j in range(m + 1):
        score[0][j] = -j

    def sub(a: str, b: str) -> int:
        if a == b:
            return 2
        sa, sb = pitch_step(a), pitch_step(b)
        if sa is not None and sb is not None and abs(sa - sb) <= 1:
            return 1
        return -1

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score[i][j] = max(score[i - 1][j - 1] + sub(detected[i - 1], truth[j - 1]),
                              score[i - 1][j] - 1,
                              score[i][j - 1] - 1)
    i, j = n, m
    pairs: list[tuple[int | None, int | None]] = []
    while i > 0 and j > 0:
        if score[i][j] == score[i - 1][j - 1] + sub(detected[i - 1], truth[j - 1]):
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif score[i][j] == score[i - 1][j] - 1:
            pairs.append((i - 1, None)); i -= 1
        else:
            pairs.append((None, j - 1)); j -= 1
    while i > 0:
        pairs.append((i - 1, None)); i -= 1
    while j > 0:
        pairs.append((None, j - 1)); j -= 1
    return pairs[::-1]


def read_pitches(jpg_path: str, **kwargs):
    """
    rectify -> detect staves -> detect noteheads -> read pitches.

    returns (rectified, staves, spacing, noteheads).
    """
    rectified = rectify_page(jpg_path)
    staves, spacing = detect_staves(rectified, **_DEFAULT_STAVE_PARAMS)
    noteheads = detect_noteheads(rectified, staves, spacing, **kwargs)
    return rectified, staves, spacing, noteheads
