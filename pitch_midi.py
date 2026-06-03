"""
pitch_midi.py
two public functions:

    infer_pitch(bbox, staff_ys, staff_spacing, symbol_class)
        given a bounding box on the rectified image, the 5 staff-line
        y-coordinates of the containing stave, the staff spacing, and the
        predicted symbol class, return the pitch string ("C4", "G5", ...)
        or None if the symbol is a rest or non-pitched "other".

    write_midi(sequence, output_path, tempo_bpm, ticks_per_beat)
        given a list of (pitch, duration) tuples — where pitch is a string
        like "C4" or None for a rest, and duration is one of
        "whole"/"half"/"quarter"/"eighth" — write a standard MIDI file.

helper / integration utilities:

    pitch_str_to_midi(pitch_str) -> int
        convert "C4" -> 60, "G5" -> 79, etc.

    build_midi_sequence(symbols, predictions, staves, staff_spacing)
        convenience wrapper: takes the raw outputs of remove_staves_and_segment
        and the CNN prediction list and returns a (pitch, duration) sequence
        ready for write_midi.
"""

from __future__ import annotations

import math
from typing import Optional

from mido import MidiFile, MidiTrack, Message, MetaMessage


# ---------------------------------------------------------------------------
# treble-clef pitch constants
# ---------------------------------------------------------------------------

# diatonic note names in scale order
_NOTE_NAMES = ["C", "D", "E", "F", "G", "A", "B"]

# semitone offsets within an octave (C=0)
_SEMITONES = [0, 2, 4, 5, 7, 9, 11]

# in treble clef the bottom staff line is E4
# E is at diatonic index 2 in [C, D, E, F, G, A, B]
_BOTTOM_LINE_DIATONIC_IDX = 2   # E
_BOTTOM_LINE_OCTAVE       = 4   # E4

# duration name -> number of quarter-note beats
_DURATION_BEATS: dict[str, float] = {
    "whole":   4.0,
    "half":    2.0,
    "quarter": 1.0,
    "eighth":  0.5,
}


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _staff_position_to_midi(staff_pos: int) -> int:
    """
    convert a staff position (integer half-spaces from the bottom line of a
    treble-clef stave) to a MIDI note number.

    reference points:
        staff_pos =  0  ->  E4  (bottom line)   MIDI 64
        staff_pos =  2  ->  G4  (2nd line)       MIDI 67
        staff_pos =  4  ->  B4  (middle line)    MIDI 71
        staff_pos =  6  ->  D5  (4th line)       MIDI 74
        staff_pos =  8  ->  F5  (top line)       MIDI 77
        staff_pos = -1  ->  D4  (space below)    MIDI 62
        staff_pos = -2  ->  C4  (ledger line)    MIDI 60

    each step of 1 in staff_pos is one diatonic step (line or space).
    sharps/flats are not computed here — we have no key-signature context.
    """
    total_diatonic = _BOTTOM_LINE_DIATONIC_IDX + staff_pos

    # python's floor division handles negative total_diatonic correctly
    octave    = _BOTTOM_LINE_OCTAVE + math.floor(total_diatonic / 7)
    note_idx  = total_diatonic % 7          # always in [0, 6]

    # MIDI note: octave * 12 + 12 accounts for the convention that
    # C-1 = 0, C0 = 12, C4 = 60
    return octave * 12 + 12 + _SEMITONES[note_idx]


def _midi_to_pitch_str(midi_note: int) -> str:
    """convert MIDI note number to human-readable pitch string."""
    octave   = (midi_note - 12) // 12
    note_idx = (midi_note - 12) % 12
    # find diatonic name (ignores sharps/flats — uses nearest natural)
    closest = min(range(7), key=lambda i: abs(_SEMITONES[i] - note_idx))
    return f"{_NOTE_NAMES[closest]}{octave}"


def _notehead_y_to_staff_position(
    notehead_y: float,
    staff_ys: list[float],
    staff_spacing: float,
) -> int:
    """
    map a notehead centre y-coordinate to an integer staff position.

    the staff is described by 5 y-coordinates sorted top-to-bottom
    (staff_ys[0] is the top line, staff_ys[4] is the bottom line).
    y increases downward, so the bottom line has the highest y value and
    corresponds to staff_pos = 0 (E4 in treble clef).

    each half-space (one diatonic step) equals staff_spacing / 2 pixels.
    we round to the nearest integer to snap to a line or space.
    """
    bottom_line_y = staff_ys[4]
    half_space    = staff_spacing / 2.0

    # positive offset = above bottom line = higher pitch
    pixel_offset = bottom_line_y - notehead_y
    return round(pixel_offset / half_space)


# ---------------------------------------------------------------------------
# public: pitch inference
# ---------------------------------------------------------------------------

def _find_notehead_y(
    bbox: tuple[int, int, int, int],
    crop: "np.ndarray | None",
) -> float:
    """
    estimate the y-coordinate of the notehead centre on the full image.

    strategy: look at the horizontal pixel-density profile of the crop.
    noteheads are roughly oval — their rows have more white pixels than
    stem rows.  we find the row with peak density in the central 70% of
    the crop height (to exclude beam rows at the very top or bottom) and
    use that row's absolute y as the notehead centre.

    falls back to the bbox vertical centre when no crop is available or
    the crop contains no useful signal.
    """
    import numpy as np

    x, y, w, h = bbox
    default_cy = y + h * 0.5   # fallback: bbox centre

    if crop is None or crop.size == 0 or h < 4:
        return default_cy

    # restrict search to the middle 70% of the height to avoid beam rows
    margin   = max(1, int(h * 0.15))
    row_sums = crop[margin: h - margin].sum(axis=1).astype(float)

    if row_sums.max() == 0:
        return default_cy

    # find the densest row within the search window
    peak_row = int(row_sums.argmax()) + margin   # relative to crop top
    return float(y + peak_row)


def infer_pitch(
    bbox: tuple[int, int, int, int],
    staff_ys: list[float],
    staff_spacing: float,
    symbol_class: str,
    crop: "np.ndarray | None" = None,
) -> Optional[str]:
    """
    infer the pitch of a musical symbol given its bounding box and stave context.

    parameters
    ----------
    bbox : (x, y, w, h)
        bounding box in pixel coordinates on the rectified image, as returned
        by remove_staves_and_segment.
    staff_ys : list of 5 floats
        y-coordinates of the five staff lines of the containing stave, sorted
        top-to-bottom (increasing y), as returned by detect_staves.
    staff_spacing : float
        mean vertical distance between adjacent staff lines (pixels).
    symbol_class : str
        predicted class from SymbolCNN, e.g. "note_quarter", "rest_half",
        "other".
    crop : np.ndarray or None
        optional grayscale crop of the symbol from the staff-removed binary
        (white symbols on black background).  when provided, the notehead
        y-coordinate is estimated from the horizontal pixel-density profile,
        which handles both stems-up and stems-down orientations correctly.
        when None, falls back to the vertical centre of the bbox.

    returns
    -------
    str or None
        pitch string like "C4", "G5", "E3", using treble-clef rules.
        returns None for rests and "other" symbols (no pitch to assign).

    notes
    -----
    - assumes treble clef; bass/alto clef support would require a different
      _BOTTOM_LINE_* offset.
    - sharps and flats are not inferred (no key-signature context); all
      pitches are natural notes.
    - ledger lines are extrapolated automatically beyond the staff.
    """
    if not symbol_class.startswith("note_"):
        return None

    notehead_cy = _find_notehead_y(bbox, crop)

    staff_pos = _notehead_y_to_staff_position(notehead_cy, staff_ys, staff_spacing)
    midi_note = _staff_position_to_midi(staff_pos)
    return _midi_to_pitch_str(midi_note)


# ---------------------------------------------------------------------------
# public: pitch string utilities
# ---------------------------------------------------------------------------

def pitch_str_to_midi(pitch_str: str) -> int:
    """
    convert a pitch string to a MIDI note number.

    examples:
        "C4" -> 60  (middle C)
        "A4" -> 69  (concert A)
        "G5" -> 79
        "E3" -> 52

    only natural notes are supported; sharps/flats are not parsed.
    raises ValueError for unrecognised strings.
    """
    if len(pitch_str) < 2:
        raise ValueError(f"unrecognised pitch string: {pitch_str!r}")
    note_name = pitch_str[:-1].upper()
    try:
        octave = int(pitch_str[-1])
    except ValueError:
        raise ValueError(f"unrecognised pitch string: {pitch_str!r}")
    if note_name not in _NOTE_NAMES:
        raise ValueError(f"unrecognised note name: {note_name!r}")
    note_idx = _NOTE_NAMES.index(note_name)
    return octave * 12 + 12 + _SEMITONES[note_idx]


# ---------------------------------------------------------------------------
# public: MIDI output
# ---------------------------------------------------------------------------

def write_midi(
    sequence: list[tuple[Optional[str], str]],
    output_path: str,
    tempo_bpm: int = 120,
    ticks_per_beat: int = 480,
) -> None:
    """
    write a monophonic MIDI file from a sequence of (pitch, duration) tuples.

    parameters
    ----------
    sequence : list of (pitch, duration)
        pitch is a string like "C4", "G5", or None for a rest.
        duration is one of "whole", "half", "quarter", "eighth".
        unknown duration strings fall back to "quarter".
    output_path : str
        destination .mid file path.
    tempo_bpm : int
        playback tempo in beats per minute (default 120).
    ticks_per_beat : int
        MIDI tick resolution (default 480, standard for DAW compatibility).

    notes
    -----
    - dotted notes, tied notes, and dynamics are not modelled.
    - all notes play at velocity 64 (mezzo-forte).
    - the file is type-0 (single track), which is the most universally
      supported MIDI format.
    """
    mid   = MidiFile(type=0, ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)

    tempo_us = int(60_000_000 / tempo_bpm)
    track.append(MetaMessage("set_tempo", tempo=tempo_us, time=0))
    track.append(MetaMessage("time_signature",
                             numerator=4, denominator=4,
                             clocks_per_click=24, notated_32nd_notes_per_beat=8,
                             time=0))

    # accumulated silence before the next note_on (used for rests)
    pending_ticks = 0

    for pitch, duration in sequence:
        beats  = _DURATION_BEATS.get(duration, 1.0)
        ticks  = int(round(beats * ticks_per_beat))

        if pitch is None:
            # rest: just accumulate silence; no note_on/off messages needed
            pending_ticks += ticks
            continue

        try:
            midi_note = pitch_str_to_midi(pitch)
        except ValueError:
            pending_ticks += ticks
            continue

        # clamp to valid MIDI range
        midi_note = max(0, min(127, midi_note))

        track.append(Message("note_on",  note=midi_note, velocity=64,
                             time=pending_ticks))
        track.append(Message("note_off", note=midi_note, velocity=0,
                             time=ticks))
        pending_ticks = 0

    # end-of-track
    track.append(MetaMessage("end_of_track", time=0))

    mid.save(output_path)


# ---------------------------------------------------------------------------
# integration helper: tie the pipeline stages together
# ---------------------------------------------------------------------------

def build_midi_sequence(
    symbols: list[tuple[tuple[int, int, int, int], object]],
    predictions: list[str],
    staves: list[list[float]],
    staff_spacing: float,
) -> list[tuple[Optional[str], str]]:
    """
    convert pipeline outputs into a (pitch, duration) sequence for write_midi.

    parameters
    ----------
    symbols : list of (bbox, crop)
        as returned by remove_staves_and_segment — sorted left-to-right,
        top-to-bottom.
    predictions : list of str
        CNN class prediction for each entry in symbols (same order).
    staves : list of list of 5 floats
        staff-line y-coordinates from detect_staves.
    staff_spacing : float
        mean inter-line spacing from detect_staves.

    returns
    -------
    list of (pitch_str | None, duration_str)
        ready to pass directly to write_midi.
        "other" symbols (clefs, barlines, etc.) are silently skipped.

    algorithm
    ---------
    for each symbol:
    1. assign it to the stave whose vertical centre is nearest to the
       symbol's notehead y-coordinate.
    2. call infer_pitch with that stave's y-coordinates.
    3. extract the duration from the class string ("note_quarter" -> "quarter").
    """
    if not staves:
        return []

    # precompute stave vertical centres for nearest-stave lookup
    stave_centres = [(sum(ys) / len(ys)) for ys in staves]

    sequence: list[tuple[Optional[str], str]] = []

    for (bbox, _crop), cls in zip(symbols, predictions):
        if cls == "other":
            continue

        parts = cls.split("_", 1)
        if len(parts) != 2 or parts[1] not in _DURATION_BEATS:
            continue

        kind, duration = parts   # kind in {"note", "rest"}

        if kind == "rest":
            sequence.append((None, duration))
            continue

        # find nearest stave by comparing symbol y-centre to stave centres
        x, y, w, h = bbox
        sym_y = y + h * 0.60
        nearest_idx = min(range(len(stave_centres)),
                          key=lambda i: abs(stave_centres[i] - sym_y))
        staff_ys = staves[nearest_idx]

        pitch = infer_pitch(bbox, staff_ys, staff_spacing, cls, crop=_crop)
        sequence.append((pitch, duration))

    return sequence
