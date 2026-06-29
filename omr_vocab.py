"""
omr_vocab.py
token vocabulary + scope filtering for the semantic PrIMuS encoding.

PrIMuS .semantic files are tab/whitespace-separated token streams, e.g.

    clef-G2  timeSignature-2/4  note-G4_quarter  note-C5_quarter  barline ...

token kinds seen in the corpus:
    clef-<X>                e.g. clef-G2 (treble), clef-C1, clef-F4
    keySignature-<X>        e.g. keySignature-FM, keySignature-CM
    timeSignature-<X>       e.g. timeSignature-2/4, timeSignature-C
    note-<pitch><oct>_<dur> e.g. note-C5_half, note-A4_half.  (trailing '.' = dotted)
    gracenote-<...>         small ornament notes
    rest-<dur>              e.g. rest-quarter
    multirest-<n>
    barline
    tie / fermata / ...     occasional articulations

scope (matches the project: monophonic treble, no accidentals/key signatures,
durations whole through eighth).  the filters here decide which staves are kept
for training so the model's domain matches the phone-photo test set.  every
token of a *kept* staff goes into the vocabulary, so CTC targets are always
in-vocab.
"""

from __future__ import annotations

import json
import os

# durations in project scope (plus their dotted variants, handled separately)
_SCOPE_DURATIONS = {"whole", "half", "quarter", "eighth"}

# pitch letters that carry no accidental
_NATURAL_PITCHES = set("ABCDEFG")


def tokenize(semantic_text: str) -> list[str]:
    """split a .semantic file's contents into tokens (tabs or spaces)."""
    return semantic_text.split()


def _note_pitch_and_dur(tok: str) -> tuple[str, str] | None:
    """
    for a 'note-...' or 'gracenote-...' token return (pitch_with_accidental, dur)
    where dur has any trailing '.' stripped.  returns None if unparseable.
        note-A4_half.  -> ('A4', 'half')
        note-Bb4_half  -> ('Bb4', 'half')
    """
    body = tok.split("-", 1)[1] if "-" in tok else tok
    if "_" not in body:
        return None
    pitch, dur = body.split("_", 1)
    dur = dur.rstrip(".")
    return pitch, dur


def _pitch_is_natural(pitch: str) -> bool:
    """'C5' natural; 'Bb4' / 'C#5' have an accidental."""
    if not pitch:
        return False
    if pitch[0] not in _NATURAL_PITCHES:
        return False
    # natural pitch is exactly letter + octave digits (no '#'/'b')
    return all(c.isdigit() for c in pitch[1:])


def in_scope(
    tokens: list[str],
    *,
    require_clef: str = "clef-G2",
    no_keysig_accidentals: bool = True,
    no_note_accidentals: bool = True,
    allowed_durations: set[str] | None = None,
    allow_dotted: bool = True,
    drop_gracenotes: bool = True,
) -> bool:
    """
    decide whether a staff's token stream falls within the project scope.
    returns True to keep the staff.
    """
    if allowed_durations is None:
        allowed_durations = _SCOPE_DURATIONS

    if not tokens:
        return False

    # exactly one clef and it must be the required one (treble)
    clefs = [t for t in tokens if t.startswith("clef-")]
    if require_clef is not None and (len(clefs) != 1 or clefs[0] != require_clef):
        return False

    for t in tokens:
        if t.startswith("keySignature-"):
            if no_keysig_accidentals and t not in ("keySignature-CM", "keySignature-Am"):
                return False
        elif t.startswith("gracenote-"):
            if drop_gracenotes:
                return False
        elif t.startswith("note-"):
            pd = _note_pitch_and_dur(t)
            if pd is None:
                return False
            pitch, dur = pd
            if dur not in allowed_durations:
                return False
            if (not allow_dotted) and t.endswith("."):
                return False
            if no_note_accidentals and not _pitch_is_natural(pitch):
                return False
        elif t.startswith("rest-"):
            dur = t.split("-", 1)[1].rstrip(".")
            if dur not in allowed_durations:
                return False
        elif t.startswith("multirest-"):
            return False  # out of scope
    return True


class Vocab:
    """bidirectional token <-> id map.  id range [0, n_classes); blank is n_classes."""

    def __init__(self, tokens: list[str]):
        # deterministic ordering for reproducibility
        self.itos = sorted(set(tokens))
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @property
    def n_classes(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi[t] for t in tokens if t in self.stoi]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.itos[i] for i in ids if 0 <= i < len(self.itos)]

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump({"itos": self.itos}, fh)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path) as fh:
            data = json.load(fh)
        v = cls.__new__(cls)
        v.itos = data["itos"]
        v.stoi = {t: i for i, t in enumerate(v.itos)}
        return v


def build_vocab(token_lists: list[list[str]]) -> Vocab:
    """build a Vocab covering every token that appears in the kept staves."""
    all_tokens: list[str] = []
    for toks in token_lists:
        all_tokens.extend(toks)
    return Vocab(all_tokens)
