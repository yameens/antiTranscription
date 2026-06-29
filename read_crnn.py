"""
read_crnn.py
phone-photo -> MIDI using the end-to-end CRNN (no segmentation, no classifier).

flow
----
1. rectify the page                          (rectify.rectify_page)
2. detect the staves                          (detect_staves.detect_staves)
3. for each stave, crop a horizontal band whose staff-to-band height ratio
   matches PrIMuS (~0.54), grayscale it, feed it to the CRNN.
4. greedy-CTC-decode each band to a semantic token sequence.
5. concatenate staves top-to-bottom -> one token stream for the page.
6. convert note/rest tokens to (pitch, duration) and write MIDI.

the crop proportion matters: the model was trained on PrIMuS staves that fill
~54% of the image height, so we reproduce that framing rather than a tight or
loose crop (a scale mismatch would hurt the conv features).
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import torch

from rectify import rectify_page
from detect_staves import detect_staves
from crnn_omr import CRNN
from omr_vocab import Vocab
from omr_dataset import prep_staff_image, IMG_HEIGHT
from crnn_decode import greedy_decode

# PrIMuS staff occupies ~0.54 of the image height -> margin ~1.7 * spacing each side
_BAND_MARGIN_SPACINGS = 1.7

_DEFAULT_STAVE_PARAMS = dict(
    hough_threshold=100, min_line_length=300, max_line_gap=40,
    angle_tolerance_deg=2.0, cluster_tol=3, morph_close_width=0,
)


def load_crnn(checkpoint: str, device: str = "cpu"):
    """load a CRNN checkpoint -> (model, vocab, height, torch.device)."""
    dev = torch.device(device if device != "auto" else
                       ("mps" if torch.backends.mps.is_available() else "cpu"))
    ckpt = torch.load(checkpoint, map_location="cpu")
    vocab = Vocab.__new__(Vocab)
    vocab.itos = ckpt["itos"]
    vocab.stoi = {t: i for i, t in enumerate(vocab.itos)}
    height = ckpt.get("height", IMG_HEIGHT)
    model = CRNN(vocab.n_classes, img_height=height)
    model.load_state_dict(ckpt["model_state"])
    model.to(dev).eval()
    return model, vocab, height, dev


def _crop_stave_band(gray_page: np.ndarray, stave: list[float],
                     spacing: float) -> np.ndarray:
    """crop the horizontal band around one stave, matching PrIMuS framing."""
    h = gray_page.shape[0]
    margin = _BAND_MARGIN_SPACINGS * spacing
    top = max(0, int(min(stave) - margin))
    bot = min(h, int(max(stave) + margin))
    return gray_page[top:bot, :]


@torch.no_grad()
def read_score(jpg_path: str, model, vocab, height, device,
               stave_params: dict | None = None) -> list[list[str]]:
    """
    returns one decoded token list per detected stave (reading order).
    """
    stave_params = stave_params or _DEFAULT_STAVE_PARAMS
    rectified = rectify_page(jpg_path)
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    staves, spacing = detect_staves(rectified, **stave_params)

    per_stave_tokens: list[list[str]] = []
    for stave in staves:
        band = _crop_stave_band(gray, stave, spacing)
        if band.shape[0] < 8 or band.shape[1] < 16:
            per_stave_tokens.append([])
            continue
        img = prep_staff_image(band, height)              # (H, W) float32
        x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
        logp = model(x).cpu()                             # (T,1,C)
        il = torch.tensor([img.shape[1] // 4], dtype=torch.long)
        ids = greedy_decode(logp, il, model.blank)[0]
        per_stave_tokens.append(vocab.decode(ids))
    return per_stave_tokens


def tokens_to_sequence(tokens: list[str]) -> list[tuple[str | None, str]]:
    """
    convert semantic tokens to (pitch, duration) for write_midi.
    notes -> (pitch, duration); rests -> (None, duration); clef/key/time/
    barline/tie are skipped.  dotted durations collapse to the base duration
    (write_midi does not model dots).
    """
    seq: list[tuple[str | None, str]] = []
    for t in tokens:
        if t.startswith("note-"):
            body = t.split("-", 1)[1]
            if "_" not in body:
                continue
            pitch, dur = body.split("_", 1)
            seq.append((pitch, dur.rstrip(".")))
        elif t.startswith("rest-"):
            dur = t.split("-", 1)[1].rstrip(".")
            seq.append((None, dur))
        # clef-, keySignature-, timeSignature-, barline, tie, fermata: skip
    return seq


def read_to_midi(jpg_path: str, checkpoint: str, midi_out: str,
                 device: str = "cpu") -> list[tuple[str | None, str]]:
    """full convenience path: photo -> CRNN -> MIDI file. returns the sequence."""
    from pitch_midi import write_midi
    model, vocab, height, dev = load_crnn(checkpoint, device)
    per_stave = read_score(jpg_path, model, vocab, height, dev)
    tokens = [t for stave in per_stave for t in stave]
    seq = tokens_to_sequence(tokens)
    write_midi(seq, midi_out)
    return seq


if __name__ == "__main__":
    import sys
    base = os.path.dirname(os.path.abspath(__file__))
    ckpt = os.path.join(base, "data", "crnn_camera.pt")
    name = sys.argv[1] if len(sys.argv) > 1 else "yankeeDoodle"
    jpg = os.path.join(base, "sheet music", f"{name}.jpg")
    out = os.path.join(base, "results", f"{name}_crnn.mid")
    seq = read_to_midi(jpg, ckpt, out, device="cpu")
    print(f"{name}: {len(seq)} symbols -> {out}")
    print("  ", seq[:20])
