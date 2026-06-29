"""
omr_dataset.py
sequence dataset for the CRNN: (full staff image, semantic token sequence).

the old crop manifest (data/manifest.csv) is one (crop, duration) pair per row
and is useless for sequence modelling.  here we build a *sequence* manifest:
one row per staff = (png_path, space-joined token string), then a torch Dataset
that yields (image_tensor, target_ids) with a CTC-friendly collate.

build step walks the extracted PrIMuS tree (package_*/<id>/<id>.{png,semantic}),
keeps only in-scope staves (omr_vocab.in_scope), subsamples for CPU training,
and persists both the manifest and the vocab.
"""

from __future__ import annotations

import csv
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from omr_vocab import tokenize, in_scope, build_vocab, Vocab
from crnn_omr import WIDTH_DOWNSAMPLE

IMG_HEIGHT = 96
MAX_WIDTH = 1600          # cap very wide staves (memory)
MIN_WIDTH = 64            # discard degenerate slivers


# ---------------------------------------------------------------------------
# corpus walking + manifest building
# ---------------------------------------------------------------------------

def _iter_sample_dirs(primus_dir: str):
    """yield (sample_id, dir_path) over package_*/<id>/ or flat <id>/ layouts."""
    pkgs = sorted(d.path for d in os.scandir(primus_dir)
                  if d.is_dir() and os.path.basename(d.path).startswith("package_"))
    roots = pkgs if pkgs else [primus_dir]
    for root in roots:
        for d in os.scandir(root):
            if d.is_dir():
                yield d.name, d.path


def build_seq_manifest(
    primus_dir: str,
    manifest_csv: str,
    vocab_json: str,
    *,
    max_staves: int | None = None,
    seed: int = 42,
    scope_kwargs: dict | None = None,
    verbose: bool = True,
) -> tuple[int, Vocab]:
    """
    scan PrIMuS, keep in-scope staves, (optionally) subsample, write manifest +
    vocab.  returns (n_kept, vocab).
    """
    scope_kwargs = scope_kwargs or {}
    rng = random.Random(seed)

    kept: list[tuple[str, list[str]]] = []   # (png_path, tokens)
    n_seen = n_scope = 0
    for sid, path in _iter_sample_dirs(primus_dir):
        png = os.path.join(path, f"{sid}.png")
        sem = os.path.join(path, f"{sid}.semantic")
        if not (os.path.isfile(png) and os.path.isfile(sem)):
            continue
        n_seen += 1
        with open(sem) as fh:
            tokens = tokenize(fh.read())
        if not in_scope(tokens, **scope_kwargs):
            continue
        n_scope += 1
        kept.append((png, tokens))

    rng.shuffle(kept)
    if max_staves is not None and len(kept) > max_staves:
        kept = kept[:max_staves]

    vocab = build_vocab([t for _, t in kept])
    os.makedirs(os.path.dirname(os.path.abspath(manifest_csv)), exist_ok=True)
    with open(manifest_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["png_path", "tokens"])
        for png, toks in kept:
            w.writerow([png, " ".join(toks)])
    vocab.save(vocab_json)

    if verbose:
        print(f"scanned {n_seen} staves, {n_scope} in scope, kept {len(kept)}")
        print(f"vocab size: {vocab.n_classes} tokens (+1 CTC blank)")
        print(f"wrote {manifest_csv} and {vocab_json}")
    return len(kept), vocab


def load_manifest(manifest_csv: str) -> list[tuple[str, list[str]]]:
    rows: list[tuple[str, list[str]]] = []
    with open(manifest_csv) as fh:
        r = csv.reader(fh)
        next(r, None)  # header
        for png, toks in r:
            rows.append((png, toks.split()))
    return rows


# ---------------------------------------------------------------------------
# image preprocessing (shared by training and phone-photo inference)
# ---------------------------------------------------------------------------

def prep_staff_image(gray: np.ndarray, height: int = IMG_HEIGHT) -> np.ndarray:
    """
    grayscale staff (white paper, dark ink, uint8) -> float32 (H, W) in [0,1]
    with ink = high, background = low; height fixed, aspect preserved, width
    rounded to a multiple of the encoder's width-downsample factor.
    """
    h0, w0 = gray.shape[:2]
    new_w = max(MIN_WIDTH, int(round(w0 * height / max(1, h0))))
    new_w = min(new_w, MAX_WIDTH)
    new_w = max(WIDTH_DOWNSAMPLE, (new_w // WIDTH_DOWNSAMPLE) * WIDTH_DOWNSAMPLE)
    resized = cv2.resize(gray, (new_w, height), interpolation=cv2.INTER_AREA)
    inv = 1.0 - (resized.astype(np.float32) / 255.0)   # ink high, bg low
    return inv


# ---------------------------------------------------------------------------
# torch Dataset + CTC collate
# ---------------------------------------------------------------------------

class SeqDataset(Dataset):
    def __init__(self, rows: list[tuple[str, list[str]]], vocab: Vocab,
                 height: int = IMG_HEIGHT, augment=None):
        """
        rows    : list of (png_path, tokens)
        vocab   : token<->id map
        augment : optional callable(uint8 gray) -> uint8 gray applied before
                  resizing (training-time camera augmentation).  None = clean.
        """
        self.rows = rows
        self.vocab = vocab
        self.height = height
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        png, tokens = self.rows[idx]
        gray = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            # corrupt/missing image: fall back to a blank staff so a batch never crashes
            gray = np.full((self.height, MIN_WIDTH), 255, np.uint8)
        if self.augment is not None:
            gray = self.augment(gray)
        img = prep_staff_image(gray, self.height)            # (H, W) float32
        target = torch.tensor(self.vocab.encode(tokens), dtype=torch.long)
        return torch.from_numpy(img).unsqueeze(0), target    # (1,H,W), (L,)


def collate_ctc(batch):
    """
    right-pad images to the widest in the batch; build the flat CTC target.
    returns:
        imgs          (B, 1, H, Wmax)
        targets       (sum L,)  concatenated
        input_lengths (B,)      = true_width // WIDTH_DOWNSAMPLE
        target_lengths(B,)      = L per sample
    """
    imgs, targets = zip(*batch)
    h = imgs[0].shape[1]
    widths = [im.shape[2] for im in imgs]
    wmax = max(widths)
    padded = torch.zeros(len(imgs), 1, h, wmax, dtype=torch.float32)
    for i, im in enumerate(imgs):
        padded[i, :, :, : im.shape[2]] = im
    input_lengths = torch.tensor([w // WIDTH_DOWNSAMPLE for w in widths], dtype=torch.long)
    target_lengths = torch.tensor([t.numel() for t in targets], dtype=torch.long)
    flat_targets = torch.cat(targets) if targets else torch.zeros(0, dtype=torch.long)
    return padded, flat_targets, input_lengths, target_lengths


if __name__ == "__main__":
    import sys
    base = os.path.dirname(os.path.abspath(__file__))
    primus = os.path.join(base, "data", "primus")
    mani = os.path.join(base, "data", "seq_manifest.csv")
    vj = os.path.join(base, "data", "vocab.json")
    mx = int(sys.argv[1]) if len(sys.argv) > 1 else None
    # treble staves, durations whole-eighth (matches the phone-photo test pieces),
    # but accidentals + key signatures allowed -> 15,354 staves instead of 2,684.
    build_seq_manifest(
        primus, mani, vj, max_staves=mx,
        scope_kwargs=dict(no_note_accidentals=False, no_keysig_accidentals=False),
    )
