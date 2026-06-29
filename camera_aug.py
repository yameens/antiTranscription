"""
camera_aug.py
synthetic "phone-photo" degradation for clean PrIMuS staff images.

this is the domain-gap remedy.  the old classifier trained on crisp PrIMuS
glyphs and collapsed 58.9% -> 22.7% on real photos because training never
simulated what a phone camera does to a printed page.  here we apply, on the
fly during training, the degradations that actually destroyed the old model's
features:

    - perspective warp           (camera not fronto-parallel)
    - spatially-varying lighting (shadows, directional light)  <- not in old aug
    - blur (gaussian + motion)
    - JPEG recompression         (phone codec artefacts)        <- not in old aug
    - residual staff lines        (imperfect staff removal)      <- not in old aug
    - morphological ink-bleed     (fills hollow half-note heads) <- the killer
    - sensor noise

each effect fires with its own probability so the model sees a wide spread of
corruption levels.  input and output are uint8 grayscale (white paper, dark
ink).  rectification already fixes gross perspective, so the warp here is mild.

`camera_augment` is the training callable passed to SeqDataset(augment=...).
"""

from __future__ import annotations

import cv2
import numpy as np


def _perspective(gray: np.ndarray, rng: np.random.Generator, mag: float = 0.04) -> np.ndarray:
    h, w = gray.shape
    d = mag * min(h, w)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jit = rng.uniform(-d, d, size=(4, 2)).astype(np.float32)
    dst = src + jit
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(gray, M, (w, h), borderValue=255,
                               flags=cv2.INTER_LINEAR)


def _illumination(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """multiply by a smooth low-frequency field -> uneven brightness / shadow."""
    h, w = gray.shape
    # random low-res field, upsampled and smoothed -> gentle gradient
    field = rng.uniform(0.55, 1.15, size=(3, 3)).astype(np.float32)
    field = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=max(h, w) * 0.15)
    out = gray.astype(np.float32) * field
    return np.clip(out, 0, 255).astype(np.uint8)


def _blur(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        s = rng.uniform(0.4, 1.4)
        return cv2.GaussianBlur(gray, (0, 0), sigmaX=s)
    # mild motion blur: average along a random direction
    k = int(rng.integers(3, 6))
    kern = np.zeros((k, k), np.float32)
    if rng.random() < 0.5:
        kern[k // 2, :] = 1.0 / k          # horizontal
    else:
        kern[:, k // 2] = 1.0 / k          # vertical
    return cv2.filter2D(gray, -1, kern)


def _jpeg(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    q = int(rng.integers(25, 75))
    ok, enc = cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return gray
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)


def _staff_residue(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    overlay a few faint horizontal lines -> imperfectly removed staff lines.
    placed as an evenly-spaced 5-line group at a random vertical offset.
    """
    h, w = gray.shape
    out = gray.copy()
    spacing = rng.uniform(h * 0.10, h * 0.16)
    top = rng.uniform(h * 0.18, h * 0.40)
    darkness = int(rng.integers(90, 170))   # gray, not black -> faint
    for i in range(5):
        y = int(top + i * spacing)
        if 0 <= y < h:
            cv2.line(out, (0, y), (w - 1, y), darkness, 1)
    return out


def _ink_bleed(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    erode the grayscale (= grow dark ink) so thin strokes thicken and hollow
    note-heads fill in -- exactly the effect that destroyed the hollow/quarter
    distinction in real photos.
    """
    k = int(rng.integers(2, 4))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    if rng.random() < 0.8:
        return cv2.erode(gray, kern)        # thicken ink
    return cv2.dilate(gray, kern)           # occasionally thin (broken strokes)


def _noise(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sigma = rng.uniform(3, 12)
    n = rng.normal(0, sigma, size=gray.shape)
    return np.clip(gray.astype(np.float32) + n, 0, 255).astype(np.uint8)


def camera_augment(gray: np.ndarray, seed: int | None = None) -> np.ndarray:
    """
    apply a random subset of phone-photo degradations to a clean grayscale
    staff.  uint8 -> uint8.  intended as SeqDataset(augment=camera_augment).
    """
    rng = np.random.default_rng(seed)
    out = gray
    if rng.random() < 0.6:
        out = _perspective(out, rng)
    if rng.random() < 0.8:
        out = _illumination(out, rng)
    if rng.random() < 0.6:
        out = _ink_bleed(out, rng)
    if rng.random() < 0.4:
        out = _staff_residue(out, rng)
    if rng.random() < 0.6:
        out = _blur(out, rng)
    if rng.random() < 0.7:
        out = _noise(out, rng)
    if rng.random() < 0.7:
        out = _jpeg(out, rng)
    return out


if __name__ == "__main__":
    # dump a montage of clean vs augmented staves for visual inspection
    import os
    from omr_dataset import load_manifest
    base = os.path.dirname(os.path.abspath(__file__))
    rows = load_manifest(os.path.join(base, "data", "seq_manifest.csv"))
    out_dir = os.path.join(base, "results")
    os.makedirs(out_dir, exist_ok=True)
    for i in range(3):
        png = rows[i][0]
        g = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
        aug = camera_augment(g, seed=i)
        # pad to same width and stack vertically
        w = max(g.shape[1], aug.shape[1])
        canvas = np.full((g.shape[0] + aug.shape[0] + 8, w), 255, np.uint8)
        canvas[: g.shape[0], : g.shape[1]] = g
        canvas[g.shape[0] + 8 :, : aug.shape[1]] = aug
        cv2.imwrite(os.path.join(out_dir, f"camera_aug_demo_{i}.png"), canvas)
    print("wrote camera_aug_demo_{0,1,2}.png to results/")
