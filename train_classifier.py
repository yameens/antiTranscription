"""
train_classifier.py
build a symbol crop dataset, then train SymbolCNN.

two-step workflow — local images (recommended for this project)
--------------------------------------------------------------
step 1 — build the crop manifest from the 5 phone-photo images (run once):
    python train_classifier.py --buildlocal

step 2 — train:
    python train_classifier.py --epochs 30 --batch 32

two-step workflow — PrIMuS (optional, larger dataset)
-----------------------------------------------------
step 1:
    python train_classifier.py --build --primusdir /path/to/primusCalvoRizoApplied2018
step 2:
    python train_classifier.py --epochs 30 --batch 64

PrIMuS download: https://grfia.dlsi.ua.es/music/works/pianoScore/
"""

import argparse
import csv
import os
import random
import warnings

import cv2
import numpy as np

# torch / torchvision are imported lazily inside train() so that
# test_pitch_midi.py can import the shared utilities without needing them

from symbol_classifier import (
    CLASSES, CLASS_TO_IDX, N_CLASSES, IMG_SIZE,
    SymbolCNN, semantic_token_to_class,
)
from sheet_utils import (
    LOCAL_IMAGES, NOTE_DIR, SHEET_DIR,
    _DEFAULT_STAVE_PARAMS, _RELAXED_STAVE_PARAMS,
    parse_ground_truth, filter_note_components, run_pipeline,
)


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
BASE              = os.path.dirname(os.path.abspath(__file__))
DATA_DIR          = os.path.join(BASE, "data")
CROPS_DIR         = os.path.join(DATA_DIR, "crops")
MANIFEST_CSV      = os.path.join(DATA_DIR, "manifest.csv")        # PrIMuS train/val
LOCAL_MANIFEST_CSV = os.path.join(DATA_DIR, "local_manifest.csv") # phone-photo test set
CHECKPOINT        = os.path.join(DATA_DIR, "best_model.pt")


# ---------------------------------------------------------------------------
# local dataset builder
# ---------------------------------------------------------------------------


def build_dataset_from_local() -> None:
    """
    build the crop manifest from the 5 local sheet-music images using their
    hand-written ground-truth .txt files.

    workflow per image
    ------------------
    1. run the full pipeline: rectify -> detect_staves -> segment
    2. filter detected components to note-like shapes
    3. align note-like components 1-to-1 with the ground-truth note sequence
       (left-to-right order matches reading order)
    4. save each crop as a PNG and record (path, class_label) in the manifest

    components that outnumber the ground-truth notes are labelled "other"
    (they are likely accidentals, ties, or extra blobs; keeping them as
    "other" examples makes the classifier more robust).
    """
    os.makedirs(CROPS_DIR, exist_ok=True)
    rows: list[tuple[str, str]] = []

    for img_name, gt_filename in LOCAL_IMAGES:
        jpg_path = os.path.join(SHEET_DIR, f"{img_name}.jpg")
        gt_path  = os.path.join(NOTE_DIR, gt_filename)

        if not os.path.isfile(jpg_path):
            print(f"  skipping {img_name}: image not found")
            continue
        if not os.path.isfile(gt_path):
            print(f"  skipping {img_name}: ground-truth not found at {gt_path}")
            continue

        print(f"  processing {img_name} ...")
        try:
            rectified, staves, spacing, symbols, _ = run_pipeline(jpg_path)
        except Exception as exc:
            print(f"    pipeline failed: {exc}")
            continue

        gt_seq = parse_ground_truth(gt_path)
        h_img, w_img = rectified.shape[:2]
        note_components = filter_note_components(symbols, spacing, w_img, staves=staves)

        print(f"    gt notes: {len(gt_seq)}  detected note-like: {len(note_components)}")

        for i, (bbox, crop) in enumerate(note_components):
            if i < len(gt_seq):
                pitch, duration = gt_seq[i]
                if pitch is None:
                    label = f"rest_{duration}"
                else:
                    label = f"note_{duration}"
            else:
                label = "other"

            # validate the label is in our vocabulary
            if label not in CLASS_TO_IDX:
                label = "other"

            fname = f"{img_name}_{i:04d}.png"
            fpath = os.path.join(CROPS_DIR, fname)
            cv2.imwrite(fpath, crop)
            rows.append((fpath, label))

    with open(LOCAL_MANIFEST_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(rows)

    from collections import Counter
    dist = Counter(r[1] for r in rows)
    print(f"\nlocal manifest written to {LOCAL_MANIFEST_CSV}: {len(rows)} crops")
    print("class distribution:")
    for cls in CLASSES:
        if dist[cls]:
            print(f"  {cls:<18} {dist[cls]:>5}")


# ---------------------------------------------------------------------------
# dataset builder: extract symbol crops from PrIMuS stave images
# ---------------------------------------------------------------------------

def _detect_staff_spacing(stave_gray: np.ndarray) -> float:
    """
    estimate staff line spacing on a clean PrIMuS stave image using a
    row-wise projection profile.  peak distances in the profile equal the
    inter-line spacing.  this avoids importing the full hough-based
    detect_staves pipeline, which is tuned for phone photos.
    """
    _, binary = cv2.threshold(stave_gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_sum = binary.sum(axis=1).astype(float)
    row_sum /= row_sum.max() + 1e-6

    # find rows that are clearly staff lines (high pixel density)
    threshold = 0.3
    staff_rows = [i for i, v in enumerate(row_sum) if v > threshold]
    if len(staff_rows) < 2:
        return float(stave_gray.shape[0]) / 6.0   # safe fallback

    # consecutive-run centres
    runs: list[float] = []
    run_start = staff_rows[0]
    prev = staff_rows[0]
    for r in staff_rows[1:]:
        if r - prev > 2:
            runs.append((run_start + prev) / 2.0)
            run_start = r
        prev = r
    runs.append((run_start + prev) / 2.0)

    if len(runs) < 2:
        return float(stave_gray.shape[0]) / 6.0

    gaps = [runs[i + 1] - runs[i] for i in range(len(runs) - 1)]
    return float(np.median(gaps))


_NOTE_CLASSES = {"note_whole", "note_half", "note_quarter", "note_eighth"}


def _extract_crops_from_stave(
    stave_path: str,
    tokens: list[str],
    crops_dir: str,
    sample_id: str,
) -> list[tuple[str, str]]:
    """
    extract note-symbol crops from one PrIMuS stave and pair each crop with
    its duration class.  the stave is silently discarded (empty list returned)
    if the segmented-blob count does not exactly equal the note-token count —
    no padding, no shifting, no guessing.

    only note_whole / note_half / note_quarter / note_eighth are kept.
    rest tokens, clef tokens, barline tokens, and time-signature tokens are
    dropped entirely.  this keeps the label<->crop relationship unambiguous:
    every blob that passes the filter should correspond to exactly one note.

    discard conditions
    ------------------
    * blob count != note-token count  (beamed notes merge → fewer blobs;
      fragmented clef/timesig remnants → extra blobs; discard either way)
    * stave has zero note tokens
    * stave image cannot be read

    filtering criteria for note-like blobs
    ----------------------------------------
    after staff-line removal, each note leaves a blob whose shape depends on
    its duration:

      whole note  — no stem, ~1×sp wide, ~0.8×sp tall, roughly circular
      half note   — hollow notehead + stem, ~1×sp wide, ~3-4×sp tall
      quarter     — filled notehead + stem, same height as half
      eighth      — filled notehead + stem + flag, ~1×sp wide, ~4-5×sp tall

    common impostors and their exclusion:
      barline     — very thin (< 0.4×sp wide), very tall   → excluded by min_w
      time-sig    — roughly square at ~2×sp, placed before first note → causes
                    count mismatch → stave discarded
      clef pieces — wider (> 3×sp) or taller (> 6×sp) blobs → excluded by
                    max_w / max_h
      tiny dots   — area too small                          → excluded by min_area
      stem-only   — too thin (< 0.4×sp wide)                → excluded by min_w

    returns a list of (crop_path, class_label) pairs, or [] on discard.
    """
    img = cv2.imread(stave_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    h_img, w = img.shape
    spacing = _detect_staff_spacing(img)

    # --- 1. extract note tokens only (no rests, no clef, no barline) ---
    note_labels = [
        semantic_token_to_class(t) for t in tokens
        if t.strip() and semantic_token_to_class(t) in _NOTE_CLASSES
    ]
    if not note_labels:
        return []

    # --- 2. binarize, remove staff lines ---
    _, binary = cv2.threshold(img, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    row_sum = binary.sum(axis=1).astype(float)
    row_sum_norm = row_sum / (w * 255.0 + 1e-6)
    staff_mask = (row_sum_norm > 0.25).astype(np.uint8) * 255
    line_pixels = staff_mask[:, np.newaxis] * np.ones(w, dtype=np.uint8)
    cleaned = cv2.bitwise_and(binary, cv2.bitwise_not(line_pixels))

    # vertical close: reconnect note stems severed by staff-line removal.
    # whole notes have no stem so the close does not hurt them.
    bridge_k = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(3, int(spacing * 0.4)))
    )
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, bridge_k)

    # open to remove single-pixel stubs left at line intersections
    stub_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, stub_k)

    # --- 3. connected components ---
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )

    # --- 4. note-shaped blob filter ---
    # min/max sizes tuned for notes, not clef fragments, barlines, or timesig.
    # whole notes are short (no stem); eighth notes are tall (stem + flag).
    # aspect-ratio cap (h/w ≤ 9) removes barlines which are extremely thin.
    min_h    = max(3, int(spacing * 0.5))   # whole-note notehead height
    max_h    = int(spacing * 6.0)            # cap below very tall clef blobs
    min_w    = max(3, int(spacing * 0.4))   # exclude 1-2 px barlines
    max_w    = int(spacing * 3.0)            # exclude wide clef/key parts
    min_area = int(spacing ** 2 * 0.20)     # exclude tiny pixel noise
    max_ar   = 9.0                           # h/w: barlines > 15, notes < 8

    components: list[tuple[int, int, int, int]] = []
    for lbl in range(1, n_labels):
        x    = int(stats[lbl, cv2.CC_STAT_LEFT])
        y    = int(stats[lbl, cv2.CC_STAT_TOP])
        bw   = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh   = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if bh < min_h or bh > max_h:
            continue
        if bw < min_w or bw > max_w:
            continue
        if bw > 0 and (bh / bw) > max_ar:   # too thin/tall → barline
            continue
        components.append((x, y, bw, bh))

    components.sort(key=lambda c: c[0])   # left-to-right reading order

    # --- 5. strict count gate: discard stave on any mismatch ---
    # beamed eighths merge into one blob → fewer blobs than tokens → discard.
    # clef/timesig remnants that survive the filter → extra blobs → discard.
    # either way the pairing would be wrong, so it is better to throw away
    # the stave than to introduce mislabeled training samples.
    if len(components) != len(note_labels):
        return []

    # --- 6. save crops and return (path, label) pairs ---
    pairs: list[tuple[str, str]] = []
    for i, (x, y, bw, bh) in enumerate(components):
        label = note_labels[i]
        crop  = cleaned[y: y + bh, x: x + bw]
        if crop.size == 0:
            continue
        fname = f"{sample_id}_{i:04d}.png"
        fpath = os.path.join(crops_dir, fname)
        cv2.imwrite(fpath, crop)
        pairs.append((fpath, label))

    return pairs


def build_manifest(primusdir: str, max_samples: int = 5000) -> None:
    """
    scan primusdir for stave images, extract symbol crops, write manifest.csv.

    primusdir should be the top-level PrIMuS directory containing one
    subdirectory per sample.  each subdirectory is named by its sample id.

    max_samples caps the number of staves processed; set to None to use all.

    staves where the filtered note-blob count does not exactly match the
    note-token count are silently discarded — no padding, no shifting.
    the summary printed at the end reports how many staves were kept vs
    discarded and gives the per-class crop distribution.
    """
    os.makedirs(CROPS_DIR, exist_ok=True)

    # if primusdir contains package_* subdirectories (the full PrIMuS layout with
    # package_aa/ and package_ab/), gather stave dirs from all of them so that a
    # single --primusdir . covers the whole corpus.  otherwise fall back to treating
    # primusdir as a flat directory of stave dirs (single-package usage).
    package_dirs = sorted(
        d.path for d in os.scandir(primusdir)
        if d.is_dir() and os.path.basename(d.path).startswith("package_")
    )
    if package_dirs:
        all_entries: list = []
        for pkg in package_dirs:
            all_entries.extend(
                d for d in os.scandir(pkg) if d.is_dir()
            )
        sample_dirs = sorted(all_entries, key=lambda d: d.name)
    else:
        sample_dirs = sorted(
            (d for d in os.scandir(primusdir) if d.is_dir()),
            key=lambda d: d.name,
        )

    if max_samples is not None:
        sample_dirs = sample_dirs[:max_samples]

    rows: list[tuple[str, str]] = []
    n_scanned    = 0   # staves with both .png + .semantic present
    n_kept       = 0   # staves where blob count == note-token count
    n_discarded  = 0   # staves that failed the strict count gate
    n_no_notes   = 0   # staves with zero note tokens (skipped before filter)

    for entry in sample_dirs:
        sid = entry.name
        stave_png  = os.path.join(entry.path, f"{sid}.png")
        semantic_f = os.path.join(entry.path, f"{sid}.semantic")

        if not os.path.isfile(stave_png) or not os.path.isfile(semantic_f):
            continue

        n_scanned += 1

        with open(semantic_f, "r") as f:
            tokens = f.read().split()

        # fast pre-check: does this stave even have note tokens?
        has_notes = any(
            semantic_token_to_class(t) in _NOTE_CLASSES
            for t in tokens if t.strip()
        )
        if not has_notes:
            n_no_notes += 1
            continue

        pairs = _extract_crops_from_stave(stave_png, tokens, CROPS_DIR, sid)

        if pairs:
            rows.extend(pairs)
            n_kept += 1
        else:
            n_discarded += 1

        if n_scanned % 200 == 0:
            print(f"  scanned {n_scanned}/{len(sample_dirs)}  "
                  f"kept {n_kept}  discarded {n_discarded}  "
                  f"crops so far {len(rows)}")

    with open(MANIFEST_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(rows)

    from collections import Counter
    dist = Counter(r[1] for r in rows)

    total_staves = n_kept + n_discarded + n_no_notes
    keep_pct     = 100 * n_kept / total_staves if total_staves else 0.0

    print(f"\n{'─'*52}")
    print(f"  staves scanned  : {n_scanned:>6}")
    print(f"  no note tokens  : {n_no_notes:>6}  (skipped before filter)")
    print(f"  count mismatch  : {n_discarded:>6}  (discarded)")
    print(f"  kept            : {n_kept:>6}  ({keep_pct:.1f}%)")
    print(f"  total crops     : {len(rows):>6}")
    print(f"{'─'*52}")
    print(f"  per-class distribution:")
    for cls in _NOTE_CLASSES:
        print(f"    {cls:<18} {dist.get(cls, 0):>6}")
    print(f"{'─'*52}")
    print(f"manifest written to {MANIFEST_CSV}")


# ---------------------------------------------------------------------------
# PyTorch dataset  (torch imported lazily here so the rest of the module
# can be imported without torch being installed)
# ---------------------------------------------------------------------------

def _make_symbol_dataset_class():
    import torch
    from torch.utils.data import Dataset as _Dataset

    class SymbolDataset(_Dataset):
        """
        loads symbol crops from the CSV manifest produced by build_manifest
        or build_dataset_from_local.
        applies torchvision transforms so augmentation is defined externally.
        """

        def __init__(self, rows: list[tuple[str, str]], transform=None):
            self.rows = rows
            self.transform = transform

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int):
            path, label = self.rows[idx]
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0

            if self.transform is not None:
                tensor = self.transform(tensor)

            label_idx = CLASS_TO_IDX.get(label, CLASS_TO_IDX["other"])
            return tensor, label_idx

    return SymbolDataset


def _load_manifest() -> list[tuple[str, str]]:
    if not os.path.isfile(MANIFEST_CSV):
        raise FileNotFoundError(
            f"manifest not found at {MANIFEST_CSV}.  "
            "run:  python train_classifier.py --build --primusdir /path/to/primusdir"
        )
    with open(MANIFEST_CSV, "r") as f:
        reader = csv.DictReader(f)
        return [(r["path"], r["label"]) for r in reader
                if os.path.isfile(r["path"])]


# ---------------------------------------------------------------------------
# augmentation transforms
# ---------------------------------------------------------------------------

def _random_morph(t: "torch.Tensor") -> "torch.Tensor":
    """
    randomly dilate or erode a single-channel binary crop to simulate
    stroke-thickness variation between clean print and phone-photo.

    dilation  -> max-pool with a small kernel (thickens strokes)
    erosion   -> negate, max-pool, negate (thins strokes)

    operates in-place on a [1, H, W] float tensor in [0, 1].
    """
    import torch
    import torch.nn.functional as F
    k = random.choice([3, 5])
    pad = k // 2
    if random.random() < 0.5:
        # dilate
        t = F.max_pool2d(t.unsqueeze(0), kernel_size=k, stride=1,
                         padding=pad).squeeze(0)
    else:
        # erode
        t = 1.0 - F.max_pool2d(1.0 - t.unsqueeze(0), kernel_size=k,
                                stride=1, padding=pad).squeeze(0)
    return t.clamp(0.0, 1.0)


def _train_transform():
    """
    heavier augmentation pipeline to simulate phone-photo conditions:
    - larger affine warp (rotation, scale, shear) for camera angle variation
    - brightness/contrast jitter for lighting changes
    - stronger gaussian blur at higher probability for camera blur
    - random morphological dilation/erosion for stroke-thickness variation
    - additive gaussian noise for sensor noise
    - random erasing for partial occlusion

    torch / torchvision imported lazily so this module loads without them.
    """
    import torch
    from torchvision import transforms
    return transforms.Compose([
        transforms.RandomAffine(
            degrees=12, translate=(0.10, 0.10),
            scale=(0.70, 1.30), shear=6, fill=0,
        ),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 2.5))], p=0.6,
        ),
        transforms.RandomApply(
            [transforms.Lambda(_random_morph)], p=0.5,
        ),
        transforms.Lambda(
            lambda t: (t + torch.randn_like(t) * 0.06).clamp(0.0, 1.0)
        ),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.12), value=0),
    ])


def _val_transform():
    from torchvision import transforms
    return transforms.Compose([])


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------

def train(
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 3e-4,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    random.seed(seed)

    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"device: {device}")

    rows = _load_manifest()
    print(f"manifest loaded: {len(rows)} samples, {N_CLASSES} classes")

    random.shuffle(rows)
    n_val = max(1, int(len(rows) * val_fraction))
    val_rows   = rows[:n_val]
    train_rows = rows[n_val:]

    SymbolDataset = _make_symbol_dataset_class()
    train_ds = SymbolDataset(train_rows, transform=_train_transform())
    val_ds   = SymbolDataset(val_rows,   transform=_val_transform())

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    from collections import Counter
    counts = Counter(CLASS_TO_IDX.get(r[1], CLASS_TO_IDX["other"])
                     for r in train_rows)
    # sqrt-inverse frequency weighting: gives rare classes a gentle 3-5x boost
    # rather than the raw 800x ratio from the old formula, which caused training
    # to destabilize and decay after epoch 3.
    freqs = torch.tensor([counts.get(i, 1) for i in range(N_CLASSES)],
                         dtype=torch.float32)
    weights = (1.0 / freqs).sqrt()
    weights = weights / weights.mean()  # center around 1.0
    weights = weights.clamp(max=5.0)   # cap so no class hijacks the gradient
    weights = weights.to(device)

    model = SymbolCNN(n_classes=N_CLASSES, img_size=IMG_SIZE).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 10
    )

    best_val_acc = 0.0
    os.makedirs(DATA_DIR, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        train_loss = 0.0
        train_correct = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss   += loss.item() * imgs.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()

        scheduler.step()

        train_acc  = train_correct / len(train_ds)
        train_loss = train_loss    / len(train_ds)

        # validate
        model.eval()
        val_correct = 0
        val_loss    = 0.0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                val_loss    += criterion(logits, labels).item() * imgs.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()

        val_acc  = val_correct / len(val_ds)
        val_loss = val_loss    / len(val_ds)

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_acc":     val_acc,
                "classes":     CLASSES,
            }, CHECKPOINT)
            marker = "  *** best ***"

        print(
            f"epoch {epoch:3d}/{epochs}"
            f"  train loss {train_loss:.4f}  acc {train_acc:.3f}"
            f"  val loss {val_loss:.4f}  acc {val_acc:.3f}"
            f"  lr {scheduler.get_last_lr()[0]:.2e}"
            f"{marker}"
        )

    print(f"\ntraining complete.  best val acc: {best_val_acc:.4f}")
    print(f"model saved to: {CHECKPOINT}")

    # per-class accuracy on full validation set
    _per_class_report(model, val_loader, device)


def _per_class_report(model, val_loader, device):
    import torch
    from collections import defaultdict
    model.eval()
    correct = defaultdict(int)
    total   = defaultdict(int)

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            for pred, lbl in zip(preds, labels):
                total[lbl.item()]   += 1
                correct[lbl.item()] += int(pred == lbl)

    print("\nper-class accuracy on validation set:")
    print(f"  {'class':<18}  {'correct':>7}  {'total':>7}  {'acc':>6}")
    print("  " + "-" * 46)
    for i, cls in enumerate(CLASSES):
        n = total[i]
        c = correct[i]
        acc = c / n if n > 0 else 0.0
        print(f"  {cls:<18}  {c:>7}  {n:>7}  {acc:>6.3f}")


# ---------------------------------------------------------------------------
# test on local phone-photo crops (domain-transfer evaluation)
# ---------------------------------------------------------------------------

def test_on_local() -> None:
    """
    load the best saved model and evaluate it on the local phone-photo crops
    (data/local_manifest.csv).  these crops were never seen during PrIMuS
    training, so this measures domain transfer: clean printed music -> phone photo.

    run after training:
        python3 train_classifier.py --testlocal
    """
    import torch
    from torch.utils.data import DataLoader
    from collections import defaultdict

    if not os.path.isfile(CHECKPOINT):
        raise FileNotFoundError(
            f"no checkpoint found at {CHECKPOINT}.  train first with --epochs."
        )
    if not os.path.isfile(LOCAL_MANIFEST_CSV):
        raise FileNotFoundError(
            f"local manifest not found at {LOCAL_MANIFEST_CSV}.  "
            "run --buildlocal first."
        )

    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cpu")
    )

    ckpt  = torch.load(CHECKPOINT, map_location=device)
    model = SymbolCNN(n_classes=N_CLASSES, img_size=IMG_SIZE).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with open(LOCAL_MANIFEST_CSV, "r") as f:
        rows = [(r["path"], r["label"]) for r in csv.DictReader(f)
                if os.path.isfile(r["path"])]

    SymbolDataset = _make_symbol_dataset_class()
    loader = DataLoader(
        SymbolDataset(rows, transform=_val_transform()),
        batch_size=32, shuffle=False, num_workers=0,
    )

    correct_total = 0
    correct_by    = defaultdict(int)
    total_by      = defaultdict(int)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            for pred, lbl in zip(preds, labels):
                total_by[lbl.item()]   += 1
                correct_by[lbl.item()] += int(pred == lbl)
                correct_total          += int(pred == lbl)

    overall = correct_total / len(rows) if rows else 0.0
    print(f"\ndomain-transfer test on local phone-photo crops")
    print(f"  checkpoint: epoch {ckpt.get('epoch', '?')}  "
          f"PrIMuS val acc {ckpt.get('val_acc', 0):.3f}")
    print(f"  local test crops: {len(rows)}")
    print(f"  overall accuracy: {correct_total}/{len(rows)}  ({overall:.3f})\n")
    print(f"  {'class':<18}  {'correct':>7}  {'total':>7}  {'acc':>6}")
    print("  " + "-" * 46)
    for i, cls in enumerate(CLASSES):
        n = total_by[i]
        c = correct_by[i]
        acc = c / n if n > 0 else 0.0
        print(f"  {cls:<18}  {c:>7}  {n:>7}  {acc:>6.3f}")


# ---------------------------------------------------------------------------
# crop-quality diagnostic
# ---------------------------------------------------------------------------

def dump_crop_montage(n_per_class: int = 12) -> None:
    """
    sample n_per_class crops per class from the PrIMuS manifest and from the
    local manifest, tile them into labeled grids, and save to results/.

    this lets you visually confirm whether local segmented crops look like
    PrIMuS symbols before committing to a manifest rebuild.
    """
    import math
    results_dir = os.path.join(BASE, "results")
    os.makedirs(results_dir, exist_ok=True)

    def _load_rows(csv_path: str) -> dict[str, list[str]]:
        if not os.path.isfile(csv_path):
            return {}
        with open(csv_path, "r") as f:
            by_class: dict[str, list[str]] = {c: [] for c in CLASSES}
            for row in csv.DictReader(f):
                cls = row["label"]
                if cls in by_class and os.path.isfile(row["path"]):
                    by_class[cls].append(row["path"])
        return by_class

    def _make_montage(by_class: dict[str, list[str]],
                      n: int, thumb: int = 64) -> np.ndarray:
        cols = n
        rows_count = len(CLASSES)
        canvas_w = cols * (thumb + 2) + 80   # 80 px label gutter on left
        canvas_h = rows_count * (thumb + 2)
        canvas = np.full((canvas_h, canvas_w), 200, dtype=np.uint8)

        for r, cls in enumerate(CLASSES):
            paths = by_class.get(cls, [])
            random.shuffle(paths)
            paths = paths[:n]
            y0 = r * (thumb + 2)
            cv2.putText(canvas, cls[:14], (2, y0 + thumb // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, 40, 1, cv2.LINE_AA)
            for c, p in enumerate(paths):
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (thumb, thumb), interpolation=cv2.INTER_AREA)
                x0 = 80 + c * (thumb + 2)
                canvas[y0: y0 + thumb, x0: x0 + thumb] = img
        return canvas

    for label, csv_path, out_name in [
        ("PrIMuS",      MANIFEST_CSV,       "crop_montage_primus.png"),
        ("local photo", LOCAL_MANIFEST_CSV, "crop_montage_local.png"),
    ]:
        by_class = _load_rows(csv_path)
        if not by_class:
            print(f"  {label}: manifest not found, skipping")
            continue
        montage = _make_montage(by_class, n_per_class)
        out_path = os.path.join(results_dir, out_name)
        cv2.imwrite(out_path, montage)
        print(f"  {label} montage -> {out_path}")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train symbol CNN on PrIMuS crops"
    )
    parser.add_argument(
        "--buildlocal", action="store_true",
        help="build local test manifest from the 5 phone-photo images",
    )
    parser.add_argument(
        "--build", action="store_true",
        help="extract crops from PrIMuS and write manifest.csv (run once)",
    )
    parser.add_argument(
        "--testlocal", action="store_true",
        help="evaluate best_model.pt on local phone-photo crops (domain transfer)",
    )
    parser.add_argument(
        "--dumpcrops", action="store_true",
        help="save per-class crop montages for PrIMuS and local manifests to results/",
    )
    parser.add_argument(
        "--primusdir", type=str, default=None,
        help="path to the top-level PrIMuS directory (required for --build)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=5000,
        help="max number of staves to process during --build (default 5000)",
    )
    parser.add_argument(
        "--epochs",  type=int,   default=30,   help="training epochs"
    )
    parser.add_argument(
        "--batch",   type=int,   default=64,   help="batch size"
    )
    parser.add_argument(
        "--lr",      type=float, default=3e-4, help="initial learning rate"
    )
    parser.add_argument(
        "--val",     type=float, default=0.15, help="validation fraction"
    )
    parser.add_argument(
        "--seed",    type=int,   default=42,   help="random seed"
    )
    args = parser.parse_args()

    if args.buildlocal:
        print("building local test manifest ...")
        build_dataset_from_local()
    elif args.build:
        if args.primusdir is None:
            parser.error("--primusdir is required when using --build")
        print(f"building PrIMuS manifest from: {args.primusdir}")
        build_manifest(args.primusdir, max_samples=args.max_samples)
    elif args.testlocal:
        test_on_local()
    elif args.dumpcrops:
        print("generating crop montages ...")
        dump_crop_montage()
    else:
        train(
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            val_fraction=args.val,
            seed=args.seed,
        )
