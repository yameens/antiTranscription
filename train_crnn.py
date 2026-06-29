"""
train_crnn.py
train the end-to-end CRNN with CTC loss on PrIMuS treble staves.

usage
-----
    # quick smoke test (validate the loop)
    python train_crnn.py --max_staves 2000 --epochs 3

    # full camera-augmented run (the real model), in the background overnight
    python train_crnn.py --epochs 30 --augment camera --out data/crnn_camera.pt

    # clean baseline for the domain-gap ablation
    python train_crnn.py --epochs 30 --augment none --out data/crnn_clean.pt

checkpoints store model weights + the vocab token list so inference is
self-contained.  the best checkpoint (lowest clean-val SER) is kept.
"""

from __future__ import annotations

import argparse
import os
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from omr_dataset import load_manifest, SeqDataset, collate_ctc, IMG_HEIGHT
from omr_vocab import Vocab
from crnn_omr import CRNN, count_params
from crnn_decode import greedy_decode, batch_ser
from camera_aug import camera_augment

BASE = os.path.dirname(os.path.abspath(__file__))


def pick_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def split_rows(rows, val_frac, seed):
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n_val = max(1, int(len(rows) * val_frac))
    val_i = set(idx[:n_val])
    train = [r for i, r in enumerate(rows) if i not in val_i]
    val = [r for i, r in enumerate(rows) if i in val_i]
    return train, val


def unflatten_targets(flat, lengths):
    """split the concatenated CTC target back into per-sample id lists."""
    out, pos = [], 0
    for L in lengths.tolist():
        out.append(flat[pos: pos + L].tolist())
        pos += L
    return out


@torch.no_grad()
def evaluate(model, loader, device, blank):
    model.eval()
    preds_all, truths_all = [], []
    for imgs, tg, il, tl in loader:
        imgs = imgs.to(device)
        logp = model(imgs).cpu()           # CTC decode on cpu
        preds_all.extend(greedy_decode(logp, il, blank))
        truths_all.extend(unflatten_targets(tg, tl))
    ser, edits, tot = batch_ser(preds_all, truths_all)
    # exact-sequence accuracy too (whole staff perfectly read)
    exact = sum(1 for p, t in zip(preds_all, truths_all) if p == t)
    return ser, exact / len(preds_all) if preds_all else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(BASE, "data", "seq_manifest.csv"))
    ap.add_argument("--vocab", default=os.path.join(BASE, "data", "vocab.json"))
    ap.add_argument("--out", default=os.path.join(BASE, "data", "crnn_best.pt"))
    ap.add_argument("--max_staves", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--height", type=int, default=IMG_HEIGHT)
    ap.add_argument("--augment", choices=["camera", "none"], default="camera")
    ap.add_argument("--device", choices=["auto", "cpu", "mps"], default="cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"device: {device}")

    vocab = Vocab.load(args.vocab)
    rows = load_manifest(args.manifest)
    if args.max_staves:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.max_staves]
    train_rows, val_rows = split_rows(rows, args.val_frac, args.seed)
    print(f"staves: {len(rows)} ({len(train_rows)} train / {len(val_rows)} val), "
          f"vocab {vocab.n_classes} (+blank)")

    aug = camera_augment if args.augment == "camera" else None
    train_ds = SeqDataset(train_rows, vocab, height=args.height, augment=aug)
    val_ds = SeqDataset(val_rows, vocab, height=args.height, augment=None)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=collate_ctc, num_workers=args.workers,
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        collate_fn=collate_ctc, num_workers=args.workers)

    model = CRNN(vocab.n_classes, img_height=args.height).to(device)
    print(f"model params: {count_params(model):,}")
    ctc = nn.CTCLoss(blank=model.blank, zero_infinity=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_ser = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n_batches = 0
        for imgs, tg, il, tl in train_dl:
            imgs = imgs.to(device)
            logp = model(imgs)                       # (T, B, C) on device
            # CTCLoss runs on cpu for portability/correctness across backends
            loss = ctc(logp.cpu(), tg, il, tl)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += loss.item()
            n_batches += 1
        val_ser, val_exact = evaluate(model, val_dl, device, model.blank)
        dt = time.time() - t0
        print(f"epoch {epoch:2d}/{args.epochs}  loss {running / max(1,n_batches):.3f}  "
              f"val_SER {val_ser:.3f}  val_exact {val_exact:.3f}  ({dt:.0f}s)",
              flush=True)
        if val_ser < best_ser:
            best_ser = val_ser
            torch.save({"model_state": model.state_dict(),
                        "itos": vocab.itos,
                        "height": args.height,
                        "val_ser": val_ser,
                        "augment": args.augment,
                        "epoch": epoch}, args.out)
            print(f"  saved {args.out} (val_SER {val_ser:.3f})", flush=True)

    print(f"done. best val_SER {best_ser:.3f}")


if __name__ == "__main__":
    main()
