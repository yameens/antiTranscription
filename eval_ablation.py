"""
eval_ablation.py
quantify how much the camera augmentation closes the clean-to-real domain gap.

evaluate a checkpoint twice on the SAME held-out PrIMuS val split:
  - clean val   (crisp engravings, the train-time distribution of a clean model)
  - camera val  (the same staves degraded with phone-photo augmentation)

a CLEAN-trained model should score well on clean val but its SER should blow up
on camera val -> that gap IS the domain gap.  a CAMERA-trained model should keep
a low SER on camera val -> the gap is closed.  running this on both checkpoints
reproduces, at the sequence level, the 58.9->22.7 collapse the old CNN showed.

usage:  python eval_ablation.py --ckpt data/crnn_camera.pt [--device cpu]
"""

from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from omr_dataset import load_manifest, SeqDataset, collate_ctc
from omr_vocab import Vocab
from crnn_omr import CRNN
from crnn_decode import greedy_decode, batch_ser
from camera_aug import camera_augment
from train_crnn import split_rows, unflatten_targets, pick_device

BASE = os.path.dirname(os.path.abspath(__file__))


def seeded_camera(idx_holder):
    """deterministic per-sample camera aug so the camera-val set is reproducible."""
    counter = {"i": 0}

    def aug(gray):
        s = counter["i"]
        counter["i"] += 1
        return camera_augment(gray, seed=1000 + s)
    return aug


@torch.no_grad()
def eval_ser(model, loader, device, blank):
    model.eval()
    preds, truths = [], []
    for imgs, tg, il, tl in loader:
        logp = model(imgs.to(device)).cpu()
        preds.extend(greedy_decode(logp, il, blank))
        truths.extend(unflatten_targets(tg, tl))
    ser, _, _ = batch_ser(preds, truths)
    exact = sum(1 for p, t in zip(preds, truths) if p == t) / max(1, len(preds))
    return ser, exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(BASE, "data", "crnn_camera.pt"))
    ap.add_argument("--manifest", default=os.path.join(BASE, "data", "seq_manifest.csv"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = pick_device(args.device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    vocab = Vocab.__new__(Vocab)
    vocab.itos = ckpt["itos"]
    vocab.stoi = {t: i for i, t in enumerate(vocab.itos)}
    height = ckpt.get("height", 96)
    model = CRNN(vocab.n_classes, img_height=height).to(device)
    model.load_state_dict(ckpt["model_state"])

    rows = load_manifest(args.manifest)
    _, val_rows = split_rows(rows, args.val_frac, args.seed)

    clean_ds = SeqDataset(val_rows, vocab, height=height, augment=None)
    cam_ds = SeqDataset(val_rows, vocab, height=height, augment=seeded_camera(None))
    clean_dl = DataLoader(clean_ds, batch_size=16, collate_fn=collate_ctc, num_workers=4)
    cam_dl = DataLoader(cam_ds, batch_size=16, collate_fn=collate_ctc, num_workers=0)

    clean_ser, clean_exact = eval_ser(model, clean_dl, device, model.blank)
    cam_ser, cam_exact = eval_ser(model, cam_dl, device, model.blank)

    print(f"checkpoint: {os.path.basename(args.ckpt)} "
          f"(trained with augment={ckpt.get('augment','?')})")
    print(f"  clean  val: SER {clean_ser:.3f}  exact-staff {clean_exact:.3f}")
    print(f"  camera val: SER {cam_ser:.3f}  exact-staff {cam_exact:.3f}")
    print(f"  domain gap (camera - clean SER): {cam_ser - clean_ser:+.3f}")


if __name__ == "__main__":
    main()
