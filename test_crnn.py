"""
test_crnn.py
evaluate the CRNN on the real phone photos and compare it head-to-head with the
geometry-only reader (the current best) and the old segment+CNN baseline.

metrics, per piece:
  - pitch:    exact and within-one-step, via the standard Needleman-Wunsch
              alignment used elsewhere in the project (align_pitches).
  - duration: accuracy over alignment-matched notes (CRNN only; geometry has
              no duration).
  - note-SER: token-level Symbol Error Rate over the note/rest sequence
              (CRNN only).

usage:  python test_crnn.py [--device cpu|mps] [--ckpt data/crnn_camera.pt]
"""

from __future__ import annotations

import argparse
import os

from sheet_utils import parse_ground_truth, NOTE_DIR
from pitch_geometry import read_pitches, align_pitches, pitch_step
from crnn_decode import symbol_error_rate
from read_crnn import load_crnn, read_score, tokens_to_sequence

BASE = os.path.dirname(os.path.abspath(__file__))
SHEET = os.path.join(BASE, "sheet music")

PIECES = ["yankeeDoodle", "twinkleTwinkleLittleStar", "maryHadLittleLamb", "cs131"]
GT_FILES = {"maryHadLittleLamb": "marryHadLittleLamb.txt"}

# old segment+CNN baseline, from the project report (Table 3) for reference
CNN_BASELINE = {
    "yankeeDoodle": (14, 50), "twinkleTwinkleLittleStar": (10, 19),
    "maryHadLittleLamb": (8, 23), "cs131": (8, 17),
}


def _gt(name):
    return parse_ground_truth(os.path.join(NOTE_DIR, GT_FILES.get(name, f"{name}.txt")))


def pitch_scores(det_pitches, gt_pitches):
    pairs = align_pitches(det_pitches, gt_pitches)
    exact = near = 0
    for di, gj in pairs:
        if di is None or gj is None:
            continue
        if det_pitches[di] == gt_pitches[gj]:
            exact += 1
            near += 1
        else:
            sd, sg = pitch_step(det_pitches[di]), pitch_step(gt_pitches[gj])
            if sd is not None and sg is not None and abs(sd - sg) <= 1:
                near += 1
    n = len(gt_pitches)
    return 100 * exact / n, 100 * near / n


def duration_accuracy(det_seq, gt_seq):
    """alignment-matched duration accuracy using pitch alignment for anchoring."""
    det_p = [p for p, _ in det_seq]
    gt_p = [p if p else f"REST" for p, _ in gt_seq]
    # align on a pitch-like token (rests as 'REST') so durations line up
    det_tok = [p if p else "REST" for p, _ in det_seq]
    pairs = align_pitches(det_tok, gt_p)
    match = total = 0
    for di, gj in pairs:
        if di is None or gj is None:
            continue
        total += 1
        if det_seq[di][1] == gt_seq[gj][1]:
            match += 1
    return (100 * match / total) if total else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(BASE, "data", "crnn_camera.pt"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model, vocab, height, dev = load_crnn(args.ckpt, args.device)
    print(f"loaded {os.path.basename(args.ckpt)} "
          f"(val_SER {load_ckpt_ser(args.ckpt):.3f}, vocab {vocab.n_classes})\n")

    hdr = f"{'piece':<26}{'CNN base':>12}{'geometry':>14}{'CRNN':>14}{'CRNN dur':>10}{'noteSER':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name in PIECES:
        gt_full = _gt(name)
        gt_pitches = [p for p, _ in gt_full if p]
        gt_notes = [(p, d) for p, d in gt_full]  # incl rests for duration/SER

        # --- geometry baseline ---
        try:
            _, _, _, nh = read_pitches(os.path.join(SHEET, f"{name}.jpg"))
            geo_pitches = [n.pitch for n in nh]
            geo_e, geo_n = pitch_scores(geo_pitches, gt_pitches)
            geo_str = f"{geo_e:.0f}/{geo_n:.0f}"
        except Exception as e:
            geo_str = "ERR"

        # --- CRNN ---
        try:
            per_stave = read_score(os.path.join(SHEET, f"{name}.jpg"),
                                   model, vocab, height, dev)
            tokens = [t for stave in per_stave for t in stave]
            crnn_seq = tokens_to_sequence(tokens)
            crnn_pitches = [p for p, _ in crnn_seq if p]
            cr_e, cr_n = pitch_scores(crnn_pitches, gt_pitches)
            cr_dur = duration_accuracy(crnn_seq, gt_notes)
            # note-level SER: predicted note/rest tokens vs GT note/rest tokens
            pred_tok = [(f"note-{p}_{d}" if p else f"rest-{d}") for p, d in crnn_seq]
            gt_tok = [(f"note-{p}_{d}" if p else f"rest-{d}") for p, d in gt_notes]
            ser = symbol_error_rate(pred_tok, gt_tok)
            cr_str = f"{cr_e:.0f}/{cr_n:.0f}"
            dur_str = f"{cr_dur:.0f}%"
            ser_str = f"{ser:.2f}"
        except Exception as e:
            cr_str, dur_str, ser_str = f"ERR:{type(e).__name__}", "-", "-"

        cb = CNN_BASELINE.get(name, ("-", "-"))
        print(f"{name:<26}{f'{cb[0]}/{cb[1]}':>12}{geo_str:>14}{cr_str:>14}{dur_str:>10}{ser_str:>9}")
    print("\ncolumns: exact/±1 pitch %  |  CRNN dur = duration acc  |  noteSER lower=better")


def load_ckpt_ser(path):
    import torch
    return torch.load(path, map_location="cpu").get("val_ser", float("nan"))


if __name__ == "__main__":
    main()
