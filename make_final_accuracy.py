"""
make_final_accuracy.py
build the 'final accuracy/' deliverable: a single composite hero infographic
summarising the CRNN+CTC upgrade, plus copies of the source result figures.

run:  python make_final_accuracy.py
"""

import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
OUT = os.path.join(BASE, "final accuracy")
os.makedirs(OUT, exist_ok=True)

# ---- palette ----
RED = "#c0392b"
AMBER = "#e6a817"
GREEN = "#1f8b4c"
BLUE = "#2c7fb8"
INK = "#1b2a3a"
MUTED = "#6b7886"

# ---- measured data ----
pieces = ["Yankee\nDoodle", "Twinkle\nTwinkle", "Mary Had\na Lamb", "CS 131"]
cnn = [14, 10, 8, 8]
geom = [86, 86, 69, 60]
crnn = [100, 90, 100, 90]


def _box(ax, x, y, w, h, text, fc, ec, tc="white", fs=8.5, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.012,rounding_size=0.02",
                 linewidth=1.4, edgecolor=ec, facecolor=fc, mutation_aspect=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold" if bold else "normal",
            zorder=5)


def _arrow(ax, x0, y0, x1, y1, color=MUTED):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=12, linewidth=1.6, color=color, zorder=4))


def main():
    fig = plt.figure(figsize=(13, 15.5))
    gs = fig.add_gridspec(4, 2, height_ratios=[0.62, 1.0, 1.0, 0.95],
                          hspace=0.42, wspace=0.22,
                          left=0.06, right=0.96, top=0.95, bottom=0.04)

    # ===== HEADLINE BAND =====
    axh = fig.add_subplot(gs[0, :]); axh.axis("off")
    axh.set_xlim(0, 1); axh.set_ylim(0, 1)
    axh.text(0.5, 0.86, "From Phone Photo to MIDI — Accuracy Upgrade",
             ha="center", va="center", fontsize=22, fontweight="bold", color=INK)
    axh.text(0.5, 0.60,
             "Replacing segment-and-classify with an end-to-end CRNN+CTC reader",
             ha="center", va="center", fontsize=12.5, color=MUTED)
    # three headline stat chips
    chips = [("8–14%", "old CNN\n(segment + classify)", RED),
             ("60–86%", "geometry-only\nreader", AMBER),
             ("90–100%", "CRNN + CTC\n(this work)", GREEN)]
    cw, gap = 0.24, 0.04
    total = 3 * cw + 2 * gap
    x0 = (1 - total) / 2
    for i, (big, small, col) in enumerate(chips):
        cx = x0 + i * (cw + gap)
        axh.add_patch(FancyBboxPatch((cx, 0.04), cw, 0.42,
                      boxstyle="round,pad=0.006,rounding_size=0.03",
                      linewidth=0, facecolor=col, alpha=0.12))
        axh.text(cx + cw / 2, 0.32, big, ha="center", va="center",
                 fontsize=20, fontweight="bold", color=col)
        axh.text(cx + cw / 2, 0.13, small, ha="center", va="center",
                 fontsize=8.7, color=INK)
        if i < 2:
            axh.text(cx + cw + gap / 2, 0.25, "→", ha="center", va="center",
                     fontsize=18, color=MUTED)
    axh.text(0.5, -0.04, "exact pitch accuracy on four real phone photos",
             ha="center", va="center", fontsize=9, color=MUTED, style="italic")

    # ===== (a) three-way pitch bars =====
    axa = fig.add_subplot(gs[1, 0])
    x = np.arange(len(pieces)); w = 0.26
    axa.bar(x - w, cnn, w, color=RED, label="old CNN")
    axa.bar(x, geom, w, color=AMBER, label="geometry-only")
    axa.bar(x + w, crnn, w, color=GREEN, label="CRNN+CTC")
    for xi, v in zip(x - w, cnn):
        axa.text(xi, v + 2, str(v), ha="center", fontsize=7.5, color=RED)
    for xi, v in zip(x + w, crnn):
        axa.text(xi, v + 2, str(v), ha="center", fontsize=8, fontweight="bold",
                 color=GREEN)
    axa.set_xticks(x); axa.set_xticklabels(pieces, fontsize=8.5)
    axa.set_ylim(0, 113); axa.set_ylabel("exact pitch (%)")
    axa.set_title("(a)  Exact pitch — three readers", fontsize=11,
                  fontweight="bold")
    axa.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=3,
               fontsize=8, frameon=False)
    axa.grid(axis="y", alpha=0.25)

    # ===== (b) duration bars (new capability) =====
    axb = fig.add_subplot(gs[1, 1])
    dur = [97, 89, 54, 98]
    axb.bar(x, dur, 0.5, color=BLUE)
    for xi, v in zip(x, dur):
        axb.text(xi, v + 2, f"{v}%", ha="center", fontsize=8.5)
    axb.set_xticks(x); axb.set_xticklabels(pieces, fontsize=8.5)
    axb.set_ylim(0, 113); axb.set_ylabel("duration accuracy (%)")
    axb.set_title("(b)  CRNN reads rhythm too\n(baselines produce no durations)",
                  fontsize=10.5, fontweight="bold")
    axb.grid(axis="y", alpha=0.25)

    # ===== (c) domain-gap ablation =====
    axc = fig.add_subplot(gs[2, 0])
    labels = ["clean\nstaves", "phone-degraded\nstaves"]
    xa = np.arange(2); wa = 0.36
    clean_m = [0.030, 0.750]; cam_m = [0.024, 0.028]
    axc.bar(xa - wa / 2, clean_m, wa, color=RED, label="clean-trained")
    axc.bar(xa + wa / 2, cam_m, wa, color=GREEN, label="camera-augmented")
    for xi, v in zip(xa - wa / 2, clean_m):
        axc.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=8, color=RED)
    for xi, v in zip(xa + wa / 2, cam_m):
        axc.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=8, color=GREEN)
    axc.set_xticks(xa); axc.set_xticklabels(labels, fontsize=8.5)
    axc.set_ylim(0, 0.85); axc.set_ylabel("Symbol Error Rate (lower better)")
    axc.set_title("(c)  Camera augmentation closes the domain gap",
                  fontsize=10.5, fontweight="bold")
    axc.legend(loc="upper left", fontsize=8.5, frameon=False)
    axc.grid(axis="y", alpha=0.25)
    axc.annotate("+0.721 gap\n↓\n+0.004", xy=(1, 0.40), ha="center",
                 fontsize=9, color=INK, fontweight="bold")

    # ===== (d) why it works: bullets =====
    axd = fig.add_subplot(gs[2, 1]); axd.axis("off")
    axd.set_xlim(0, 1); axd.set_ylim(0, 1)
    axd.text(0.0, 0.97, "(d)  Why the upgrade works", fontsize=10.5,
             fontweight="bold", color=INK, va="top")
    bullets = [
        ("No segmentation.", "CTC reads the whole staff left-to-right, so a "
         "merged beam can’t delete a note."),
        ("Pitch + duration jointly.", "one model emits the full token "
         "sequence (note-C5_quarter…)."),
        ("Domain-matched training.", "synthetic phone-photo augmentation "
         "(blur, shadow, JPEG, ink-bleed)."),
        ("Compact & local.", "3.1M params, trained on Apple-GPU (MPS) in "
         "minutes, no cloud."),
    ]
    y = 0.84
    for head, body in bullets:
        axd.text(0.02, y, "▸", fontsize=10, color=GREEN, va="top")
        axd.text(0.08, y, head, fontsize=9.5, color=INK, fontweight="bold",
                 va="top")
        axd.text(0.08, y - 0.055, body, fontsize=8.6, color=MUTED, va="top",
                 wrap=True)
        y -= 0.20

    # ===== (e) old vs new pipeline schematic =====
    axe = fig.add_subplot(gs[3, :]); axe.axis("off")
    axe.set_xlim(0, 1); axe.set_ylim(0, 1)
    axe.text(0.5, 0.97, "(e)  Pipeline: cascade vs. end-to-end",
             ha="center", fontsize=11, fontweight="bold", color=INK, va="top")

    # shared front-end
    bw, bh = 0.135, 0.17
    _box(axe, 0.02, 0.55, bw, bh, "rectify\npage", "#34495e", "#34495e")
    _box(axe, 0.18, 0.55, bw, bh, "detect\nstaves", "#34495e", "#34495e")
    _arrow(axe, 0.155, 0.635, 0.18, 0.635)
    axe.text(0.0975, 0.40, "shared classical front-end", ha="center",
             fontsize=7.8, color=MUTED, style="italic")

    # OLD path (top-right)
    oy = 0.74
    axe.text(0.355, oy + 0.10, "OLD", fontsize=9, fontweight="bold", color=RED)
    _box(axe, 0.34, oy, 0.13, 0.13, "remove\nstaves", RED, RED, fs=7.5)
    _box(axe, 0.49, oy, 0.13, 0.13, "segment\ncrops", RED, RED, fs=7.5)
    _box(axe, 0.64, oy, 0.13, 0.13, "CNN\nclassify", RED, RED, fs=7.5)
    _box(axe, 0.79, oy, 0.13, 0.13, "8–22%", "#f5d9d5", RED, tc=RED, fs=9)
    for x0 in [0.315, 0.47, 0.62, 0.77]:
        _arrow(axe, x0, oy + 0.065, x0 + 0.025, oy + 0.065, color=RED)
    axe.text(0.555, oy - 0.07, "every merge / fragment loses a note (cascade)",
             ha="center", fontsize=7.4, color=RED, style="italic")

    # NEW path (bottom-right)
    ny = 0.40
    axe.text(0.355, ny + 0.10, "NEW", fontsize=9, fontweight="bold", color=GREEN)
    _box(axe, 0.34, ny, 0.20, 0.13, "crop staff band", GREEN, GREEN, fs=8)
    _box(axe, 0.58, ny, 0.21, 0.13, "CRNN + CTC\n(image → tokens)", GREEN,
         GREEN, fs=8)
    _box(axe, 0.83, ny, 0.13, 0.13, "90–100%", "#d6efe0", GREEN, tc=GREEN,
         fs=9)
    for x0, x1 in [(0.315, 0.34), (0.54, 0.58), (0.79, 0.83)]:
        _arrow(axe, x0, ny + 0.065, x1, ny + 0.065, color=GREEN)
    axe.text(0.585, ny - 0.07, "no segmentation — reads pitch + duration "
             "directly", ha="center", fontsize=7.4, color=GREEN, style="italic")

    # connectors from front-end to both paths
    _arrow(axe, 0.315, 0.635, 0.335, 0.74 + 0.065, color=MUTED)
    _arrow(axe, 0.315, 0.635, 0.335, 0.40 + 0.065, color=MUTED)

    fig.savefig(os.path.join(OUT, "hero_infographic.png"), dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved final accuracy/hero_infographic.png")

    # ---- copy source figures with clean names ----
    copies = {
        "crnn_comparison.png": "pitch_comparison.png",
        "crnn_ablation.png": "domain_gap_ablation.png",
        "camera_aug_demo_0.png": "camera_augmentation.png",
    }
    for src, dst in copies.items():
        s = os.path.join(RESULTS, src)
        if os.path.isfile(s):
            shutil.copy(s, os.path.join(OUT, dst))
            print(f"copied {dst}")
        else:
            print(f"WARN missing {src}")


if __name__ == "__main__":
    main()
