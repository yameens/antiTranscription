"""
make_final_graphs.py
one-off comparison graphs, each isolating a single trait, in Times New Roman.
outputs individual PNGs to 'final accuracy/final graphs/'.

run:  python make_final_graphs.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

# ---- Times New Roman everywhere ----
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.titleweight": "bold",
    "figure.dpi": 200,
    "savefig.dpi": 200,
})

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "final accuracy", "final graphs")
os.makedirs(OUT, exist_ok=True)

# ---- palette (consistent semantics across all graphs) ----
OLD, GEO, NEW, BLUE, INK, MUTED, GRID = (
    "#c0392b", "#c98a0e", "#1f8b4c", "#2c6e8f", "#1b2530", "#6b7886", "#dfe5ea")

pieces = ["Yankee\nDoodle", "Twinkle\nTwinkle", "Mary Had\na Lamb", "CS 131"]
cnn = [14, 10, 8, 8]
geo = [86, 86, 69, 60]
crnn = [100, 90, 100, 90]
crnn_dur = [97, 89, 54, 98]
geo_pm1 = [89, 90, 69, 83]
crnn_pm1 = [100, 90, 100, 90]


def _style(ax, ytitle, title, subtitle=None, ymax=113):
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ytitle, fontsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#aab4bd")
    ax.tick_params(colors=INK, labelsize=11)
    ax.yaxis.grid(True, color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=16, color=INK, pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=10.5, color=MUTED, style="italic")


def _labels(ax, xs, ys, color, fs=10, fmt="{:.0f}", dy=2, bold=False):
    for x, y in zip(xs, ys):
        ax.text(x, y + dy, fmt.format(y), ha="center", fontsize=fs, color=color,
                fontweight="bold" if bold else "normal")


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", os.path.relpath(p, BASE))


# === 1. EXACT PITCH — three readers ===
def g_exact_pitch():
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    x = np.arange(len(pieces)); w = 0.26
    ax.bar(x - w, cnn, w, color=OLD, label="Old CNN  (segment + classify)")
    ax.bar(x, geo, w, color=GEO, label="Geometry-only")
    ax.bar(x + w, crnn, w, color=NEW, label="CRNN + CTC  (this work)")
    _labels(ax, x - w, cnn, OLD, 9)
    _labels(ax, x, geo, "#9c6f0a", 9)
    _labels(ax, x + w, crnn, NEW, 10, bold=True)
    ax.set_xticks(x); ax.set_xticklabels(pieces)
    _style(ax, "Exact pitch accuracy (%)",
           "Exact Pitch Accuracy on Real Phone Photos",
           "the end-to-end CRNN wins on every piece")
    ax.legend(fontsize=10.5, loc="lower center", bbox_to_anchor=(0.5, 1.10),
              ncol=1, frameon=False, handlelength=1.3)
    save(fig, "01_exact_pitch_comparison.png")


# === 2. DURATION (rhythm) — a trait only the CRNN has ===
def g_duration():
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    x = np.arange(len(pieces))
    bars = ax.bar(x, crnn_dur, 0.55, color=BLUE)
    bars[2].set_color("#b9512e")  # flag the weak one (Mary)
    _labels(ax, x, crnn_dur, INK, 11, fmt="{:.0f}%", dy=2)
    ax.axhline(np.mean(crnn_dur), color=MUTED, ls="--", lw=1.2)
    ax.text(len(pieces) - 0.5, np.mean(crnn_dur) + 2,
            f"mean {np.mean(crnn_dur):.0f}%", ha="right", color=MUTED, fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(pieces)
    _style(ax, "Duration accuracy (%)",
           "Rhythm: a Capability the Baselines Lack",
           "the CRNN reads note durations; geometry & the old CNN do not")
    save(fig, "02_duration_accuracy.png")


# === 3. DOMAIN-GAP ROBUSTNESS — SER, clean vs phone ===
def g_domain_gap():
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    cond = ["Clean staves", "Phone-degraded staves"]
    x = np.arange(2); w = 0.36
    clean_m = [0.030, 0.750]; cam_m = [0.024, 0.028]
    ax.bar(x - w / 2, clean_m, w, color=OLD, label="Clean-trained")
    ax.bar(x + w / 2, cam_m, w, color=NEW, label="Camera-augmented (this work)")
    _labels(ax, x - w / 2, clean_m, OLD, 10, fmt="{:.3f}", dy=0.012)
    _labels(ax, x + w / 2, cam_m, NEW, 10, fmt="{:.3f}", dy=0.012, bold=True)
    ax.set_xticks(x); ax.set_xticklabels(cond)
    _style(ax, "Symbol Error Rate  (lower is better)",
           "Robustness to the Clean-to-Real Domain Gap",
           "camera augmentation cuts the gap from +0.721 to +0.004", ymax=0.85)
    ax.annotate("", xy=(1 - w / 2, 0.72), xytext=(1 + w / 2, 0.10),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.1))
    ax.text(1.18, 0.42, "99.4%\nsmaller\ngap", ha="center", va="center",
            fontsize=10.5, color=NEW, fontweight="bold")
    ax.legend(fontsize=10.5, loc="upper left", frameon=False)
    save(fig, "03_domain_gap_robustness.png")


# === 4. HEADLINE IMPROVEMENT — average exact pitch ===
def g_headline():
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    methods = ["Old CNN", "Geometry-only", "CRNN + CTC"]
    avgs = [np.mean(cnn), np.mean(geo), np.mean(crnn)]
    cols = [OLD, GEO, NEW]
    y = np.arange(len(methods))[::-1]
    ax.barh(y, avgs, 0.6, color=cols)
    for yi, v in zip(y, avgs):
        ax.text(v + 1.5, yi, f"{v:.0f}%", va="center", fontsize=13,
                color=INK, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(methods, fontsize=12.5)
    ax.set_xlim(0, 108)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#aab4bd")
    ax.xaxis.grid(True, color=GRID, linewidth=0.9); ax.set_axisbelow(True)
    ax.tick_params(labelsize=11)
    ax.set_xlabel("Average exact pitch across four pieces (%)", fontsize=12)
    ax.set_title("Headline: 10%  →  95% Exact Pitch", fontsize=17,
                 color=INK, pad=24)
    ax.text(0.5, 1.02, "a 9.5× improvement over the original CNN reader",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=10.5,
            color=MUTED, style="italic")
    save(fig, "04_headline_improvement.png")


# === 5. PITCH PRECISION — exact vs within-one-step ===
def g_tolerance():
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    x = np.arange(len(pieces)); w = 0.38
    # geometry exact vs CRNN exact, with ±1 as faint extension
    ax.bar(x - w / 2, geo_pm1, w, color=GEO, alpha=0.32)
    ax.bar(x - w / 2, geo, w, color=GEO, label="Geometry  (exact)")
    ax.bar(x + w / 2, crnn_pm1, w, color=NEW, alpha=0.32)
    ax.bar(x + w / 2, crnn, w, color=NEW, label="CRNN  (exact)")
    _labels(ax, x - w / 2, geo, "#7a5708", 9)
    _labels(ax, x + w / 2, crnn, NEW, 10, bold=True)
    ax.set_xticks(x); ax.set_xticklabels(pieces)
    _style(ax, "Pitch accuracy (%)",
           "Pitch Precision: Exact vs. Within One Step",
           "solid = exact, faded cap = within ±1 diatonic step")
    ax.legend(fontsize=10.5, loc="lower center", bbox_to_anchor=(0.5, 1.10),
              ncol=2, frameon=False)
    save(fig, "05_pitch_precision.png")


# === 6. CAPABILITY MATRIX — which method has which trait ===
def g_capability():
    traits = ["Reads pitch", "Reads duration\n(rhythm)",
              "Segmentation-free", "Robust to\nphone photos",
              "Reads whole\nstaff at once"]
    methods = ["Old CNN", "Geometry-only", "CRNN + CTC"]
    # 2 = yes, 1 = partial, 0 = no
    M = np.array([
        [1, 2, 2],   # reads pitch  (old CNN end-to-end poor; geometry & CRNN strong)
        [1, 0, 2],   # duration
        [0, 2, 2],   # segmentation-free
        [0, 2, 2],   # robust
        [0, 0, 2],   # whole staff
    ])
    cmap = {0: ("#f4d9d4", OLD, "No"),
            1: ("#fbf0d6", "#b07d10", "Partial"),
            2: ("#d8efe1", NEW, "Yes")}
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    nrows, ncols = M.shape
    for i in range(nrows):
        for j in range(ncols):
            fc, tc, sym = cmap[M[i, j]]
            ax.add_patch(plt.Rectangle((j, nrows - 1 - i), 0.94, 0.94,
                         facecolor=fc, edgecolor="white", linewidth=3))
            ax.text(j + 0.47, nrows - 1 - i + 0.47, sym, ha="center",
                    va="center", fontsize=13.5, color=tc, fontweight="bold")
    ax.set_xlim(-0.05, ncols); ax.set_ylim(-0.05, nrows + 0.05)
    ax.set_xticks([j + 0.47 for j in range(ncols)])
    ax.set_xticklabels(methods, fontsize=12.5, fontweight="bold")
    ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top")
    ax.set_yticks([nrows - 1 - i + 0.47 for i in range(nrows)])
    ax.set_yticklabels(traits, fontsize=11.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Capability Matrix: One Model, Every Trait", fontsize=16,
                 color=INK, pad=34)
    fig.text(0.5, 0.02,
             "green = full capability     amber = partial     red = none",
             ha="center", fontsize=10.5, color=MUTED)
    save(fig, "06_capability_matrix.png")


if __name__ == "__main__":
    g_exact_pitch()
    g_duration()
    g_domain_gap()
    g_headline()
    g_tolerance()
    g_capability()
    print("\nall graphs written to:", os.path.relpath(OUT, BASE))
