"""
make_ablation_figure.py
domain-gap ablation figure: clean-trained vs camera-augmented CRNN, each
evaluated on clean and on phone-photo-degraded validation staves.  numbers from
eval_ablation.py.  saves results/crnn_ablation.png.

the story: a clean-trained sequence model is fine on clean engravings (SER 0.03)
but collapses on degraded staves (SER 0.75) -- the same clean-to-real gap the
old CNN showed.  training with synthetic camera augmentation closes it almost
entirely (0.72 -> 0.004).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "results")
os.makedirs(OUT, exist_ok=True)

# from eval_ablation.py
clean_model = {"clean": 0.030, "camera": 0.750}
cam_model = {"clean": 0.024, "camera": 0.028}

labels = ["evaluated on\nclean staves", "evaluated on\nphone-degraded staves"]
x = np.arange(len(labels))
w = 0.35

fig, ax = plt.subplots(figsize=(7.8, 5))
b1 = ax.bar(x - w / 2, [clean_model["clean"], clean_model["camera"]], w,
            label="clean-trained CRNN", color="#c0392b")
b2 = ax.bar(x + w / 2, [cam_model["clean"], cam_model["camera"]], w,
            label="camera-augmented CRNN (this work)", color="#1f8b4c")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
            f"{b.get_height():.3f}", ha="center", fontsize=9)

ax.annotate("domain gap\ncollapses", xy=(1 - w / 2, 0.750), xytext=(0.30, 0.62),
            fontsize=9, color="#c0392b", ha="center",
            arrowprops=dict(arrowstyle="->", color="#c0392b"))
ax.annotate("gap closed\n(0.72 -> 0.004)", xy=(1 + w / 2, 0.028),
            xytext=(1.42, 0.22), fontsize=9, color="#1f8b4c", ha="center",
            arrowprops=dict(arrowstyle="->", color="#1f8b4c"))

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Symbol Error Rate (lower = better)")
ax.set_ylim(0, 0.85)
ax.set_title("Camera augmentation closes the clean-to-real domain gap\n"
             "(CRNN+CTC, held-out PrIMuS treble staves)",
             fontsize=11, fontweight="bold")
ax.legend(loc="upper left", fontsize=9.5)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
out = os.path.join(OUT, "crnn_ablation.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {os.path.relpath(out, BASE)}")
