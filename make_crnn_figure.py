"""
make_crnn_figure.py
results figure for the end-to-end CRNN: a three-way pitch comparison plus the
durations only the CRNN can produce.  numbers are the measured values from
test_crnn.py on the four real phone photos (see README / test_crnn.py to
reproduce).  saves to results/crnn_comparison.png.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "results")
os.makedirs(OUT, exist_ok=True)

pieces = ["Yankee\nDoodle", "Twinkle\nTwinkle", "Mary Had\na Lamb", "CS 131"]
# exact-pitch % (from test_crnn.py)
cnn      = [14, 10, 8, 8]      # old segment+CNN baseline (project report Table 3)
geometry = [86, 86, 69, 60]   # geometry-only reader
crnn     = [100, 90, 100, 90] # end-to-end CRNN+CTC (camera-augmented)
crnn_dur = [97, 89, 54, 98]   # CRNN duration accuracy

x = np.arange(len(pieces))
w = 0.26

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5),
                               gridspec_kw={"width_ratios": [1.55, 1.0]})

# ---- left: three-way exact-pitch comparison ----
axL.bar(x - w, cnn, w, label="segment + CNN (baseline)", color="#c0392b")
axL.bar(x,     geometry, w, label="geometry-only", color="#e6a817")
axL.bar(x + w, crnn, w, label="CRNN + CTC (this work)", color="#1f8b4c")
for xi, v in zip(x - w, cnn):
    axL.text(xi, v + 2, str(v), ha="center", fontsize=8, color="#c0392b")
for xi, v in zip(x, geometry):
    axL.text(xi, v + 2, str(v), ha="center", fontsize=8, color="#9c6f0a")
for xi, v in zip(x + w, crnn):
    axL.text(xi, v + 2, str(v), ha="center", fontsize=8, fontweight="bold",
             color="#1f8b4c")
axL.set_xticks(x); axL.set_xticklabels(pieces, fontsize=9)
axL.set_ylabel("exact pitch accuracy (%)")
axL.set_ylim(0, 112)
axL.set_title("(a)  Exact pitch on real phone photos", fontsize=11, fontweight="bold")
axL.legend(loc="lower center", bbox_to_anchor=(0.5, 1.10), ncol=3, fontsize=8.5,
           framealpha=0.95)
axL.grid(axis="y", alpha=0.3)

# ---- right: CRNN durations (the new capability) ----
axR.bar(x, crnn_dur, 0.5, color="#2c7fb8")
for xi, v in zip(x, crnn_dur):
    axR.text(xi, v + 1.5, f"{v}%", ha="center", fontsize=9)
axR.set_xticks(x); axR.set_xticklabels(pieces, fontsize=9)
axR.set_ylabel("duration accuracy (%)")
axR.set_ylim(0, 108)
axR.set_title("(b)  CRNN duration accuracy\n(geometry/CNN-baseline cannot read pitch+duration jointly)",
              fontsize=10, fontweight="bold")
axR.grid(axis="y", alpha=0.3)

fig.suptitle("End-to-end CRNN+CTC vs. segment-and-classify and geometry-only readers",
             fontsize=12, fontweight="bold", y=1.02)
fig.tight_layout()
out = os.path.join(OUT, "crnn_comparison.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {os.path.relpath(out, BASE)}")
