"""
make_geometry_figure.py
visualize the geometry-only pitch reader on yankeeDoodle:
  left  - rectified photo with each detected notehead circled and pitch-labelled
          (green = exact vs ground truth, orange = within one step, red = wrong)
  right - melodic-contour chart, detected vs ground-truth pitch in reading order

saves to results/ and to the 'final final/' folder requested for the paper.
"""

import os
import warnings

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

from pitch_geometry import read_pitches, align_pitches, pitch_step, _NOTE_NAMES
from sheet_utils import parse_ground_truth, NOTE_DIR

BASE   = os.path.dirname(os.path.abspath(__file__))
SHEET  = os.path.join(BASE, "sheet music")
OUTS   = [os.path.join(BASE, "results"), os.path.join(BASE, "final final")]
for d in OUTS:
    os.makedirs(d, exist_ok=True)

NAME = "yankeeDoodle"


def main() -> None:
    rect, staves, spacing, nh = read_pitches(os.path.join(SHEET, f"{NAME}.jpg"))
    gt = [p for p, _ in parse_ground_truth(os.path.join(NOTE_DIR, f"{NAME}.txt")) if p]
    det = [n.pitch for n in nh]

    pairs = align_pitches(det, gt)
    # per-detection correctness from the alignment
    det_status = {}   # det_idx -> "exact" | "near" | "wrong"
    exact = near = 0
    for di, gj in pairs:
        if di is None or gj is None:
            if di is not None:
                det_status[di] = "wrong"
            continue
        sd, sg = pitch_step(det[di]), pitch_step(gt[gj])
        if det[di] == gt[gj]:
            det_status[di] = "exact"; exact += 1; near += 1
        elif sd is not None and sg is not None and abs(sd - sg) <= 1:
            det_status[di] = "near"; near += 1
        else:
            det_status[di] = "wrong"

    pct_exact = 100 * exact / len(gt)
    pct_near  = 100 * near / len(gt)

    # ---- left panel: annotated photo --------------------------------------
    vis = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB).copy()
    h, w = vis.shape[:2]
    for stave in staves:
        for y in stave:
            cv2.line(vis, (0, int(y)), (w - 1, int(y)), (205, 205, 205), 1)
    colors = {"exact": (20, 160, 40), "near": (230, 150, 0), "wrong": (220, 40, 40)}
    for i, n in enumerate(nh):
        c = colors.get(det_status.get(i, "wrong"))
        cv2.circle(vis, (int(n.cx), int(n.cy)), 9, c, 2)
        cv2.putText(vis, n.pitch, (int(n.cx) - 12, int(n.cy) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1, cv2.LINE_AA)

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(16, 7.2), gridspec_kw={"width_ratios": [1.04, 1.0]})
    fig.suptitle(
        f"Geometry-only pitch reading - Yankee Doodle\n"
        f"{exact}/{len(gt)} exact ({pct_exact:.0f}%)   "
        f"{near}/{len(gt)} within one step ({pct_near:.0f}%)",
        fontsize=13, fontweight="bold", y=0.99)

    axL.imshow(vis)
    axL.set_title("(a)  Detected noteheads + treble-clef pitch\n"
                  "green = exact, orange = +-1 step, red = wrong", fontsize=10)
    axL.axis("off")

    # ---- right panel: melodic contour (detector placed at aligned indices) -
    gt_steps  = [pitch_step(p) for p in gt]
    det_aligned = [np.nan] * len(gt)   # detector pitch at each GT slot
    for di, gj in pairs:
        if di is not None and gj is not None:
            det_aligned[gj] = pitch_step(det[di])

    axR.plot(range(len(gt_steps)), gt_steps, "o-", color="#1f5fbf",
             label="ground truth", linewidth=2, markersize=6, alpha=0.9)
    axR.plot(range(len(det_aligned)), det_aligned, "s--", color="#d11a1a",
             label="geometry detector", linewidth=1.6, markersize=5, alpha=0.85)
    # mark undetected notes (alignment gaps) with a hollow marker on the GT line
    misses = [j for j, v in enumerate(det_aligned) if np.isnan(v)]
    if misses:
        axR.scatter(misses, [gt_steps[j] for j in misses], s=90,
                    facecolors="none", edgecolors="#d11a1a", linewidths=1.6,
                    zorder=5, label="missed (not detected)")

    valid = [v for v in det_aligned if not np.isnan(v)]
    lo = min(min(gt_steps), min(valid)) - 1
    hi = max(max(gt_steps), max(valid)) + 1
    ticks = list(range(lo, hi + 1))
    axR.set_yticks(ticks)
    axR.set_yticklabels([f"{_NOTE_NAMES[t % 7]}{t // 7}" for t in ticks], fontsize=8)
    axR.set_xlabel("note index (reading order)")
    axR.set_ylabel("pitch")
    axR.set_title("(b)  Melodic contour: detector vs ground truth", fontsize=10)
    axR.grid(True, alpha=0.3)
    axR.legend(loc="lower right", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for d in OUTS:
        out = os.path.join(d, f"{NAME}_geometry_pitch.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"saved {os.path.relpath(out, BASE)}")
    plt.close(fig)

    print(f"\n{NAME}: detected {len(det)} noteheads vs {len(gt)} GT notes")
    print(f"  exact:        {exact}/{len(gt)}  ({pct_exact:.0f}%)")
    print(f"  within 1 step:{near}/{len(gt)}  ({pct_near:.0f}%)")


if __name__ == "__main__":
    main()
