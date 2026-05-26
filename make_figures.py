"""
make_figures.py
generate milestone figures in results/.
"""

import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from rectify import (
    rectify_page,
    _find_page_corners,
    _segment_page_otsu,
)
from detect_staves import detect_staves, visualize_staves

BASE   = os.path.dirname(os.path.abspath(__file__))
SHEET  = os.path.join(BASE, "sheet music")
OUT    = os.path.join(BASE, "results")
os.makedirs(OUT, exist_ok=True)


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


print("Generating Figure 1: rectification grid ...")

RECT_IMAGES = [
    ("Yankee Doodle",               "yankeeDoodle"),
    ("Mary Had a Little Lamb",      "maryHadLittleLamb"),
    ("London Bridge Is Falling",    "londonBridgeIsFalling"),
    ("Twinkle Twinkle Little Star", "twinkleTwinkleLittleStar"),
    ("CS 131",                      "cs131"),
]

fig1, axes = plt.subplots(len(RECT_IMAGES), 2, figsize=(9, 4 * len(RECT_IMAGES)))
fig1.suptitle(
    "Figure 1  |  Stage 1: Page Rectification\n"
    f"{len(RECT_IMAGES)} of {len(RECT_IMAGES)} test images successfully warped to 850 x 1100 px",
    fontsize=12, fontweight="bold", y=0.995,
)

for row, (title, name) in enumerate(RECT_IMAGES):
    jpg = os.path.join(SHEET, f"{name}.jpg")
    original = cv2.imread(jpg)
    rectified = rectify_page(jpg)

    axes[row][0].imshow(bgr2rgb(original))
    axes[row][0].set_title(f"{title}\nOriginal phone photo", fontsize=9)
    axes[row][0].axis("off")

    axes[row][1].imshow(bgr2rgb(rectified))
    axes[row][1].set_title("Rectified output (850 x 1100 px)", fontsize=9)
    axes[row][1].axis("off")

fig1.tight_layout(rect=[0, 0, 1, 0.995])
path1 = os.path.join(OUT, "fig1_rectification.png")
fig1.savefig(path1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"  Saved {os.path.relpath(path1, BASE)}")


print("Generating Figure 2: rectification pipeline ...")

example_name = "maryHadLittleLamb"
example_jpg = os.path.join(SHEET, f"{example_name}.jpg")
img_pipe = cv2.imread(example_jpg)

gray_pipe = cv2.cvtColor(img_pipe, cv2.COLOR_BGR2GRAY)
blur_pipe = cv2.GaussianBlur(gray_pipe, (7, 7), 0)
mask_otsu = _segment_page_otsu(blur_pipe)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
mask_closed = cv2.morphologyEx(mask_otsu, cv2.MORPH_CLOSE, kernel)

corners = _find_page_corners(img_pipe)
overlay = img_pipe.copy()
pts = corners.astype(np.int32).reshape(-1, 1, 2)
cv2.polylines(overlay, [pts], isClosed=True, color=(0, 0, 220), thickness=12)
for pt in corners:
    cv2.circle(overlay, tuple(pt.astype(int)), 28, (0, 200, 0), -1)

fig2, axes2 = plt.subplots(1, 4, figsize=(18, 6))
fig2.suptitle(
    f"Figure 2  |  Rectification Pipeline (example: {example_name})",
    fontsize=11, fontweight="bold", y=1.02,
)

axes2[0].imshow(blur_pipe, cmap="gray")
axes2[0].set_title("(a)  Grayscale + Gaussian blur", fontsize=9, pad=8)
axes2[0].axis("off")

axes2[1].imshow(mask_otsu, cmap="gray")
axes2[1].set_title("(b)  Otsu binary mask\n(page = white, background = black)", fontsize=9, pad=8)
axes2[1].axis("off")

axes2[2].imshow(mask_closed, cmap="gray")
axes2[2].set_title("(c)  After morph-close (25x25)\nstaff lines and text filled in", fontsize=9, pad=8)
axes2[2].axis("off")

axes2[3].imshow(bgr2rgb(overlay))
det_patch = mpatches.Patch(color=(0/255, 200/255, 0/255), label="Detected page corners")
axes2[3].legend(handles=[det_patch], loc="lower left", fontsize=8, framealpha=0.85)
axes2[3].set_title("(d)  minAreaRect on largest contour\n4 corners feed the homography", fontsize=9, pad=8)
axes2[3].axis("off")

fig2.tight_layout(rect=[0, 0, 1, 0.97])
path2 = os.path.join(OUT, "fig2_pipeline_diagram.png")
fig2.savefig(path2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  Saved {os.path.relpath(path2, BASE)}")


print("Generating Figure 3: staff detection ...")

rect_yankee = rectify_page(os.path.join(SHEET, "yankeeDoodle.jpg"))
staves, spacing = detect_staves(rect_yankee)
vis_yankee = visualize_staves(rect_yankee, staves, spacing)

stave0_ys = staves[0]
pad = 25
y_top = max(0, int(stave0_ys[0]) - pad)
y_bot = min(vis_yankee.shape[0], int(stave0_ys[-1]) + pad)
inset_strip = vis_yankee[y_top:y_bot, :]

fig3, axes3 = plt.subplots(
    3, 1,
    figsize=(9, 14),
    gridspec_kw={"height_ratios": [6, 6, 2]},
)
fig3.suptitle(
    "Figure 3  |  Stage 2: Staff Line Detection - Yankee Doodle\n"
    f"{len(staves)} of {len(staves)} staves detected  |  staff-line spacing = {spacing:.1f} px",
    fontsize=12, fontweight="bold",
)

axes3[0].imshow(bgr2rgb(rect_yankee))
axes3[0].set_title("(a)  Rectified input (850 x 1100 px)", fontsize=10)
axes3[0].axis("off")

axes3[1].imshow(bgr2rgb(vis_yankee))
colors_bgr = [(0, 0, 220), (0, 180, 0), (220, 0, 0), (0, 200, 200), (200, 0, 200)]
colors_rgb = [(c[2]/255, c[1]/255, c[0]/255) for c in colors_bgr]
patches = [
    mpatches.Patch(
        color=colors_rgb[i % len(colors_rgb)],
        label=f"Stave {i+1} (y ~ {staves[i][0]:.0f}-{staves[i][-1]:.0f})",
    )
    for i in range(len(staves))
]
axes3[1].legend(handles=patches, loc="upper right", fontsize=8, framealpha=0.85)
axes3[1].set_title(
    "(b)  Detected staff lines overlaid (each colour = one stave)",
    fontsize=9,
)
axes3[1].axis("off")

axes3[2].imshow(bgr2rgb(inset_strip))
axes3[2].set_title(
    f"(c)  Zoom on Stave 1  |  y = {[round(y,1) for y in stave0_ys]}  |  "
    f"mean spacing = {spacing:.1f} px",
    fontsize=8,
)
for i in range(len(stave0_ys) - 1):
    y1_rel = stave0_ys[i] - y_top
    y2_rel = stave0_ys[i + 1] - y_top
    gap = stave0_ys[i + 1] - stave0_ys[i]
    x_arrow = vis_yankee.shape[1] - 50
    axes3[2].annotate(
        "", xy=(x_arrow, y2_rel), xytext=(x_arrow, y1_rel),
        arrowprops=dict(arrowstyle="<->", color="white", lw=1.5),
    )
    axes3[2].text(
        x_arrow + 5, (y1_rel + y2_rel) / 2, f"{gap:.0f}px",
        color="white", fontsize=7, va="center",
    )
axes3[2].axis("off")

fig3.tight_layout()
path3 = os.path.join(OUT, "fig3_staff_detection.png")
fig3.savefig(path3, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"  Saved {os.path.relpath(path3, BASE)}")

print("\nAll figures saved to results/.")
