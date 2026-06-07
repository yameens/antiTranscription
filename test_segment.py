"""
test_segment.py
batch runner for remove_staves_and_segment on all test images.
saves visualizations to results/ and prints a summary table.
"""

import os
import warnings

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rectify import rectify_page
from detect_staves import detect_staves
from segment_symbols import remove_staves_and_segment, visualize_segments


BASE   = os.path.dirname(os.path.abspath(__file__))
SHEET  = os.path.join(BASE, "sheet music")
OUTDIR = os.path.join(BASE, "results")
os.makedirs(OUTDIR, exist_ok=True)

IMAGES = [
    "yankeeDoodle",
    "twinkleTwinkleLittleStar",
    "maryHadLittleLamb",
    "cs131",
]

DEFAULT_STAVE_PARAMS = dict(
    hough_threshold     = 100,
    min_line_length     = 300,
    max_line_gap        = 40,
    angle_tolerance_deg = 2.0,
    cluster_tol         = 3,
    morph_close_width   = 0,
)

RELAXED_STAVE_PARAMS = dict(
    hough_threshold     = 60,
    min_line_length     = 150,
    max_line_gap        = 60,
    angle_tolerance_deg = 3.0,
    cluster_tol         = 5,
    morph_close_width   = 60,
)

results = []

for name in IMAGES:
    jpg = os.path.join(SHEET, f"{name}.jpg")
    print(f"\n{'='*60}")
    print(f"Processing: {name}")

    rectified = rectify_page(jpg)

    # detect staves (try default, fall back to relaxed)
    staves, spacing = None, None
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        try:
            staves, spacing = detect_staves(rectified, **DEFAULT_STAVE_PARAMS)
        except ValueError:
            pass

    if staves is None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                staves, spacing = detect_staves(rectified, **RELAXED_STAVE_PARAMS)
            except ValueError as exc:
                print(f"  stave detection failed: {exc}")
                results.append((name, 0, 0, "FAIL"))
                continue

    print(f"  staves detected: {len(staves)}  spacing: {spacing:.1f} px")

    symbols, cleaned = remove_staves_and_segment(rectified, staves, spacing)
    print(f"  symbols found:   {len(symbols)}")

    # save the staff-removed binary
    cleaned_path = os.path.join(OUTDIR, f"{name}_cleaned.png")
    cv2.imwrite(cleaned_path, cleaned)

    # save the bounding-box overlay
    vis = visualize_segments(rectified, symbols, staff_lines=staves)
    vis_path = os.path.join(OUTDIR, f"{name}_segments.png")
    cv2.imwrite(vis_path, vis)

    # side-by-side figure for the milestone
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    fig.suptitle(name, fontsize=13, fontweight="bold")

    axes[0].imshow(cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB))
    axes[0].set_title("(a)  Rectified input")
    axes[0].axis("off")

    axes[1].imshow(cleaned, cmap="gray")
    axes[1].set_title("(b)  Staff-removed binary")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"(c)  {len(symbols)} symbols detected")
    axes[2].axis("off")

    plt.tight_layout()
    fig_path = os.path.join(OUTDIR, f"{name}_segments_compare.png")
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"  saved: {os.path.relpath(fig_path, BASE)}")
    results.append((name, len(staves), len(symbols), "PASS"))


SEP = "-" * 65
print(f"\n{SEP}")
print(f"{'IMAGE':<28} {'STAVES':>6} {'SYMBOLS':>8}  STATUS")
print(SEP)
for name, n_staves, n_sym, status in results:
    print(f"{name:<28} {n_staves:>6} {n_sym:>8}  {status}")
print(SEP)
