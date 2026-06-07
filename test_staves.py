"""
test_staves.py
batch runner for detect_staves on all test images.
"""

import os
import warnings

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rectify import rectify_page
from detect_staves import detect_staves, visualize_staves


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

DEFAULT_PARAMS = dict(
    hough_threshold     = 100,
    min_line_length     = 300,
    max_line_gap        = 40,
    angle_tolerance_deg = 2.0,
    cluster_tol         = 3,
    morph_close_width   = 0,
)

RELAXED_PARAMS = dict(
    hough_threshold     = 60,
    min_line_length     = 150,
    max_line_gap        = 60,
    angle_tolerance_deg = 3.0,
    cluster_tol         = 5,
    morph_close_width   = 60,
)


def run_one(name: str) -> dict:
    jpg_path = os.path.join(SHEET, f"{name}.jpg")
    print(f"\n{'='*60}")
    print(f"Processing: {name}.jpg")

    try:
        rectified = rectify_page(jpg_path)
    except Exception as exc:
        print(f"  RECTIFY FAIL: {exc}")
        return dict(name=name, staves=0, spacing="--", params_used="--",
                    status="FAIL (rectify)", note=str(exc))

    print(f"  Rectified shape: {rectified.shape}")

    params_label = "default"
    staves = None
    spacing = None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            staves, spacing = detect_staves(rectified, **DEFAULT_PARAMS)
        except ValueError as exc:
            default_err = str(exc)
            print(f"  Default params FAILED: {default_err}")
        else:
            for w in caught:
                print(f"  WARNING: {w.message}")

    if staves is None:
        print("  Retrying with relaxed params ...")
        params_label = "relaxed"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                staves, spacing = detect_staves(rectified, **RELAXED_PARAMS)
            except ValueError as exc:
                relaxed_err = str(exc)
                print(f"  Relaxed params ALSO FAILED: {relaxed_err}")
                return dict(name=name, staves=0, spacing="--",
                            params_used="both failed",
                            status="FAIL (detect)",
                            note=f"default: {default_err} | relaxed: {relaxed_err}")
            else:
                for w in caught:
                    print(f"  WARNING (relaxed): {w.message}")

    print(f"  Staves detected : {len(staves)}  (params: {params_label})")
    print(f"  Staff spacing   : {spacing:.1f} px")
    for i, sy in enumerate(staves):
        print(f"    Stave {i+1}: y = {[round(y,1) for y in sy]}")

    vis = visualize_staves(rectified, staves, spacing)
    out_path = os.path.join(OUTDIR, f"{name}_staves.png")
    cv2.imwrite(out_path, vis)
    print(f"  Saved: {os.path.relpath(out_path, BASE)}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    fig.suptitle(name, fontsize=13, fontweight="bold")
    axes[0].imshow(cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Rectified (input)")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        f"Staves detected: {len(staves)}  spacing={spacing:.1f}px  ({params_label})"
    )
    axes[1].axis("off")
    plt.tight_layout()
    fig_path = os.path.join(OUTDIR, f"{name}_staves_compare.png")
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.relpath(fig_path, BASE)}")

    note = ""
    if params_label == "relaxed":
        note = "Required relaxed params"

    return dict(
        name=name,
        staves=len(staves),
        spacing=f"{spacing:.1f}",
        params_used=params_label,
        status="PASS",
        note=note,
    )


results = [run_one(name) for name in IMAGES]

SEP = "-" * 85
print(f"\n{SEP}")
print(f"{'IMAGE':<28} {'STAVES':>6} {'SPACING':>9} {'PARAMS':>10}  STATUS / NOTE")
print(SEP)
for r in results:
    note_str = f"  {r['note']}" if r['note'] else ""
    print(
        f"{r['name']:<28} {str(r['staves']):>6} {str(r['spacing']):>9} "
        f"{r['params_used']:>10}  {r['status']}{note_str}"
    )
print(SEP)

passed  = sum(1 for r in results if r["status"] == "PASS")
relaxed = sum(1 for r in results if r["params_used"] == "relaxed")
print(f"\nResult : {passed}/{len(results)} images detected successfully "
      f"({relaxed} required relaxed Hough parameters).")
