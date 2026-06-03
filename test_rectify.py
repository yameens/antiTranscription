"""
test_rectify.py
rigorous real-photo test harness for rectify_page.

for each *.jpg in sheet music/, this script:
  - runs the full detection cascade via rectify_page_debug
  - saves a 3-panel figure: original+corners overlay, binary mask, rectified output
  - computes two objective metrics on the rectified page (no ground truth needed):
      staff_slant_deg  -- mean absolute slope of near-horizontal dark runs
                          (target < 1.0 deg after good rectification)
      border_bg_frac   -- fraction of a 30-px inner border that is background
                          (tight corners -> near 0; background bleed -> higher)
  - prints a summary table

usage:
    python3 test_rectify.py                    # test all 5 images
    python3 test_rectify.py my_angled_photo.jpg  # test one specific image
"""

import os
import sys
import glob
import math

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from rectify import rectify_page_debug, OUT_W, OUT_H

BASE       = os.path.dirname(os.path.abspath(__file__))
SHEET_DIR  = os.path.join(BASE, "sheet music")
RESULTS    = os.path.join(BASE, "results", "adjusted_rectified")
os.makedirs(RESULTS, exist_ok=True)

# ─── corner overlay colors per cascade method ────────────────────────────────
_METHOD_COLOR = {
    "hough":   (0,   200,  60),   # green  = best
    "polydp":  (0,   160, 255),   # orange = ok
    "minrect": (0,    40, 220),   # red    = fallback (no perspective correction)
}

PASS_SLANT_DEG  = 1.5   # staff lines within this many degrees = pass
PASS_BORDER_BG  = 0.10  # border background fraction below this = pass


# ─── metric 1: staff-line slant ──────────────────────────────────────────────

def _measure_staff_slant(rectified_bgr: np.ndarray) -> float:
    """
    detect near-horizontal dark runs on the rectified page and return the
    mean absolute slope (in degrees).  a perfect rectification has slant ≈ 0.

    approach:
      1. binarize with Otsu (dark ink = 255)
      2. horizontal morphological close to join short runs
      3. find connected components with high aspect ratio (wide lines)
      4. fit a line to each and record abs(angle)
    """
    gray = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # keep only horizontal runs (close horizontally, open vertically)
    h_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    horiz  = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kern)

    n, _, stats, _ = cv2.connectedComponentsWithStats(horiz, connectivity=8)
    slopes = []
    for lbl in range(1, n):
        w = int(stats[lbl, cv2.CC_STAT_WIDTH])
        h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        # only long thin components that look like staff lines
        if w < OUT_W * 0.15 or h > 8 or area < 50:
            continue
        x0 = int(stats[lbl, cv2.CC_STAT_LEFT])
        y0 = int(stats[lbl, cv2.CC_STAT_TOP])
        roi = horiz[y0: y0 + h, x0: x0 + w]
        pts = np.column_stack(np.where(roi > 0))  # (row, col)
        if len(pts) < 10:
            continue
        pts_xy = pts[:, [1, 0]].astype(np.float32)  # (x, y)
        vx, vy, _, _ = cv2.fitLine(pts_xy, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        angle_deg = abs(math.degrees(math.atan2(float(vy), float(vx))))
        if angle_deg > 90:
            angle_deg = 180 - angle_deg
        slopes.append(angle_deg)

    return float(np.mean(slopes)) if slopes else 0.0


# ─── metric 2: border background fraction ────────────────────────────────────

def _measure_border_bg(rectified_bgr: np.ndarray, band: int = 30) -> float:
    """
    fraction of pixels in the inner border band that are dark (background).
    tight corners -> near 0; background bleed -> higher.
    """
    gray = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    border_mask = np.zeros_like(gray)
    # top, bottom, left, right bands
    border_mask[:band, :]    = 1
    border_mask[h-band:, :]  = 1
    border_mask[:, :band]    = 1
    border_mask[:, w-band:]  = 1

    border_pixels = border_mask.sum()
    dark_in_border = ((gray < 80) & (border_mask > 0)).sum()
    return float(dark_in_border) / (border_pixels + 1e-6)


# ─── per-image processing ────────────────────────────────────────────────────

def process_image(img_path: str) -> dict:
    name = os.path.splitext(os.path.basename(img_path))[0]
    print(f"  {name} ...", end=" ", flush=True)

    original_bgr = cv2.imread(img_path)
    if original_bgr is None:
        print("FAIL (file unreadable)")
        return {"name": name, "status": "FAIL", "method": "-",
                "slant": None, "border": None, "note": "file unreadable"}

    try:
        rectified, corners, method, sealed_mask = rectify_page_debug(img_path)
    except Exception as exc:
        print(f"FAIL ({exc})")
        return {"name": name, "status": "FAIL", "method": "-",
                "slant": None, "border": None, "note": str(exc)}

    slant  = _measure_staff_slant(rectified)
    border = _measure_border_bg(rectified)

    # ── 3-panel figure ──────────────────────────────────────────────────────

    # panel 1: original with quad overlay + corner dots
    vis = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    color_bgr = _METHOD_COLOR.get(method, (255, 255, 0))
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    pts = corners.astype(np.int32)
    for i in range(4):
        cv2.line(vis, tuple(pts[i]), tuple(pts[(i + 1) % 4]), color_rgb, 3)
    for pt in pts:
        cv2.circle(vis, tuple(pt), 10, color_rgb, -1)
    labels = ["TL", "TR", "BR", "BL"]
    for pt, lbl in zip(pts, labels):
        cv2.putText(vis, lbl, (pt[0] + 12, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_rgb, 2, cv2.LINE_AA)

    # panel 2: binary page mask
    mask_rgb = cv2.cvtColor(sealed_mask, cv2.COLOR_GRAY2RGB)

    # panel 3: rectified output
    rect_rgb = cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(name, fontsize=13, fontweight="bold")

    axes[0].imshow(vis)
    axes[0].set_title(
        f"original + detected corners  [{method.upper()}]", fontsize=10
    )
    axes[0].axis("off")
    patch = mpatches.Patch(
        color=np.array(color_rgb) / 255,
        label=f"method: {method}"
    )
    axes[0].legend(handles=[patch], loc="lower right", fontsize=8)

    axes[1].imshow(mask_rgb, cmap="gray")
    axes[1].set_title("page mask (sealed)", fontsize=10)
    axes[1].axis("off")

    slant_pass  = slant  <= PASS_SLANT_DEG
    border_pass = border <= PASS_BORDER_BG
    overall_ok  = slant_pass and border_pass and method != "minrect"
    status_str  = "PASS" if overall_ok else "WARN"

    axes[2].imshow(rect_rgb)
    axes[2].set_title(
        f"rectified  {OUT_W}x{OUT_H}\n"
        f"staff slant: {slant:.2f}° {'✓' if slant_pass else '✗'}   "
        f"border bg: {border*100:.1f}% {'✓' if border_pass else '✗'}",
        fontsize=10,
    )
    axes[2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(RESULTS, f"{name}_rectified.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    note = f"-> {os.path.relpath(out_path, BASE)}"
    print(f"{status_str}  [{method}]  slant={slant:.2f}°  border={border*100:.1f}%  {note}")

    return {
        "name": name, "status": status_str, "method": method,
        "slant": slant, "border": border, "note": note,
    }


# ─── entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) > 1:
        # single image passed on command line
        image_paths = [sys.argv[1]]
    else:
        # auto-discover all JPGs in sheet music/
        image_paths = sorted(glob.glob(os.path.join(SHEET_DIR, "*.jpg")))
        if not image_paths:
            sys.exit(
                f"No .jpg files found in '{SHEET_DIR}'.  "
                "Pass an image path as argument or add photos to that directory."
            )

    print(f"\nTesting {len(image_paths)} image(s):\n")
    results = [process_image(p) for p in image_paths]

    # ── summary table ────────────────────────────────────────────────────────
    SEP = "-" * 82
    print(f"\n{SEP}")
    print(f"  {'IMAGE':<32}  {'METHOD':<8}  {'SLANT':>7}  {'BORDER':>8}  STATUS")
    print(SEP)
    for r in results:
        if r["slant"] is None:
            print(f"  {r['name']:<32}  {r['method']:<8}  {'N/A':>7}  {'N/A':>8}  {r['status']}")
        else:
            slant_str  = f"{r['slant']:.2f}°"
            border_str = f"{r['border']*100:.1f}%"
            print(f"  {r['name']:<32}  {r['method']:<8}  {slant_str:>7}  {border_str:>8}  {r['status']}")
    print(SEP)

    passed = sum(1 for r in results if r["status"] == "PASS")
    warned = sum(1 for r in results if r["status"] == "WARN")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    print(f"\n  {passed} PASS  |  {warned} WARN  |  {failed} FAIL  "
          f"(of {len(results)} images)")
    print(f"  PASS criteria: method != minrect, "
          f"staff slant <= {PASS_SLANT_DEG}°, border bg <= {PASS_BORDER_BG*100:.0f}%")
    if method_counts := {m: sum(1 for r in results if r["method"]==m)
                         for m in ("hough","polydp","minrect")}:
        print(f"  corner methods: " +
              "  ".join(f"{m}={c}" for m, c in method_counts.items() if c))
    print()


if __name__ == "__main__":
    main()
