"""
batch_test.py
run rectify_page on all test images and save side-by-side comparisons.
"""

import os
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rectify import rectify_page

BASE = os.path.dirname(os.path.abspath(__file__))
SHEET = os.path.join(BASE, "sheet music")

IMAGES = [
    ("cs131",                   "cs131.jpg"),
    ("londonBridgeIsFalling",   "londonBridgeIsFalling.jpg"),
    ("maryHadLittleLamb",       "maryHadLittleLamb.jpg"),
    ("twinkleTwinkleLittleStar","twinkleTwinkleLittleStar.jpg"),
    ("yankeeDoodle",            "yankeeDoodle.jpg"),
]

RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_comparison(name: str, original_bgr, rectified_bgr) -> str:
    """save side-by-side png and return its path."""
    orig_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    rect_rgb = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    fig.suptitle(name, fontsize=13, fontweight="bold")

    axes[0].imshow(orig_rgb)
    axes[0].set_title("Original (phone photo)")
    axes[0].axis("off")

    axes[1].imshow(rect_rgb)
    axes[1].set_title(
        f"Rectified  ({rectified_bgr.shape[1]} x {rectified_bgr.shape[0]} px)"
    )
    axes[1].axis("off")

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"{name}_rectified.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


results = []

for name, filename in IMAGES:
    img_path = os.path.join(SHEET, filename)
    print(f"Processing: {filename} ...", end=" ", flush=True)

    original = cv2.imread(img_path)
    if original is None:
        msg = "cv2.imread returned None — file not found or unreadable"
        print(f"FAIL  ({msg})")
        results.append((name, "FAIL", msg))
        continue

    try:
        rectified = rectify_page(img_path)
        out_path = save_comparison(name, original, rectified)
        shape_str = f"{rectified.shape[1]}x{rectified.shape[0]} px"
        print(f"PASS  -> {os.path.relpath(out_path, BASE)}  [{shape_str}]")
        results.append((name, "PASS", shape_str))

    except ValueError as exc:
        print(f"FAIL  ({exc})")
        results.append((name, "FAIL (ValueError)", str(exc)))

    except Exception as exc:
        print(f"FAIL  ({type(exc).__name__}: {exc})")
        results.append((name, f"FAIL ({type(exc).__name__})", str(exc)))


SEP = "-" * 80
print(f"\n{SEP}")
print(f"{'IMAGE':<35}  {'STATUS':<22}  DETAIL")
print(SEP)
for name, status, detail in results:
    short_detail = (detail[:38] + "...") if len(detail) > 41 else detail
    print(f"{name:<35}  {status:<22}  {short_detail}")
print(SEP)

passed = sum(1 for _, s, _ in results if s == "PASS")
print(f"\nResult: {passed}/{len(results)} images rectified successfully.")
if passed < len(results):
    print("See failure details above for milestone write-up notes.")
