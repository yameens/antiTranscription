"""
test_rectify.py
quick visual test for rectify_page.
"""

import sys
import cv2
import matplotlib.pyplot as plt

from rectify import rectify_page


def main() -> None:
    image_path = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"

    print(f"Input image : {image_path}")

    original_bgr = cv2.imread(image_path)
    if original_bgr is None:
        sys.exit(
            f"Error: could not load '{image_path}'.  "
            "Pass the path to your photo as the first argument, e.g.:\n"
            "    python test_rectify.py my_sheet_music.jpg"
        )
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    rectified_bgr = rectify_page(image_path)
    rectified_rgb = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB)

    print(f"Output shape: {rectified_bgr.shape}  (height x width x channels)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    fig.suptitle("Sheet Music Page Rectifier", fontsize=14, fontweight="bold")

    axes[0].imshow(original_rgb)
    axes[0].set_title("Original (phone photo)")
    axes[0].axis("off")

    axes[1].imshow(rectified_rgb)
    axes[1].set_title(
        f"Rectified  ({rectified_bgr.shape[1]} x {rectified_bgr.shape[0]} px)"
    )
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
