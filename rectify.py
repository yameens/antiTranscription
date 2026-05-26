"""
rectify.py
perspective-correct a phone photo of sheet music against a contrasting
background and return the unwarped page as a numpy array.
"""

import cv2
import numpy as np


# output canvas: 850 x 1100 px = 8.5 x 11 in at 100 dpi
OUT_W = 850
OUT_H = 1100

# destination corners for the homography (tl, tr, br, bl)
_DST_CORNERS = np.array(
    [[0, 0], [OUT_W - 1, 0], [OUT_W - 1, OUT_H - 1], [0, OUT_H - 1]],
    dtype=np.float32,
)

# reject contours covering less than this fraction of the frame
_MIN_PAGE_AREA_FRACTION = 0.20


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """return corners ordered as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]  # tl
    ordered[2] = pts[np.argmax(s)]  # br

    diff = np.diff(pts, axis=1).flatten()
    ordered[1] = pts[np.argmin(diff)]  # tr
    ordered[3] = pts[np.argmax(diff)]  # bl

    return ordered


def _segment_page_otsu(gray: np.ndarray) -> np.ndarray:
    """binary mask: page = 255, background = 0 via otsu threshold."""
    _, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return mask


def _segment_page_adaptive(gray: np.ndarray) -> np.ndarray:
    """fallback mask when otsu fails (e.g. page and desk are both bright)."""
    return cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=151,
        C=-10,
    )


def _largest_contour(mask: np.ndarray) -> tuple[np.ndarray | None, float]:
    """return (largest_contour, area_fraction) or (none, 0.0)."""
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, 0.0
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    total = mask.shape[0] * mask.shape[1]
    return largest, area / total


def _find_page_corners(img: np.ndarray) -> np.ndarray:
    """detect four page corners as (4, 2) float32 in tl/tr/br/bl order."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # light blur before thresholding
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 25x25 close fills staff lines and text inside the page mask
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))

    def _mask_to_corners(mask: np.ndarray) -> tuple[np.ndarray, float] | None:
        sealed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        contour, frac = _largest_contour(sealed)
        if contour is None or frac < _MIN_PAGE_AREA_FRACTION:
            return None
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        return box.astype(np.float32), frac

    # try otsu first, then adaptive threshold
    result = _mask_to_corners(_segment_page_otsu(blurred))
    if result is None:
        result = _mask_to_corners(_segment_page_adaptive(blurred))

    if result is None:
        raise ValueError(
            "Could not locate the page in the image.  No detected blob covered "
            f"at least {int(_MIN_PAGE_AREA_FRACTION * 100)}% of the frame.  "
            "Try: (a) more contrast between page and background, "
            "(b) ensuring all four corners are in frame, "
            "(c) brighter or more uniform lighting."
        )

    box, _frac = result
    return _order_corners(box)


def rectify_page(image_path: str) -> np.ndarray:
    """load image, rectify page, return (1100, 850, 3) bgr array."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path!r}")

    corners = _find_page_corners(img)
    H = cv2.getPerspectiveTransform(corners, _DST_CORNERS)
    return cv2.warpPerspective(img, H, (OUT_W, OUT_H), flags=cv2.INTER_CUBIC)
