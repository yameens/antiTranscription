"""
rectify.py
perspective-correct a phone photo of sheet music against a contrasting
background and return the unwarped page as a numpy array.

corner-detection cascade (first valid result wins):
  1. Hough edges -> 4 page lines -> intersect  (primary, survives occluded corners)
  2. approxPolyDP adaptive-epsilon 4-vertex quad (fallback 1)
  3. minAreaRect bounding rectangle             (fallback 2, old behavior)
"""

import math
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

# reject page candidates covering less than this fraction of the frame
_MIN_PAGE_AREA_FRACTION = 0.15

# Hough tuning (exposed for the test harness to override if needed)
_HOUGH_THRESHOLD      = 80    # votes needed to accept a line
_HOUGH_MIN_LINE_LEN   = 100   # minimum segment length in px
_HOUGH_MAX_LINE_GAP   = 30    # maximum gap to bridge within a segment
_HOUGH_ANGLE_SPLIT    = 40    # degrees from horizontal to split h vs v groups


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _order_corners(pts: np.ndarray) -> np.ndarray:
    """return corners ordered as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]   # tl: smallest x+y
    ordered[2] = pts[np.argmax(s)]   # br: largest  x+y

    diff = np.diff(pts, axis=1).flatten()
    ordered[1] = pts[np.argmin(diff)]  # tr: smallest y-x
    ordered[3] = pts[np.argmax(diff)]  # bl: largest  y-x

    return ordered


def _line_intersection(p1, d1, p2, d2):
    """
    intersect two lines defined as (point, direction_unit_vector).
    returns (x, y) float or None if lines are parallel.
    """
    # solve: p1 + t*d1 = p2 + s*d2
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-6:
        return None
    t = (dx * d2[1] - dy * d2[0]) / cross
    x = p1[0] + t * d1[0]
    y = p1[1] + t * d1[1]
    return float(x), float(y)


def _segments_to_line(segments):
    """
    fit a single line through all (x1,y1,x2,y2) segments using cv2.fitLine.
    returns (point, unit_direction) or None.
    """
    pts = []
    for x1, y1, x2, y2 in segments:
        pts.append([x1, y1])
        pts.append([x2, y2])
    if len(pts) < 2:
        return None
    pts_arr = np.array(pts, dtype=np.float32)
    vx, vy, cx, cy = cv2.fitLine(pts_arr, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    return (float(cx), float(cy)), (float(vx), float(vy))


# ---------------------------------------------------------------------------
# segmentation
# ---------------------------------------------------------------------------

def _clahe_equalize(gray: np.ndarray) -> np.ndarray:
    """
    apply CLAHE (contrast-limited adaptive histogram equalization) to
    normalize local brightness.  this makes shadow regions and lit regions
    comparable, which is the key fix for steep-angle / uneven-lighting shots
    where global Otsu misclassifies the shadowed far-end of the page as
    background and the lit desk as page.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _segment_page_otsu(gray: np.ndarray) -> np.ndarray:
    """binary mask: page = 255, background = 0 via otsu threshold."""
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def _segment_page_bright(gray: np.ndarray) -> np.ndarray:
    """
    bright-pixel mask: keep pixels above a fixed brightness threshold.
    most effective when the background is wood/carpet (non-white) and the
    page is white — even in shadow a white page stays brighter than a desk.
    applied after CLAHE so the threshold is relative to the equalized image.
    """
    _, mask = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    return mask


def _segment_page_adaptive(gray: np.ndarray) -> np.ndarray:
    """fallback mask when otsu and bright-threshold both fail."""
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        151, -10,
    )


def _largest_contour(mask: np.ndarray):
    """return (largest_contour, area_fraction) or (None, 0.0)."""
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, 0.0
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    total = mask.shape[0] * mask.shape[1]
    return largest, area / total


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def _validate_quad(quad: np.ndarray, frame_shape: tuple) -> bool:
    """
    accept a 4-point quadrilateral if:
    - all 4 corners are within the frame (with a small margin)
    - the hull is convex
    - its area covers at least _MIN_PAGE_AREA_FRACTION of the frame
    - no side is shorter than 5% of the longer frame dimension
    - opposite sides differ by no more than 60% (not wildly trapezoidal)
    """
    h, w = frame_shape[:2]
    margin = -20  # allow corners up to 20 px outside the frame

    pts = quad.reshape(4, 2).astype(np.float32)

    # all corners roughly in-frame
    if np.any(pts[:, 0] < margin) or np.any(pts[:, 0] > w - margin):
        return False
    if np.any(pts[:, 1] < margin) or np.any(pts[:, 1] > h - margin):
        return False

    # convex hull check
    hull = cv2.convexHull(pts.astype(np.int32))
    if len(hull) != 4:
        return False

    # area fraction
    area = cv2.contourArea(pts)
    if area / (h * w) < _MIN_PAGE_AREA_FRACTION:
        return False

    # minimum side length
    min_dim = min(h, w) * 0.05
    sides = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
    if min(sides) < min_dim:
        return False

    # opposite sides not too unequal (top vs bottom, left vs right)
    if max(sides[0], sides[2]) / (min(sides[0], sides[2]) + 1e-6) > 2.5:
        return False
    if max(sides[1], sides[3]) / (min(sides[1], sides[3]) + 1e-6) > 2.5:
        return False

    return True


# ---------------------------------------------------------------------------
# corner detectors
# ---------------------------------------------------------------------------

def _corners_via_hough(sealed_mask: np.ndarray, frame_shape: tuple):
    """
    primary method: detect the 4 page edges via HoughLinesP on the outer
    contour boundary, classify segments into horizontal/vertical groups using
    image-centre-based splitting, fit one line per group, intersect adjacent
    pairs to get 4 corners.

    uses image-centre splitting (not median) so page edges are correctly
    separated even when the four sides produce unequal numbers of segments.

    returns ordered (4,2) float32 array or None on failure.
    """
    h, w = frame_shape[:2]

    # draw only the outer contour as a 3-px edge (thicker = more Hough votes)
    contours, _ = cv2.findContours(
        sealed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    edge = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(edge, [largest], -1, 255, 3)

    # scale Hough params to image size so the method works at any resolution
    min_len = max(_HOUGH_MIN_LINE_LEN, int(min(h, w) * 0.04))
    lines = cv2.HoughLinesP(
        edge,
        rho=1,
        theta=np.pi / 180,
        threshold=_HOUGH_THRESHOLD,
        minLineLength=min_len,
        maxLineGap=_HOUGH_MAX_LINE_GAP,
    )
    if lines is None or len(lines) < 4:
        return None

    segs = lines.reshape(-1, 4).tolist()

    # classify by angle into horizontal vs vertical
    h_segs, v_segs = [], []
    for x1, y1, x2, y2 in segs:
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if angle > 90:
            angle = 180 - angle
        if angle < _HOUGH_ANGLE_SPLIT:
            h_segs.append((x1, y1, x2, y2))
        else:
            v_segs.append((x1, y1, x2, y2))

    if len(h_segs) < 2 or len(v_segs) < 2:
        return None

    # image-centre-based split: top = h segments above centre, bottom = below
    # this is robust because we know the page spans the image
    cy, cx = h / 2, w / 2
    top_segs  = [s for s in h_segs if (s[1] + s[3]) / 2 < cy]
    bot_segs  = [s for s in h_segs if (s[1] + s[3]) / 2 >= cy]
    left_segs  = [s for s in v_segs if (s[0] + s[2]) / 2 < cx]
    right_segs = [s for s in v_segs if (s[0] + s[2]) / 2 >= cx]

    # need at least one segment per side
    if not top_segs or not bot_segs or not left_segs or not right_segs:
        return None

    # fit one line per group
    top   = _segments_to_line(top_segs)
    bot   = _segments_to_line(bot_segs)
    left  = _segments_to_line(left_segs)
    right = _segments_to_line(right_segs)

    if None in (top, bot, left, right):
        return None

    # intersect adjacent pairs: tl=top∩left, tr=top∩right, br=bot∩right, bl=bot∩left
    tl = _line_intersection(top[0],  top[1],  left[0],  left[1])
    tr = _line_intersection(top[0],  top[1],  right[0], right[1])
    br = _line_intersection(bot[0],  bot[1],  right[0], right[1])
    bl = _line_intersection(bot[0],  bot[1],  left[0],  left[1])

    if None in (tl, tr, br, bl):
        return None

    quad = np.array([tl, tr, br, bl], dtype=np.float32)

    if not _validate_quad(quad, frame_shape):
        return None

    return _order_corners(quad)


def _corners_via_polydp(contour: np.ndarray, frame_shape: tuple):
    """
    fallback 1: adaptive-epsilon approxPolyDP until exactly 4 convex points.
    tries epsilon from 1% to 8% of the perimeter.

    returns ordered (4,2) float32 array or None.
    """
    peri = cv2.arcLength(contour, True)
    for eps_frac in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]:
        approx = cv2.approxPolyDP(contour, eps_frac * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            if _validate_quad(quad, frame_shape):
                return _order_corners(quad)
    return None


def _corners_via_minrect(contour: np.ndarray, frame_shape: tuple):
    """
    fallback 2: minimum-area bounding rectangle (original behavior).
    always returns a valid ordered quad but cannot correct perspective.
    """
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float32)
    return _order_corners(box)


# ---------------------------------------------------------------------------
# main corner-finding entry point
# ---------------------------------------------------------------------------

def _find_page_corners(img: np.ndarray):
    """
    detect four page corners as (4,2) float32 in tl/tr/br/bl order.
    returns (corners, method_name, sealed_mask).
    raises ValueError if no method succeeds.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # CLAHE normalizes local brightness for shadow/glare recovery.
    # prepared here but only used in fallback passes so that well-lit images
    # (where raw Otsu already works cleanly) are unaffected.
    equalized = _clahe_equalize(blurred)

    # close kernel: small kernel (25px) for the raw-Otsu pass — just fills note
    # gaps.  larger kernel for CLAHE passes — bridges shadow holes that can span
    # hundreds of pixels on steep-angle phone shots.
    small_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    large_k = max(25, img.shape[0] // 40)
    large_close = cv2.getStructuringElement(cv2.MORPH_RECT, (large_k, large_k))

    def _try_mask(mask: np.ndarray, kernel):
        sealed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contour, frac = _largest_contour(sealed)
        if contour is None or frac < _MIN_PAGE_AREA_FRACTION:
            return None, None, None
        return contour, frac, sealed

    # pass 1: original behavior — raw Otsu + small close (works on well-lit shots)
    contour, frac, sealed = _try_mask(_segment_page_otsu(blurred), small_close)
    # pass 2: CLAHE + Otsu + large close (handles lighting gradient / shadows)
    if contour is None:
        contour, frac, sealed = _try_mask(_segment_page_otsu(equalized), large_close)
    # pass 3: CLAHE + bright-pixel threshold + large close (wood/carpet background)
    if contour is None:
        contour, frac, sealed = _try_mask(_segment_page_bright(equalized), large_close)
    # pass 4: CLAHE + adaptive threshold + large close (last resort)
    if contour is None:
        contour, frac, sealed = _try_mask(_segment_page_adaptive(equalized), large_close)

    if contour is None:
        raise ValueError(
            "Could not locate the page in the image.  No detected blob covered "
            f"at least {int(_MIN_PAGE_AREA_FRACTION * 100)}% of the frame.  "
            "Try: (a) more contrast between page and background, "
            "(b) ensuring all four corners are in frame, "
            "(c) brighter or more uniform lighting."
        )

    frame_shape = img.shape

    # cascade: try each method in order, return first valid result
    corners = _corners_via_hough(sealed, frame_shape)
    if corners is not None:
        return corners, "hough", sealed

    corners = _corners_via_polydp(contour, frame_shape)
    if corners is not None:
        return corners, "polydp", sealed

    # minAreaRect always succeeds (legacy fallback)
    corners = _corners_via_minrect(contour, frame_shape)
    return corners, "minrect", sealed


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def rectify_page(image_path: str) -> np.ndarray:
    """load image, rectify page, return (1100, 850, 3) bgr array."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path!r}")

    corners, _method, _mask = _find_page_corners(img)
    H = cv2.getPerspectiveTransform(corners, _DST_CORNERS)
    return cv2.warpPerspective(img, H, (OUT_W, OUT_H), flags=cv2.INTER_CUBIC)


def rectify_page_debug(image_path: str):
    """
    rectify and return diagnostic information.

    returns:
        rectified  - (1100, 850, 3) bgr rectified image
        corners    - (4, 2) float32 source corners in tl/tr/br/bl order
        method     - string: "hough", "polydp", or "minrect"
        sealed_mask - binary page mask after morphological close
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path!r}")

    corners, method, sealed_mask = _find_page_corners(img)
    H = cv2.getPerspectiveTransform(corners, _DST_CORNERS)
    rectified = cv2.warpPerspective(img, H, (OUT_W, OUT_H), flags=cv2.INTER_CUBIC)
    return rectified, corners, method, sealed_mask
