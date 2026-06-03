"""
segment_symbols.py
remove staff lines from a rectified page, then segment the remaining
musical symbols using connected component analysis.
"""

import cv2
import numpy as np


# minimum bounding-box dimensions to keep as a real symbol
_MIN_W = 4
_MIN_H = 4

# maximum bounding-box dimensions (reject page-wide blobs, margin noise)
_MAX_W_FRACTION = 0.6   # fraction of image width
_MAX_H_FRACTION = 0.4   # fraction of image height


def remove_staves_and_segment(
    image: np.ndarray,
    staff_lines: list[list[float]],
    staff_spacing: float,
    line_thickness_factor: float = 0.6,
    min_symbol_area: int = 40,
) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    """
    remove staff lines from a rectified sheet-music page and return
    cropped images of every detected musical symbol.

    parameters
    ----------
    image : np.ndarray
        bgr rectified page, e.g. output of rectify_page().
    staff_lines : list[list[float]]
        list of staves; each stave is a list of 5 y-coordinates as returned
        by detect_staves().  used to know exactly which rows to mask out.
    staff_spacing : float
        mean vertical distance between adjacent lines in a stave, also from
        detect_staves().  used to set the mask half-thickness: lines are
        erased over a strip of height = staff_spacing * line_thickness_factor
        centred on each detected y-coordinate.
    line_thickness_factor : float
        fraction of staff_spacing to use as the half-height of each erased
        strip.  0.6 erases a strip slightly wider than the line itself,
        which handles thick printed lines and minor y-coordinate jitter
        without eating into adjacent note heads.
    min_symbol_area : int
        minimum connected-component area in pixels to keep.  filters out
        isolated specks, pepper noise, and serif serifs.

    returns
    -------
    list of (bbox, symbol_image) tuples, sorted left-to-right within each
    stave row, then top-to-bottom across staves.
    each bbox is (x, y, w, h) in pixel coordinates on the input image.
    each symbol_image is a grayscale crop of the binarized staff-removed image.
    """
    h_img, w_img = image.shape[:2]
    half = max(1, int(round(staff_spacing * line_thickness_factor / 2)))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # binarize: staff lines and note heads become black on white paper
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=25,
        C=10,
    )
    # invert so symbols are white on black (required for connectedComponents)
    inv = cv2.bitwise_not(binary)

    # --- staff line removal ---
    # strategy: mask-based erasure at known y-coordinates.
    # for each detected staff line, zero out a horizontal strip of height
    # 2*half + 1 centred on that y.  this is more precise than a global
    # horizontal opening, which would also erase ledger lines, ties, and
    # beams that run horizontally.
    staff_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    for stave in staff_lines:
        for y in stave:
            y_int = int(round(y))
            y_lo = max(0, y_int - half)
            y_hi = min(h_img, y_int + half + 1)
            staff_mask[y_lo:y_hi, :] = 255

    # apply the mask: pixels that belong to staff lines are erased
    cleaned = cv2.bitwise_and(inv, cv2.bitwise_not(staff_mask))

    # vertical morphological close to bridge note-stem fragments cut by masking.
    # each mask strip is 2*half+1 pixels tall, so stems are severed every
    # staff_spacing pixels.  closing with a kernel slightly taller than the
    # mask strip height (2*half+3) reconnects the pieces without re-introducing
    # the staff lines, because those rows are now zero and the close fills them
    # back only where adjacent non-zero pixels exist above AND below.
    bridge_h = half * 2 + 3
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, bridge_h))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, bridge_kernel)

    # small opening to remove isolated 1-2 px specks that survive bridging
    stub_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, stub_kernel)

    # --- connected component analysis ---
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )

    max_w = int(w_img * _MAX_W_FRACTION)
    max_h = int(h_img * _MAX_H_FRACTION)

    symbols: list[tuple[tuple[int, int, int, int], np.ndarray]] = []

    for lbl in range(1, n_labels):  # skip background label 0
        x  = int(stats[lbl, cv2.CC_STAT_LEFT])
        y  = int(stats[lbl, cv2.CC_STAT_TOP])
        w  = int(stats[lbl, cv2.CC_STAT_WIDTH])
        h  = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        area = int(stats[lbl, cv2.CC_STAT_AREA])

        if area < min_symbol_area:
            continue
        if w < _MIN_W or h < _MIN_H:
            continue
        if w > max_w or h > max_h:
            continue

        crop = cleaned[y: y + h, x: x + w]
        symbols.append(((x, y, w, h), crop))

    # sort by stave row first (approximate stave from y centre), then by x
    symbols.sort(key=lambda s: (s[0][1] + s[0][3] / 2, s[0][0]))

    return symbols, cleaned


def visualize_segments(
    image: np.ndarray,
    symbols: list[tuple[tuple[int, int, int, int], np.ndarray]],
    staff_lines: list[list[float]] | None = None,
) -> np.ndarray:
    """
    draw bounding boxes around every detected symbol on a copy of image.
    optionally also redraw the detected staff lines in a muted colour
    so you can see what was removed and what was kept.

    returns annotated bgr image.
    """
    vis = image.copy()
    h_img, w_img = vis.shape[:2]

    # light gray staff-line reference
    if staff_lines is not None:
        for stave in staff_lines:
            for y in stave:
                cv2.line(vis, (0, int(round(y))), (w_img - 1, int(round(y))),
                         (180, 180, 180), 1)

    # green bounding boxes around each symbol
    for (x, y, w, h), _ in symbols:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 1)

    cv2.putText(
        vis,
        f"{len(symbols)} symbols detected",
        (8, h_img - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (0, 200, 0), 1, cv2.LINE_AA,
    )
    return vis
