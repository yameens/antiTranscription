"""
detect_staves.py
detect staff lines in a rectified sheet-music page; return staves and spacing.
"""

import math
import warnings

import cv2
import numpy as np


# bgr colours for visualize_staves (cycles if more than 5 staves)
_STAVE_COLORS = [
    (0,   0,   220),   # red
    (0,   180,  0),    # green
    (220,  0,   0),    # blue
    (0,   200, 200),   # cyan
    (200,  0,  200),   # magenta
]


def _cluster_ys(ys: list[float], tol: int) -> list[float]:
    """merge y-values within tol pixels into one representative y per cluster."""
    if not ys:
        return []
    sorted_ys = sorted(ys)
    clusters: list[list[float]] = [[sorted_ys[0]]]
    for y in sorted_ys[1:]:
        if y - clusters[-1][-1] <= tol:
            clusters[-1].append(y)
        else:
            clusters.append([y])
    return [float(np.mean(c)) for c in clusters]


def detect_staves(
    rectified_image: np.ndarray,
    hough_threshold: int = 100,
    min_line_length: int = 300,
    max_line_gap: int = 40,
    angle_tolerance_deg: float = 2.0,
    cluster_tol: int = 3,
    morph_close_width: int = 0,
) -> tuple[list[list[float]], float]:
    """
    detect staff lines; return (list of 5-line staves, mean intra-stave spacing).

    raises ValueError if fewer than 5 line clusters are found.
    """
    gray = cv2.cvtColor(rectified_image, cv2.COLOR_BGR2GRAY)

    # adaptive threshold: blocksize=25, c=10
    binary = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=25,
        C=10,
    )

    # invert so staff lines are white for hough
    inverted = cv2.bitwise_not(binary)

    # optional horizontal morph-close to bridge gaps from note heads
    hough_input = inverted
    if morph_close_width > 0:
        h_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_close_width, 1)
        )
        hough_input = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, h_kernel)

    lines = cv2.HoughLinesP(
        hough_input,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    if lines is None:
        raise ValueError(
            "HoughLinesP found no lines at all.  "
            "Try lowering hough_threshold or min_line_length."
        )

    # keep near-horizontal segments only
    tol_rad = math.radians(angle_tolerance_deg)
    horizontal_ys: list[float] = []

    for seg in lines:
        x1, y1, x2, y2 = seg[0]
        angle = math.atan2(abs(y2 - y1), abs(x2 - x1))
        if angle <= tol_rad:
            horizontal_ys.append((y1 + y2) / 2.0)

    if not horizontal_ys:
        raise ValueError(
            f"No near-horizontal lines survived the {angle_tolerance_deg}-degree "
            "filter.  Try increasing angle_tolerance_deg."
        )

    clustered = _cluster_ys(horizontal_ys, tol=cluster_tol)

    if len(clustered) < 5:
        raise ValueError(
            f"Only {len(clustered)} distinct horizontal line(s) detected "
            f"(need at least 5 for one stave).  "
            "Try: lowering hough_threshold, lowering min_line_length, "
            "or increasing max_line_gap."
        )

    # group clustered y-values into staves of 5 (gap-split + sliding window)
    ys = sorted(clustered)
    gaps = np.diff(ys)
    median_gap = float(np.median(gaps))
    boundary_threshold = 1.5 * median_gap

    gap_groups: list[list[float]] = []
    current_group: list[float] = [ys[0]]
    for i, gap in enumerate(gaps):
        if gap > boundary_threshold:
            gap_groups.append(current_group)
            current_group = [ys[i + 1]]
        else:
            current_group.append(ys[i + 1])
    gap_groups.append(current_group)

    def _is_valid_stave(lines: list[float]) -> bool:
        if len(lines) != 5:
            return False
        inner_gaps = [lines[i + 1] - lines[i] for i in range(4)]
        mean_g = float(np.mean(inner_gaps))
        if mean_g < 5 or mean_g > 50:
            return False
        cv = float(np.std(inner_gaps)) / mean_g if mean_g > 0 else 1.0
        return cv < 0.4

    staves: list[list[float]] = []
    warned_skips: list[str] = []

    def _best_window(lines: list[float]) -> list[float] | None:
        best: list[float] | None = None
        best_cv = float("inf")
        for i in range(len(lines) - 4):
            window = lines[i: i + 5]
            if not _is_valid_stave(window):
                continue
            inner = [window[j + 1] - window[j] for j in range(4)]
            mean_g = float(np.mean(inner))
            cv = float(np.std(inner)) / mean_g if mean_g > 0 else 1.0
            if cv < best_cv:
                best_cv = cv
                best = window
        return best

    for g in gap_groups:
        if _is_valid_stave(g):
            staves.append(g)
        elif len(g) > 5:
            recovered = _best_window(g)
            if recovered is not None:
                staves.append(recovered)
            else:
                warned_skips.append(
                    f"{len(g)} lines at y={[round(y, 1) for y in g]} "
                    "(no valid 5-subset found)"
                )
        else:
            warned_skips.append(
                f"{len(g)} lines at y={[round(y, 1) for y in g]}"
            )

    # sliding-window fallback when gap-split finds nothing
    if not staves and len(ys) >= 5:
        candidates: list[list[float]] = []
        for i in range(len(ys) - 4):
            window = ys[i: i + 5]
            if _is_valid_stave(window):
                candidates.append(window)

        accepted: list[list[float]] = []
        for cand in candidates:
            overlap = any(
                len(set(cand) & set(acc)) >= 3 for acc in accepted
            )
            if not overlap:
                accepted.append(cand)
        staves = accepted

    if warned_skips:
        warnings.warn(
            "Some candidate line groups were not valid staves and were "
            f"skipped: {'; '.join(warned_skips)}.  "
            "This may indicate noise or an imperfect rectification.  "
            "Try tuning cluster_tol or hough_threshold.",
            stacklevel=2,
        )

    if not staves:
        raise ValueError(
            "No complete 5-line staves were found after grouping.  "
            f"Detected {len(ys)} line clusters: {[round(y,1) for y in ys]}.  "
            "Try adjusting cluster_tol or the Hough parameters."
        )

    # mean spacing across all intra-stave gaps
    intra_gaps: list[float] = []
    for stave in staves:
        for i in range(len(stave) - 1):
            intra_gaps.append(stave[i + 1] - stave[i])
    staff_spacing = float(np.mean(intra_gaps))

    return staves, staff_spacing


def visualize_staves(
    rectified_image: np.ndarray,
    staves: list[list[float]],
    staff_spacing: float,
) -> np.ndarray:
    """draw detected staff lines on a copy of the rectified image."""
    vis = rectified_image.copy()
    h, w = vis.shape[:2]

    for stave_idx, stave_ys in enumerate(staves):
        color = _STAVE_COLORS[stave_idx % len(_STAVE_COLORS)]
        line_thickness = 2

        for line_y in stave_ys:
            y = int(round(line_y))
            cv2.line(vis, (0, y), (w - 1, y), color, line_thickness)

        label_y = max(int(round(stave_ys[0])) - 6, 12)
        cv2.putText(
            vis,
            f"Stave {stave_idx + 1}",
            (8, label_y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.55,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    cv2.putText(
        vis,
        f"spacing={staff_spacing:.1f}px  staves={len(staves)}",
        (8, h - 10),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(50, 50, 50),
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    return vis
