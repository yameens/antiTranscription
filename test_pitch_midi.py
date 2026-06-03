"""
test_pitch_midi.py
end-to-end test of the pitch inference and MIDI writing stages.

for each sheet-music image it:
  1. runs the full pipeline: rectify -> detect_staves -> remove_staves_and_segment
  2. filters detected components to note-like shapes
  3. aligns detected components 1-to-1 with the ground-truth note sequence
  4. calls infer_pitch on each aligned component
  5. compares inferred pitch against ground truth and reports accuracy
  6. writes two MIDI files to results/:
       <name>_gt.mid        ground truth (gold standard reference)
       <name>_inferred.mid  inferred pitches + ground-truth durations

run:
    python3 test_pitch_midi.py
"""

import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from pitch_midi import infer_pitch, write_midi
from sheet_utils import (
    parse_ground_truth,
    filter_note_components,
    run_pipeline,
    LOCAL_IMAGES,
    NOTE_DIR,
)


BASE   = os.path.dirname(os.path.abspath(__file__))
SHEET  = os.path.join(BASE, "sheet music")
OUTDIR = os.path.join(BASE, "results")
os.makedirs(OUTDIR, exist_ok=True)


def _nearest_stave(y_centre: float, staves: list[list[float]]) -> list[float]:
    centres = [sum(s) / len(s) for s in staves]
    return staves[min(range(len(centres)), key=lambda i: abs(centres[i] - y_centre))]


def _pitch_distance(p1: str, p2: str) -> int:
    """
    diatonic step distance between two pitch strings.
    e.g. "C4" vs "D4" -> 1,  "C4" vs "G4" -> 4.
    used for ±1-step tolerance reporting.
    """
    NOTE_NAMES = ["C", "D", "E", "F", "G", "A", "B"]
    def to_steps(p):
        name, oct_ = p[:-1].upper(), int(p[-1])
        if name not in NOTE_NAMES:
            return 0
        return oct_ * 7 + NOTE_NAMES.index(name)
    try:
        return abs(to_steps(p1) - to_steps(p2))
    except (ValueError, IndexError):
        return 99


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

summary_rows = []

for img_name, gt_filename in LOCAL_IMAGES:
    jpg_path = os.path.join(SHEET, f"{img_name}.jpg")
    gt_path  = os.path.join(NOTE_DIR, gt_filename)
    print(f"\n{'='*60}")
    print(f"image: {img_name}")

    if not os.path.isfile(jpg_path) or not os.path.isfile(gt_path):
        print("  skipping: file not found")
        summary_rows.append((img_name, 0, 0, 0, 0, "SKIP"))
        continue

    # ground truth
    gt_seq = parse_ground_truth(gt_path)
    gt_notes = [(p, d) for p, d in gt_seq]
    print(f"  ground truth notes: {len(gt_notes)}")

    # pipeline
    try:
        rectified, staves, spacing, symbols, cleaned = run_pipeline(jpg_path)
    except Exception as exc:
        print(f"  pipeline failed: {exc}")
        summary_rows.append((img_name, 0, len(gt_notes), 0, 0, "FAIL"))
        continue

    print(f"  staves: {len(staves)}  spacing: {spacing:.1f} px  "
          f"raw symbols: {len(symbols)}")

    h_img, w_img = rectified.shape[:2]
    note_comps = filter_note_components(symbols, spacing, w_img, staves=staves)
    print(f"  note-like components after filter: {len(note_comps)}")

    # re-sort by (stave_index, x) so components are in reading order.
    # remove_staves_and_segment sorts by (y_centre, x), which puts high-
    # pitched notes first within a stave regardless of horizontal position.
    # that scrambles the 1-to-1 alignment with the ground-truth sequence.
    _stave_centres = [sum(s) / len(s) for s in staves]
    note_comps.sort(key=lambda sym: (
        min(range(len(_stave_centres)),
            key=lambda i: abs(_stave_centres[i] - (sym[0][1] + sym[0][3] / 2))),
        sym[0][0],   # x = reading order within the stave
    ))

    # align components to ground truth 1:1 left-to-right
    n_align = min(len(note_comps), len(gt_notes))
    aligned_pairs = list(zip(note_comps[:n_align], gt_notes[:n_align]))

    # infer pitches
    exact_hits = 0
    near_hits  = 0   # within ±1 diatonic step
    inferred_seq: list[tuple[str | None, str]] = []
    annot_rows: list[tuple[tuple, str, str, bool]] = []  # (bbox, gt_pitch, inf_pitch, correct)

    for (bbox, crop), (gt_pitch, gt_dur) in aligned_pairs:
        x, y, w, h = bbox
        sym_y = y + h * 0.5
        staff_ys = _nearest_stave(sym_y, staves)
        inf_pitch = infer_pitch(bbox, staff_ys, spacing,
                                "note_quarter", crop=crop)

        if gt_pitch is None:
            # rest: no pitch to evaluate; carry forward in MIDI sequence
            inferred_seq.append((None, gt_dur))
            continue

        dist  = _pitch_distance(inf_pitch, gt_pitch) if inf_pitch else 99
        exact = (inf_pitch == gt_pitch)
        near  = (dist <= 1)
        exact_hits += int(exact)
        near_hits  += int(near)

        inferred_seq.append((inf_pitch, gt_dur))
        annot_rows.append((bbox, gt_pitch, inf_pitch or "?", exact))

    # fill remaining gt notes (if fewer components than notes) using ground truth
    for gt_pitch, gt_dur in gt_notes[n_align:]:
        inferred_seq.append((gt_pitch, gt_dur))   # fall back to gt for MIDI

    n_note_gt = sum(1 for p, _ in gt_notes if p is not None)
    n_compared = len(annot_rows)
    pct_exact = 100 * exact_hits / n_compared if n_compared else 0.0
    pct_near  = 100 * near_hits  / n_compared if n_compared else 0.0
    print(f"  aligned pairs: {n_align}  notes compared: {n_compared}")
    print(f"  pitch exact:   {exact_hits}/{n_compared}  ({pct_exact:.0f}%)")
    print(f"  pitch ±1 step: {near_hits}/{n_compared}  ({pct_near:.0f}%)")

    # --- write ground-truth MIDI ---
    gt_midi_path = os.path.join(OUTDIR, f"{img_name}_gt.mid")
    write_midi(gt_notes, gt_midi_path, tempo_bpm=100)
    print(f"  GT MIDI:       results/{img_name}_gt.mid")

    # --- write inferred MIDI ---
    inf_midi_path = os.path.join(OUTDIR, f"{img_name}_inferred.mid")
    write_midi(inferred_seq, inf_midi_path, tempo_bpm=100)
    print(f"  inferred MIDI: results/{img_name}_inferred.mid")

    # --- visualisation: annotated bounding boxes coloured by correctness ---
    vis = cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB).copy()

    # draw staff lines in light gray
    for stave in staves:
        for sy in stave:
            cv2.line(vis, (0, int(sy)), (w_img - 1, int(sy)), (190, 190, 190), 1)

    for (x, y, w, h), gt_p, inf_p, correct in annot_rows:
        color = (30, 180, 30) if correct else (220, 50, 50)  # green / red
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)
        # label: gt on top, inferred below
        cv2.putText(vis, gt_p,  (x, max(0, y - 9)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 100, 180), 1, cv2.LINE_AA)
        cv2.putText(vis, inf_p, (x, y + h + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, color,         1, cv2.LINE_AA)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"{img_name}  —  pitch inference  "
                 f"(exact {pct_exact:.0f}%  ±1-step {pct_near:.0f}%)",
                 fontsize=12, fontweight="bold", y=1.01)

    axes[0].imshow(vis)
    axes[0].set_title("(a)  inferred (green=correct, red=wrong)\n"
                      "     label above bbox = ground truth, below = inferred")
    axes[0].axis("off")

    # piano-roll style: gt vs inferred
    NOTE_NAMES = ["C", "D", "E", "F", "G", "A", "B"]
    def pitch_to_step(p):
        if p is None: return None
        try:
            return int(p[-1]) * 7 + NOTE_NAMES.index(p[:-1].upper())
        except (ValueError, IndexError):
            return None

    gt_steps  = [pitch_to_step(p) for p, _ in gt_notes  if p is not None]
    inf_steps = [pitch_to_step(p) for p, _ in inferred_seq if p is not None]
    max_len   = max(len(gt_steps), len(inf_steps), 1)

    axes[1].plot(range(len(gt_steps)),  gt_steps,  "b.-", label="ground truth", alpha=0.8)
    axes[1].plot(range(len(inf_steps)), inf_steps, "r.-", label="inferred",     alpha=0.8)
    axes[1].set_xlabel("note index (left to right)")
    axes[1].set_ylabel("diatonic pitch step")
    axes[1].set_title("(b)  pitch contour: ground truth vs inferred")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUTDIR, f"{img_name}_pitch_eval.png")
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure:        results/{img_name}_pitch_eval.png")

    summary_rows.append((img_name, n_compared, exact_hits, near_hits,
                         len(staves), "PASS"))


# ---------------------------------------------------------------------------
# summary table
# ---------------------------------------------------------------------------
SEP = "-" * 72
print(f"\n{SEP}")
print(f"{'IMAGE':<28} {'COMPARED':>8} {'EXACT':>6} {'EXACT%':>7} "
      f"{'±1STEP':>7} {'±1%':>5}  STATUS")
print(SEP)
for row in summary_rows:
    if row[-1] in ("SKIP", "FAIL"):
        print(f"{row[0]:<28}  {'—':>8}  {'—':>6}  {'—':>7}  {'—':>7}  {'—':>5}  {row[-1]}")
    else:
        name, n_cmp, n_exact, n_near, n_staves, status = row
        pct_e = 100 * n_exact / n_cmp if n_cmp else 0.0
        pct_n = 100 * n_near  / n_cmp if n_cmp else 0.0
        print(f"{name:<28} {n_cmp:>8} {n_exact:>6} {pct_e:>6.0f}%"
              f" {n_near:>7} {pct_n:>4.0f}%  {status}")
print(SEP)
print("MIDI files and eval figures saved to results/")
