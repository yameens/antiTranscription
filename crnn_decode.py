"""
crnn_decode.py
greedy CTC decoding and the Symbol Error Rate (SER) metric.

CTC emits one label per time-step over an alphabet of (vocab + blank).  the
standard greedy decode collapses the raw frame-level argmax into a token
sequence by (1) merging consecutive duplicates, then (2) dropping the blank.

SER (the canonical PrIMuS metric) is the token-level Levenshtein edit distance
between prediction and ground truth, normalised by the ground-truth length --
i.e. the fraction of symbols wrong.  0.0 is perfect.
"""

from __future__ import annotations

import torch


def greedy_decode(logp: torch.Tensor, input_lengths: torch.Tensor,
                  blank: int) -> list[list[int]]:
    """
    logp          : (T, B, C) log-probabilities from CRNN.forward
    input_lengths : (B,) valid time-steps per sample (rest is right-padding)
    returns a list of token-id sequences (blanks/repeats already collapsed).
    """
    argmax = logp.argmax(dim=2)          # (T, B)
    T, B = argmax.shape
    out: list[list[int]] = []
    for b in range(B):
        n = int(input_lengths[b].item())
        prev = -1
        seq: list[int] = []
        for t in range(min(n, T)):
            c = int(argmax[t, b].item())
            if c != prev and c != blank:
                seq.append(c)
            prev = c
        out.append(seq)
    return out


def edit_distance(a: list, b: list) -> int:
    """token-level Levenshtein distance."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def symbol_error_rate(pred: list, truth: list) -> float:
    """edit_distance / len(truth); guards the empty-truth case."""
    if not truth:
        return 0.0 if not pred else 1.0
    return edit_distance(pred, truth) / len(truth)


def batch_ser(preds: list[list[int]], truths: list[list[int]]) -> tuple[float, int, int]:
    """
    aggregate SER over a batch as total_edits / total_truth_symbols (the
    corpus-level definition, not the mean of per-sample rates).
    returns (ser, total_edits, total_truth_len).
    """
    tot_edits = tot_len = 0
    for p, t in zip(preds, truths):
        tot_edits += edit_distance(p, t)
        tot_len += len(t)
    ser = tot_edits / tot_len if tot_len else 0.0
    return ser, tot_edits, tot_len
