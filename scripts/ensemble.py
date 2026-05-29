#!/usr/bin/env python
"""Phase 3 stub: 2-seed ensemble via line-level edit-distance consensus.

Given two prediction CSVs (same image keys, different seeds / configs), pick
the better transcription per matched region:

  - bbox: union of the two bboxes (small expansion harmless under IoU≥0.5
    matching; alternative: pick the one with higher confidence).
  - type: majority — if both agree, keep; if disagree, prefer 'handwritten'
    on uncertain lines (most common GT class).
  - text: keep the longer-confidence-or-edit-distance-closer-to-consensus
    line. Implementation: tokenize both into char arrays, compute their
    Levenshtein alignment, keep the chars where they agree; for diverging
    runs, prefer the one with fewer non-Cyrillic chars (proxy for less
    hallucination — the model often slips Latin lookalikes when uncertain).

Sketch:

    def merge(pred_a, pred_b):
        pairs = match_by_iou(pred_a.regions, pred_b.regions, thresh=0.3)
        out = []
        for ra, rb in pairs:
            out.append({
                "bbox": union(ra.bbox, rb.bbox),
                "type": pick_type(ra.type, rb.type),
                "text": consensus(ra.text, rb.text),
            })
        # Unmatched regions: keep with low confidence
        return out

Usage (once implemented):
    python scripts/ensemble.py \\
        --in predictions/val_v1.csv \\
        --in predictions/val_v2.csv \\
        --out submissions/val_ensemble.csv
"""
from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inputs", action="append", required=False,
                    help="Prediction CSVs to ensemble (pass multiple times).")
    ap.add_argument("--out", default="submissions/ensemble.csv")
    args = ap.parse_args()

    print("NOT YET IMPLEMENTED.")
    print()
    print("Implement when you have two Phase-2 checkpoints to combine. The math")
    print("for marginal gain is small (~+0.005-0.015 composite) — leave this")
    print("for the final week. Sketch is in the module docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
