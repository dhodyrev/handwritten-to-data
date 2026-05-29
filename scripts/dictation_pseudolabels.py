#!/usr/bin/env python
"""Phase 3 stub: align Ukrainian National Dictation canonical text to detected
lines on dictation-source pages, producing pseudo-GT for the slice silver is
missing (silver has zero dictation pages — domain gap).

Approach (pseudo-code, fill in when you wire it up):

1. Pull canonical dictation text from the official UA National Dictation
   archive — each year has a transcript PDF. Build a year → canonical_text map.
   Save under data/dictation_canonical/{YYYY}.txt.

2. For each dictation page, infer year from the train metadata if present
   (otherwise classify with the model). Pull the canonical text for that year.

3. Run the existing detect step (Qwen3-VL) to get line bboxes; transcribe
   each line with the current model checkpoint. You now have N predicted
   lines vs a single canonical paragraph.

4. Align: greedy + DP. Treat predicted lines and canonical sentences as
   nodes in an edit-distance DP. Match each predicted line to a substring
   of the canonical text such that the total Levenshtein is minimized.
   ``rapidfuzz.process.extract`` works as a first cut; the gold version is
   a Needleman-Wunsch on character-level with affine gap.

5. Replace each predicted line's text with the aligned canonical substring
   when alignment confidence > threshold. Drop lines that can't be aligned
   (likely OCR errors or non-canonical annotations).

6. Output as JSONL in the same shape as silver_clean.jsonl — drop into
   training/data/{dictation_pseudo}.jsonl and re-run prepare_swift_data.py
   with --include-silver --include-dictation-pseudo (TODO flag).

Estimated yield: ~359 dictation train pages exist; with N≈20 lines/page this
adds ~7k high-quality lines to the fine-tune set. Especially valuable because
they fix the silver domain gap.
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical-dir", default="data/dictation_canonical",
                    help="Directory of YYYY.txt canonical dictation texts.")
    ap.add_argument("--in-manifest", default="data/cv/train.jsonl")
    ap.add_argument("--out-manifest", default="data/cv/dictation_pseudo.jsonl")
    args = ap.parse_args()

    print("NOT YET IMPLEMENTED.")
    print()
    print("To wire this up:")
    print(f"  1. Drop canonical dictation transcripts as {args.canonical_dir}/YYYY.txt")
    print("  2. pip install rapidfuzz")
    print("  3. Replace this stub with the alignment loop (see module docstring).")
    print()
    print("Until then, train on train_clean + silver_clean only — the dictation slice")
    print("just won't get the extra ~7k high-quality lines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
