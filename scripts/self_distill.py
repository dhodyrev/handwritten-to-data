#!/usr/bin/env python
"""Phase 3 stub: self-distillation pass.

After Phase 2 LoRA training, re-predict the silver set with the fine-tuned
model. Keep regions where:
  - normalized text matches the silver text (no change)         → high agreement
  - or predicted text is short and the model's per-token logprob mean > thresh

Drop the rest. Re-train a second LoRA pass on the cleaner set.

The original baseline does not expose logprobs through vLLM by default — to
enable, add ``"logprobs": True`` and a top_logprobs setting in qwen_call's
extra_body, then average exp(logprob) per generated token in the response.

This script is a scaffold; implementation should:
1. Load val predictions JSON files from outputs/.
2. For each region's transcription, compare its normalized form to the
   normalized silver GT for the same bbox (via IoU match).
3. Keep matched + low-edit-distance regions, drop the rest. Write a new
   data/cv/silver_distilled.jsonl.
4. Optionally pass --logprob-threshold to require confidence as well.
"""
from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--silver", default="data/cv/silver_clean.jsonl")
    ap.add_argument("--predictions-dir", required=False,
                    help="Directory of per-image prediction JSONs from a Phase-2 model.")
    ap.add_argument("--out", default="data/cv/silver_distilled.jsonl")
    ap.add_argument("--agree-cer-thresh", type=float, default=0.10,
                    help="Drop regions where pred vs silver CER exceeds this.")
    args = ap.parse_args()

    print("NOT YET IMPLEMENTED.")
    print()
    print("Wire up after running Phase 2:")
    print(f"  1. Run inference on silver with the fine-tuned model")
    print(f"     (run scripts/run_inference.py --manifest data/cv/silver_clean.jsonl")
    print(f"      --adapter <path/to/lora> on a Kaggle GPU notebook).")
    print(f"  2. Pass --predictions-dir to this script along with --silver.")
    print(f"  3. The script will IoU-match GT silver regions to predictions, drop")
    print(f"     pairs with normalized CER > {args.agree_cer_thresh}, and emit")
    print(f"     {args.out} for a second-round LoRA train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
