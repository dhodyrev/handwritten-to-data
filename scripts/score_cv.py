#!/usr/bin/env python
"""Score a submission CSV against a CV split using the official metric.

Examples:
    python scripts/score_cv.py --pred predictions/val_baseline.csv
    python scripts/score_cv.py --pred predictions/val_p1.csv --split val
    python scripts/score_cv.py --pred predictions/val_p1.csv \\
        --solution data/cv/val_solution.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from htr.data import read_jsonl
from htr.metric import score_detailed


def build_solution_from_manifest(manifest_path: str | Path) -> pd.DataFrame:
    """Convert HF metadata.jsonl items into the solution CSV shape:
    columns image, regions (regions is a JSON list of GT region dicts)."""
    items = read_jsonl(manifest_path)
    rows = []
    for it in items:
        rows.append({
            "image": Path(it["file_name"]).name,
            "regions": json.dumps(it.get("regions", []), ensure_ascii=False),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="Prediction CSV (image, regions).")
    ap.add_argument("--split", choices=["train", "val"],
                    help="Build solution from data/cv/{split}.jsonl.")
    ap.add_argument("--solution", help="Explicit solution CSV path.")
    args = ap.parse_args()

    if not args.split and not args.solution:
        ap.error("pass --split or --solution")

    if args.solution:
        sol = pd.read_csv(args.solution)
    else:
        sol = build_solution_from_manifest(Path("data/cv") / f"{args.split}.jsonl")

    sub = pd.read_csv(args.pred)
    # Align: keep only images present in both
    sub = sub[sub["image"].isin(set(sol["image"]))].reset_index(drop=True)
    # Submission must include every solution image — fill missing as []
    missing = set(sol["image"]) - set(sub["image"])
    if missing:
        print(f"WARNING: {len(missing)} images missing from prediction — filling with []")
        fill = pd.DataFrame({"image": list(missing), "regions": ["[]"] * len(missing)})
        sub = pd.concat([sub, fill], ignore_index=True)

    r = score_detailed(sol, sub, "image")
    print()
    print(f"  Images evaluated       : {r['n_images']}")
    print(f"  Matched regions (IoU≥.5): {r['n_matched_regions']}")
    print(f"  False positives        : {r['n_false_positives']}")
    print(f"  False negatives        : {r['n_false_negatives']}")
    print()
    print(f"  Detection F1           : {r['detection_f1']:.4f}   (precision {r['detection_precision']:.3f} / recall {r['detection_recall']:.3f})")
    print(f"  Classification accuracy: {r['classification_accuracy']:.4f}")
    print(f"  Region CER             : {r['region_cer']:.4f}   → score {1-r['region_cer']:.4f}")
    print(f"  Page CER               : {r['page_cer']:.4f}   → score {1-r['page_cer']:.4f}")
    print(f"  ──────────────────────────────────────────────────")
    print(f"  Composite score        : {r['composite_score']:.4f}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
