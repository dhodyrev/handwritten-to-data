#!/usr/bin/env python
"""Pull train metadata from HF and write a stratified CV split.

Usage:
    python scripts/build_cv_split.py [--val-frac 0.10] [--seed 42]

Writes data/cv/{train,val}.jsonl.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from htr.data import build_cv_split, iter_metadata, source_summary, write_jsonl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="data/cv")
    args = ap.parse_args()

    print(f"Pulling train metadata from HuggingFace ({args.val_frac:.0%} val, seed={args.seed})...")
    items = list(iter_metadata("train"))
    print(f"  total train items: {len(items)}")
    print(f"  source breakdown : {source_summary(items)}")

    train, val = build_cv_split(items, val_frac=args.val_frac, seed=args.seed)
    out_dir = Path(args.out_dir)
    n_train = write_jsonl(out_dir / "train.jsonl", train)
    n_val = write_jsonl(out_dir / "val.jsonl", val)

    print(f"\n  train: {n_train} → {out_dir / 'train.jsonl'}")
    print(f"  val  : {n_val} → {out_dir / 'val.jsonl'}")
    print(f"  val source breakdown: {source_summary(val)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
