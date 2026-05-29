#!/usr/bin/env python
"""Apply Petrunia's silver cleaning recipe + the matching train filter.

Writes data/cv/{train,silver}_clean.jsonl ready for ms-swift data prep.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from htr.data import iter_metadata, source_summary, write_jsonl
from htr.silver import (
    build_train_vocab,
    filter_items,
    is_valid_silver_region,
    is_valid_train_region,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seq-len", type=int, default=260,
                    help="Drop regions whose text exceeds this length (99th pct of train).")
    ap.add_argument("--max-oov", type=int, default=2,
                    help="Drop silver regions with > N chars outside train vocab.")
    ap.add_argument("--out-dir", default="data/cv")
    args = ap.parse_args()

    print("Loading train metadata...")
    train_items = list(iter_metadata("train"))
    print(f"  train pages: {len(train_items)}; sources: {source_summary(train_items)}")

    print("Building train character vocabulary...")
    vocab = build_train_vocab(train_items)
    print(f"  vocab size: {len(vocab)} unique chars")

    print("Filtering train (length cap only)...")
    clean_train, _ = filter_items(
        train_items,
        lambda r: is_valid_train_region(r, max_seq_len=args.max_seq_len),
        label="train",
    )

    print("Loading silver metadata...")
    silver_items = list(iter_metadata("silver"))
    print(f"  silver pages: {len(silver_items)}; sources: {source_summary(silver_items)}")

    print("Filtering silver (repetition + length + vocab)...")
    clean_silver, _ = filter_items(
        silver_items,
        lambda r: is_valid_silver_region(r, vocab,
                                         max_seq_len=args.max_seq_len,
                                         max_oov=args.max_oov),
        label="silver",
    )

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "train_clean.jsonl", clean_train)
    write_jsonl(out_dir / "silver_clean.jsonl", clean_silver)
    print(f"\nWrote {out_dir / 'train_clean.jsonl'} and {out_dir / 'silver_clean.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
