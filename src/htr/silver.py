"""Silver-data cleaning recipe (per Petrunia's EDA).

Three filters, applied per-region:
1. Repetition loop: ``(.)\\1{9,}`` — VLM token-loop on dotted-line fills.
2. Length cap: ≤260 chars (99th percentile of train + safety margin).
3. Vocab filter: ≤``max_oov`` chars outside the train alphabet (catches
   hallucinated Latin / Greek / Chinese chunks).

After per-region filtering, pages with zero remaining text-bearing regions
are dropped entirely (no signal left for OCR training).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable


REPETITION = re.compile(r"(.)\1{9,}")


def build_train_vocab(train_items: Iterable[dict]) -> set[str]:
    """Union of all characters in train ground-truth transcriptions."""
    vocab: set[str] = set()
    for it in train_items:
        for r in it.get("regions", []):
            text = r.get("text") or ""
            vocab.update(text)
    return vocab


def is_valid_silver_region(region: dict, train_vocab: set[str],
                           *, max_seq_len: int = 260,
                           max_oov: int = 2) -> bool:
    text = (region.get("text") or "")
    if not text:
        return False
    if len(text) > max_seq_len:
        return False
    if REPETITION.search(text):
        return False
    oov = sum(1 for c in text if c not in train_vocab)
    if oov > max_oov:
        return False
    return True


def is_valid_train_region(region: dict, *, max_seq_len: int = 260) -> bool:
    """Train-side filter — drops the 0.1% outlier paragraph-in-one-bbox cases
    (e.g. 00411.jpg's 525-char block) so the LoRA tokenizer budget doesn't
    have to accommodate them."""
    text = (region.get("text") or "")
    if not text:
        return False
    if len(text) > max_seq_len:
        return False
    return True


def filter_items(items: list[dict], region_filter, *, label: str = "") -> tuple[list[dict], dict]:
    """Apply per-region filter; drop pages with zero text-bearing regions left."""
    kept_pages: list[dict] = []
    stats = Counter()
    for it in items:
        in_regs = it.get("regions", [])
        out_regs = []
        for r in in_regs:
            if region_filter(r):
                out_regs.append(r)
            else:
                stats["dropped_regions"] += 1
        stats["input_regions"] += len(in_regs)
        if out_regs:
            new_it = dict(it)
            new_it["regions"] = out_regs
            kept_pages.append(new_it)
            stats["kept_regions"] += len(out_regs)
        else:
            stats["dropped_empty_pages"] += 1
    stats["input_pages"] = len(items)
    stats["kept_pages"] = len(kept_pages)
    if label:
        print(f"  [{label}] pages {stats['input_pages']} → {stats['kept_pages']}; "
              f"regions {stats['input_regions']} → {stats['kept_regions']} "
              f"(dropped {stats['dropped_regions']}, "
              f"{stats['dropped_empty_pages']} empty pages)")
    return kept_pages, dict(stats)
