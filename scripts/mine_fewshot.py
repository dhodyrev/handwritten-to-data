#!/usr/bin/env python
"""Mine high-quality (description, gold_text) few-shot examples from train.

Picks short-to-medium clean GT lines per writing_type, filters out edge cases
(strikethrough, very long, non-Cyrillic for handwritten/printed). Writes a YAML
fragment you can paste into transcribe.few_shot_examples in your pipeline config.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

from htr.data import read_jsonl

CYRILLIC = re.compile(r"[А-ЯҐЄІЇа-яґєії]")


def _suitable(rtype: str, text: str) -> bool:
    if not text or len(text) < 8 or len(text) > 120:
        return False
    if "~~" in text or "[" in text:
        return False
    if rtype in ("handwritten", "printed"):
        # Want at least 2 Cyrillic chars so it's actually a UK line, not a number.
        if len(CYRILLIC.findall(text)) < 2:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/cv/train.jsonl")
    ap.add_argument("--out", default="configs/_fewshot.yaml")
    ap.add_argument("--per-type", type=int, default=6)
    args = ap.parse_args()

    items = read_jsonl(args.manifest)

    pool: dict[str, list[str]] = defaultdict(list)
    for it in items:
        for r in it.get("regions", []):
            text = (r.get("text") or "").strip()
            rtype = r.get("type", "handwritten")
            if r.get("language", "uk") != "uk" or r.get("legibility") == "illegible":
                continue
            if rtype in ("image", "graph"):
                continue
            if _suitable(rtype, text):
                pool[rtype].append(text)

    fewshot: dict[str, list[list[str]]] = {}
    for rtype, texts in pool.items():
        texts = sorted(set(texts), key=len)
        chosen = texts[: args.per_type]
        # Description is intentionally generic — we don't pass crop images here.
        fewshot[rtype] = [
            [f"a {rtype} line from a Ukrainian student page", t]
            for t in chosen
        ]
        print(f"  {rtype:11s}: pool={len(texts):4d} → chose {len(chosen)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(
        {"transcribe": {"few_shot_examples": fewshot}},
        sort_keys=True, allow_unicode=True,
    ))
    print(f"\nWrote {out_path}")
    print("Paste its transcribe.few_shot_examples block into configs/pipeline_p1.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
