#!/usr/bin/env python
"""Convert cleaned train + silver into Unsloth/TRL chat-format JSONL.

Two-task setup (separately trainable LoRA adapters):

1. ``detect``    : image → JSON list of {bbox_2d (0-1000), block_type, writing_type}.
   Bboxes normalized to a 0-1000 scale (resize-invariant — sidesteps Qwen3-VL
   ``smart_resize`` coordinate drift).

2. ``transcribe`` : crop image → JSON {"text": "..."}.
   One sample per line; crop computed from GT bbox with the same margin as
   inference (image_ops.crop_with_margin).

Output format per record::

    {
      "messages": [
        {"role": "user",      "content": [
            {"type": "image", "image": "<abs path>"},
            {"type": "text",  "text": "<prompt>"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "<json>"}]}
      ]
    }

Compatible with Unsloth + TRL ``SFTTrainer`` vision pipeline.
Output: data/lora/{task}/{train,val}.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PIL import Image
from tqdm import tqdm

from htr.data import materialize_image, read_jsonl
from htr.image_ops import crop_with_margin, load_image_exif

# ── Shared prompts (must mirror inference-time prompts in spirit) ─

DETECT_PROMPT = (
    "Detect content blocks in this Ukrainian document.\n"
    "Return JSON: {\"blocks\":[{\"bbox_2d\":[x1,y1,x2,y2],"
    "\"block_type\":\"text_block|table|formula|image|graph|annotation\","
    "\"writing_type\":\"handwritten|printed\"}]} "
    "with coordinates on a 0-1000 scale. ONLY valid JSON."
)

TRANSCRIBE_PROMPT = (
    "Transcribe the central line in this crop. Ukrainian Cyrillic only. "
    "Return JSON: {\"text\":\"...\"}"
)


def _to_norm_bbox(bbox: list[int], w: int, h: int) -> list[int]:
    """Pixel bbox → 0-1000 normalized."""
    x1, y1, x2, y2 = bbox
    return [
        max(0, min(1000, int(x1 * 1000 / max(1, w)))),
        max(0, min(1000, int(y1 * 1000 / max(1, h)))),
        max(0, min(1000, int(x2 * 1000 / max(1, w)))),
        max(0, min(1000, int(y2 * 1000 / max(1, h)))),
    ]


def _region_to_block_type(r: dict) -> tuple[str, str]:
    rtype = r.get("type", "handwritten")
    if rtype in ("handwritten", "printed"):
        return "text_block", rtype
    return rtype, "handwritten"


def _chat_sample(image_path: str, prompt: str, response: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": response},
            ]},
        ],
    }


def build_detect_sample(item: dict, image_path: str) -> dict | None:
    try:
        img = load_image_exif(image_path)
    except Exception:
        return None
    w, h = img.size
    blocks = []
    for r in item.get("regions", []):
        block_type, writing_type = _region_to_block_type(r)
        blocks.append({
            "bbox_2d": _to_norm_bbox(r["bbox"], w, h),
            "block_type": block_type,
            "writing_type": writing_type,
        })
    response = json.dumps({"blocks": blocks}, ensure_ascii=False)
    return _chat_sample(image_path, DETECT_PROMPT, response)


def build_transcribe_samples(item: dict, image_path: str, *,
                             crop_dir: Path, margin: float = 0.05) -> list[dict]:
    try:
        img = load_image_exif(image_path)
    except Exception:
        return []
    out = []
    page_stem = Path(item["file_name"]).stem
    for i, r in enumerate(item.get("regions", [])):
        text = (r.get("text") or "").strip()
        if not text:
            continue
        if r.get("legibility") == "illegible" or r.get("language", "uk") == "other":
            continue
        if r.get("type") in ("image", "graph"):
            continue
        crop = crop_with_margin(img, r["bbox"], margin)
        cw, ch = crop.size
        if min(cw, ch) < 16:
            continue
        if max(cw, ch) < 128:
            scale = 256 / max(cw, ch)
            crop = crop.resize((int(cw * scale), int(ch * scale)), Image.LANCZOS)
        crop_path = crop_dir / f"{page_stem}_{i:03d}.jpg"
        if not crop_path.exists():
            crop.save(crop_path, format="JPEG", quality=92)
        response = json.dumps({"text": text}, ensure_ascii=False)
        out.append(_chat_sample(str(crop_path), TRANSCRIBE_PROMPT, response))
    return out


def _write_jsonl(path: Path, items: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["detect", "transcribe", "both"], default="both")
    ap.add_argument("--train-jsonl", default="data/cv/train_clean.jsonl")
    ap.add_argument("--silver-jsonl", default="data/cv/silver_clean.jsonl")
    ap.add_argument("--val-jsonl", default="data/cv/val.jsonl")
    ap.add_argument("--out-root", default="data/lora")
    ap.add_argument("--max-pages", type=int, default=0,
                    help="Cap pages per split (0 = no cap). Use for smoke tests.")
    ap.add_argument("--include-silver", action="store_true",
                    help="Mix silver_clean into the train output.")
    args = ap.parse_args()

    train_items = read_jsonl(args.train_jsonl)
    val_items = read_jsonl(args.val_jsonl)
    if args.include_silver and os.path.exists(args.silver_jsonl):
        silver_items = read_jsonl(args.silver_jsonl)
        for it in silver_items:
            it["_origin"] = "silver"
        for it in train_items:
            it["_origin"] = "train"
        train_items = train_items + silver_items
        print(f"  mixed silver in: total train = {len(train_items)}")

    out_root = Path(args.out_root)
    cap = args.max_pages or None

    tasks = ["detect", "transcribe"] if args.task == "both" else [args.task]
    for task in tasks:
        print(f"\n=== {task} / train ===")
        _process_split(train_items, "train", task, out_root, cap)
        print(f"=== {task} / val ===")
        _process_split(val_items, "val", task, out_root, cap)
    return 0


def _process_split(items, split, task, out_root, cap):
    crop_dir = out_root / task / f"_crops_{split}"
    crop_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict] = []
    for it in tqdm(items[:cap] if cap else items, desc=f"{task}/{split}"):
        hf_split = it.get("_origin", "train")
        if hf_split not in ("train", "silver"):
            hf_split = "train"
        try:
            image_path = materialize_image(hf_split, it["file_name"])
        except Exception as e:
            print(f"skip {it['file_name']}: {e}")
            continue
        if task == "detect":
            s = build_detect_sample(it, image_path)
            if s is not None:
                samples.append(s)
        else:
            samples.extend(build_transcribe_samples(it, image_path, crop_dir=crop_dir))
    out_path = out_root / task / f"{split}.jsonl"
    n = _write_jsonl(out_path, samples)
    print(f"  wrote {n} samples → {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
