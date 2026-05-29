#!/usr/bin/env python
"""Run the Qwen3-VL pipeline on a CV split or arbitrary manifest.

Sequential single-GPU execution (one Unsloth model in-process). Designed for
Kaggle notebooks (dual T4 / single L4). Per-image JSON caches make re-runs
incremental.

Examples:
    # CV val with default pipeline config (HF auto-download)
    python scripts/run_inference.py --split val --out predictions/val_baseline.csv

    # Phase 1 toggles
    python scripts/run_inference.py --split val \\
        --config configs/pipeline_p1.yaml \\
        --out predictions/val_p1.csv

    # Kaggle test set (images mounted at /kaggle/input/handwritten-to-data/test)
    python scripts/run_inference.py --manifest data/cv/test.jsonl \\
        --image-root /kaggle/input/handwritten-to-data/test \\
        --out submissions/test_v1.csv

    # Load a LoRA adapter on top of the base model
    python scripts/run_inference.py --split val \\
        --adapter /kaggle/working/lora_adapter \\
        --out predictions/val_lora.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tqdm.auto import tqdm

from htr.backend import DEFAULT_MODEL, load_model
from htr.config import load_pipeline_config
from htr.data import materialize_image, read_jsonl
from htr.pipeline import run_pipeline


def _clean(t: str) -> str:
    return "" if (not t or t.startswith("[ERROR:")) else t


def regions_to_submission(result: dict) -> list[dict]:
    return [
        {
            "bbox": r["bbox"],
            "type": r.get("type", "handwritten"),
            "text": _clean(r.get("transcription", "")),
        }
        for r in result.get("regions", [])
    ]


def process_one(item: dict, image_root: str | None,
                json_dir: Path, split_for_hf: str | None,
                cfg) -> dict:
    file_name = item["file_name"]
    uuid = Path(file_name).stem

    cache_path = json_dir / f"{uuid}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    if image_root:
        image_path = str(Path(image_root) / file_name)
    elif split_for_hf:
        image_path = materialize_image(split_for_hf, file_name)
    else:
        raise RuntimeError("must pass either --image-root or --split (for HF auto-download)")

    try:
        result = run_pipeline(image_path, uuid, item.get("source", "unknown"), cfg)
    except Exception as e:
        print(f"ERROR {uuid}: {e}")
        result = {"uuid": uuid, "source": item.get("source", "unknown"), "regions": []}

    cache_path.write_text(json.dumps(result, ensure_ascii=False))
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"],
                    help="Read manifest from data/cv/{split}.jsonl (HF auto-download).")
    ap.add_argument("--manifest", help="Explicit JSONL manifest path.")
    ap.add_argument("--image-root", help="Directory holding images (for --manifest).")
    ap.add_argument("--out", required=True, help="Output submission CSV path.")
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HuggingFace model id (default: Unsloth Qwen2.5-VL-7B 4bit).")
    ap.add_argument("--adapter", default=None, help="Path to a trained LoRA adapter.")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not args.split and not args.manifest:
        ap.error("pass --split or --manifest")

    # Resolve manifest
    if args.manifest:
        manifest_path = Path(args.manifest)
        hf_split = None
    else:
        manifest_path = Path("data/cv") / f"{args.split}.jsonl"
        hf_split = "train"
    items = read_jsonl(manifest_path)
    if args.limit:
        items = items[: args.limit]
    print(f"Manifest: {len(items)} items from {manifest_path}")

    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    json_dir = out_csv.parent / (out_csv.stem + "_jsons")
    json_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_pipeline_config(args.config)
    print(f"Pipeline config: {cfg}")

    # Warm the model so the first per-image latency is honest.
    load_model(args.model, adapter_path=args.adapter)

    results = []
    for it in tqdm(items, desc="inference"):
        results.append(process_one(it, args.image_root, json_dir, hf_split, cfg))

    rows = []
    for it, res in zip(items, results, strict=False):
        rows.append({
            "image": Path(it["file_name"]).name,
            "regions": json.dumps(regions_to_submission(res or {}), ensure_ascii=False),
        })
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "regions"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_csv}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
