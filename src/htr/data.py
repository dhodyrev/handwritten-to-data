"""HF dataset access + CV split construction.

Dataset: UkrainianCatholicUniversity/rukopys (train, silver, test).
Metadata-only download by default (a few MB); image bytes are pulled lazily
on first access via huggingface_hub.hf_hub_download (cached in HF_HOME).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

from huggingface_hub import hf_hub_download

REPO_ID = "UkrainianCatholicUniversity/rukopys"
SPLITS = ("train", "silver", "test")


def download_metadata(split: str, *, repo_id: str = REPO_ID) -> str:
    """Download split/metadata.jsonl and return the local path."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    return hf_hub_download(
        repo_id=repo_id, filename=f"{split}/metadata.jsonl", repo_type="dataset",
    )


def iter_metadata(split: str, *, repo_id: str = REPO_ID) -> Iterator[dict]:
    path = download_metadata(split, repo_id=repo_id)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def download_image(split: str, file_name: str, *, repo_id: str = REPO_ID) -> str:
    """Pull a single image file. Cached under HF_HOME."""
    return hf_hub_download(
        repo_id=repo_id, filename=f"{split}/{file_name}", repo_type="dataset",
    )


# ── CV split construction ─────────────────────────────────────────────

def _stable_hash(s: str) -> int:
    """Stable hash so the same file_name always lands in the same bucket
    regardless of dict iteration / Python version / PYTHONHASHSEED."""
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)


def build_cv_split(
    items: Iterable[dict],
    *,
    val_frac: float = 0.10,
    stratify_by: str = "source",
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Hash-stratified split. Each source contributes ``val_frac`` of its items
    to val. Hash bucketing means re-runs and additions are stable.
    """
    by_strat: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_strat[it.get(stratify_by, "_")].append(it)

    train: list[dict] = []
    val: list[dict] = []
    threshold = int(val_frac * (2**32))
    salt = f"htr-cv-{seed}".encode()

    for _, group in sorted(by_strat.items()):
        # Sort for determinism, then bucket by stable hash + salt.
        for it in sorted(group, key=lambda x: x["file_name"]):
            h = _stable_hash(it["file_name"] + ":" + salt.decode())
            if (h % (2**32)) < threshold:
                val.append(it)
            else:
                train.append(it)

    # Light shuffle inside train/val for batching variety (seeded).
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def write_jsonl(path: str | Path, items: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def materialize_image(split: str, file_name: str, *, repo_id: str = REPO_ID) -> str:
    """Idempotently fetch an image to the HF cache; return local path."""
    return download_image(split, file_name, repo_id=repo_id)


def source_summary(items: Iterable[dict]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for it in items:
        out[it.get("source", "_")] += 1
    return dict(sorted(out.items()))
