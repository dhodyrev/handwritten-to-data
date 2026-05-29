"""Sync detect → transcribe pipeline.

Config-driven (PipelineConfig). Phase 0 = baseline parity. Phase 1 toggles:
- ``deskew`` (image.deskew)
- ``nms_iou`` (postproc.nms_iou; None disables)
- per-source ``crop_margins`` (transcribe.crop_margin per source)
- ``few_shot_examples`` (transcribe.few_shot per rtype)

In-process, single-threaded — Kaggle notebooks run one model on one GPU.
"""
from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field

from PIL import Image

from . import prompts as P
from .backend import DEFAULT_LANGUAGE, log, qwen_call
from .image_ops import (
    crop_with_margin,
    deskew,
    encode_jpeg,
    encode_jpeg_b64,
    load_image_exif,
)
from .postproc import nms_tasks
from .schemas import Region, RegionTask

# ── Config ─────────────────────────────────────────────────────────────────

# Default per-source crop margins for transcribe.
DEFAULT_CROP_MARGINS = {
    "school": 0.05,
    "university": 0.05,
    "dictation": 0.05,
    "archive": 0.03,
    "default": 0.05,
}


@dataclass
class PipelineConfig:
    # Image
    deskew: bool = False
    deskew_max_angle_deg: float = 10.0
    # Detection post-processing
    nms_iou: float | None = None      # e.g. 0.5; None disables
    # Transcription
    crop_margins: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CROP_MARGINS))
    few_shot_examples: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    # Routing
    math_shortcircuit_min_lines: int = 3  # math per-line must yield > N lines or fall back


# ── JSON parsing helpers ──────────────────────────────────────────────────

def parse_json(raw: str) -> list | dict | None:
    if not raw or raw.startswith("[ERROR:"):
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"[\[\{][\s\S]*[\]\}]", raw)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                return None
        return None


def _scale_bbox(bbox: list, w: int, h: int,
                off_x: int = 0, off_y: int = 0) -> list[int] | None:
    if len(bbox) < 4:
        return None
    x1 = int(bbox[0] * w / 1000) + off_x
    y1 = int(bbox[1] * h / 1000) + off_y
    x2 = int(bbox[2] * w / 1000) + off_x
    y2 = int(bbox[3] * h / 1000) + off_y
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return [x1, y1, x2, y2]


def _items_from_payload(raw: str, *keys: str) -> list:
    """Pull a list out of Qwen JSON, tolerating bare-array outputs."""
    data = parse_json(raw)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in keys:
            if k in data:
                return data[k] or []
        return []
    return []


def _parse_blocks(raw: str, img_w: int, img_h: int) -> list[dict]:
    blocks = []
    for item in _items_from_payload(raw, "blocks", "regions"):
        bbox = item.get("bbox_2d", item.get("bbox", []))
        scaled = _scale_bbox(bbox, img_w, img_h)
        if scaled is None:
            continue
        blocks.append({
            "bbox": scaled,
            "block_type":   item.get("block_type", "text_block"),
            "writing_type": item.get("writing_type", "handwritten"),
        })
    return blocks


def _parse_lines(raw: str, crop_w: int, crop_h: int,
                 off_x: int, off_y: int) -> list[list[int]]:
    out: list[list[int]] = []
    for item in _items_from_payload(raw, "lines", "blocks"):
        bbox = item.get("bbox_2d", item.get("bbox", item)) if isinstance(item, dict) else item
        scaled = _scale_bbox(bbox, crop_w, crop_h, off_x, off_y)
        if scaled is not None:
            out.append(scaled)
    return out


def _parse_perline(raw: str, img_w: int, img_h: int) -> list[RegionTask]:
    tasks: list[RegionTask] = []
    for item in _items_from_payload(raw, "blocks", "lines", "regions"):
        bbox = item.get("bbox_2d", item.get("bbox", []))
        scaled = _scale_bbox(bbox, img_w, img_h)
        if scaled is None:
            continue
        block_type = item.get("block_type", "text_block")
        rtype = "handwritten" if block_type == "text_block" else block_type
        tasks.append({"bbox": scaled, "rtype": rtype, "legibility": "legible"})
    return tasks


_TRANSCRIBE_VARIANTS = {
    "formula":    (P.QWEN_FORMULA_PROMPT,    1024),
    "table":      (P.QWEN_TABLE_PROMPT,      1024),
    "annotation": (P.QWEN_ANNOTATION_PROMPT,  512),
}


def _prompt_for(rtype: str, cfg: PipelineConfig) -> tuple[str, int]:
    base, max_tok = _TRANSCRIBE_VARIANTS.get(rtype, (P.QWEN_TRANSCRIBE_PROMPT, 1024))
    examples = cfg.few_shot_examples.get(rtype) or []
    return P.with_few_shot(base, examples), max_tok


def _crop_transcribe(img: Image.Image, bbox: list[int], rtype: str,
                     cfg: PipelineConfig, source: str) -> str:
    margin = cfg.crop_margins.get(source, cfg.crop_margins.get("default", 0.05))
    crop = crop_with_margin(img, bbox, margin)
    cw, ch = crop.size
    if max(cw, ch) < 128:
        scale = 256 / max(cw, ch)
        crop = crop.resize((int(cw * scale), int(ch * scale)), Image.LANCZOS)
    crop_b64 = base64.b64encode(encode_jpeg(crop)).decode()

    prompt, max_tok = _prompt_for(rtype, cfg)
    raw = qwen_call(crop_b64, prompt, max_tokens=max_tok)
    if raw.startswith("[ERROR:"):
        return raw
    parsed = parse_json(raw)
    if parsed and isinstance(parsed, dict):
        return parsed.get("text", "") or ""
    return "[ERROR:parse_failed]"


# ── Pipeline steps ─────────────────────────────────────────────────────

def _classify_page(b64: str) -> str:
    raw = qwen_call(b64, P.QWEN_CLASSIFY_PAGE_PROMPT, max_tokens=64)
    parsed = parse_json(raw)
    if isinstance(parsed, dict):
        return "math" if parsed.get("page_type") == "math" else "text"
    return "text"


def _detect_perline(b64: str, source: str,
                    img_w: int, img_h: int) -> list[RegionTask]:
    prompt = (P.QWEN_UNIVERSITY_PERLINE_PROMPT if source == "university"
              else P.QWEN_SCHOOL_PERLINE_PROMPT)
    raw = qwen_call(b64, prompt)
    return _parse_perline(raw, img_w, img_h)


def _lines_in_block(img: Image.Image, block: dict) -> list[RegionTask]:
    btype = block["block_type"]
    bbox = block["bbox"]

    if btype == "text_block":
        wtype = block.get("writing_type", "handwritten")
        img_w, img_h = img.size
        x1, y1, x2, y2 = bbox
        mx, my = int((x2 - x1) * 0.03), int((y2 - y1) * 0.03)
        cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
        cx2, cy2 = min(img_w, x2 + mx), min(img_h, y2 + my)
        crop = img.crop((cx1, cy1, cx2, cy2))
        crop_w, crop_h = crop.size
        crop_b64 = base64.b64encode(encode_jpeg(crop)).decode()
        raw = qwen_call(crop_b64, P.QWEN_LINE_DETECT_PROMPT)
        lines = _parse_lines(raw, crop_w, crop_h, cx1, cy1)
        if not lines:
            return [{"bbox": bbox, "rtype": wtype, "legibility": "legible"}]
        return [{"bbox": lb, "rtype": wtype, "legibility": "legible"}
                for lb in lines]

    return [{"bbox": bbox, "rtype": btype, "legibility": "legible"}]


def _detect(img: Image.Image, b64: str, source: str,
            cfg: PipelineConfig) -> list[RegionTask]:
    img_w, img_h = img.size

    if source in ("school", "university"):
        page_type = _classify_page(b64)
        log(f"    page type: {page_type}")
        if page_type == "math":
            tasks = _detect_perline(b64, source, img_w, img_h)
            if len(tasks) >= cfg.math_shortcircuit_min_lines:
                log(f"    per-line: {len(tasks)} lines")
                if cfg.nms_iou is not None:
                    before = len(tasks)
                    tasks = nms_tasks(tasks, cfg.nms_iou)
                    if before != len(tasks):
                        log(f"    NMS: {before} → {len(tasks)}")
                return tasks
            log(f"    per-line yielded {len(tasks)}, falling back to block→line")

    raw = qwen_call(b64, P.QWEN_BLOCK_DETECT_PROMPT)
    blocks = _parse_blocks(raw, img_w, img_h)
    log(f"    blocks: {len(blocks)}")
    tasks: list[RegionTask] = []
    for b in blocks:
        tasks.extend(_lines_in_block(img, b))
    if cfg.nms_iou is not None:
        before = len(tasks)
        tasks = nms_tasks(tasks, cfg.nms_iou)
        if before != len(tasks):
            log(f"    NMS: {before} → {len(tasks)}")
    log(f"    regions: {len(tasks)}")
    return tasks


def _transcribe(img: Image.Image, tasks: list[RegionTask],
                source: str, cfg: PipelineConfig) -> list[Region]:
    regions: list[Region] = []
    for t in tasks:
        if t["rtype"] in ("image", "graph") or t["legibility"] == "illegible":
            text = ""
        else:
            text = _crop_transcribe(img, t["bbox"], t["rtype"], cfg, source)
            text = (text or "").split("\n")[0].strip()
        regions.append({
            "type": t["rtype"],
            "bbox": t["bbox"],
            "language": DEFAULT_LANGUAGE,
            "legibility": t["legibility"],
            "transcription": text,
        })
    return regions


# ── Public runner ──────────────────────────────────────────────────────

def run_pipeline(image_path: str, uuid: str, source: str,
                 cfg: PipelineConfig | None = None) -> dict:
    """Detect blocks/lines, transcribe each. Sequential — single GPU."""
    cfg = cfg or PipelineConfig()

    img = load_image_exif(image_path)
    if cfg.deskew:
        img = deskew(img, max_angle_deg=cfg.deskew_max_angle_deg)
    img_w, img_h = img.size
    b64 = encode_jpeg_b64(img)
    log(f"\n  {source}/{uuid[:20]}  {img_w}×{img_h}"
        + (" [deskewed]" if cfg.deskew else ""))

    tasks = _detect(img, b64, source, cfg)
    regions = _transcribe(img, tasks, source, cfg)
    regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
    return {"uuid": uuid, "source": source, "regions": regions}


__all__ = [
    "PipelineConfig",
    "DEFAULT_CROP_MARGINS",
    "run_pipeline",
    "parse_json",
]
