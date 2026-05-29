"""YAML → PipelineConfig loader (kept tiny — no schema lib)."""
from __future__ import annotations

from pathlib import Path

import yaml

from .pipeline import DEFAULT_CROP_MARGINS, PipelineConfig


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    data = yaml.safe_load(Path(path).read_text())

    image = data.get("image") or {}
    postproc = data.get("postproc") or {}
    transcribe = data.get("transcribe") or {}
    routing = data.get("routing") or {}

    # Few-shot: YAML emits list[list[str,str]]; coerce inner to tuple for typing.
    fs_raw = transcribe.get("few_shot_examples") or {}
    few_shot = {k: [tuple(ex) for ex in v] for k, v in fs_raw.items()}

    crop_margins = dict(DEFAULT_CROP_MARGINS)
    crop_margins.update(transcribe.get("crop_margins") or {})

    return PipelineConfig(
        deskew=bool(image.get("deskew", False)),
        deskew_max_angle_deg=float(image.get("deskew_max_angle_deg", 10.0)),
        nms_iou=postproc.get("nms_iou"),
        crop_margins=crop_margins,
        few_shot_examples=few_shot,
        math_shortcircuit_min_lines=int(routing.get("math_shortcircuit_min_lines", 3)),
    )
