"""JSON schemas (kept for documentation / future reuse) + TypedDicts."""
from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


BLOCK_TYPES = ["text_block", "table", "formula", "image", "graph", "annotation"]

_BBOX_SCHEMA = {
    "type": "array", "items": {"type": "integer"},
    "minItems": 4, "maxItems": 4,
}

PAGE_TYPE_SCHEMA = {
    "type": "object",
    "properties": {"page_type": {"type": "string", "enum": ["text", "math"]}},
    "required": ["page_type"],
}

BLOCKS_SCHEMA = {
    "type": "object",
    "properties": {
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "bbox_2d":      _BBOX_SCHEMA,
                    "block_type":   {"type": "string", "enum": BLOCK_TYPES},
                    "writing_type": {"type": "string",
                                     "enum": ["handwritten", "printed"]},
                },
                "required": ["bbox_2d", "block_type", "writing_type"],
            },
        },
    },
    "required": ["blocks"],
}

LINES_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"bbox_2d": _BBOX_SCHEMA},
                "required": ["bbox_2d"],
            },
        },
    },
    "required": ["lines"],
}

PERLINE_SCHEMA = BLOCKS_SCHEMA

TRANSCRIBE_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}


# ── In-process pipeline state types ────────────────────────────────────

class RegionTask(TypedDict, total=False):
    bbox: list[int]
    rtype: str       # handwritten|printed|formula|table|annotation|image|graph
    legibility: str  # legible|illegible
    multiline: bool  # text_block to transcribe as N lines + split into bands


class Region(TypedDict):
    type: str
    bbox: list[int]
    language: str
    legibility: str
    transcription: str
