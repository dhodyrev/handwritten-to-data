"""Sync in-process Qwen3-VL backend via Unsloth 4-bit.

All GPU dependencies (torch, unsloth, transformers, PIL→tensor pipeline) are
lazy-imported inside ``load_model`` and ``qwen_call`` so the library still
imports cleanly on a CPU-only Mac. Importing ``htr.backend`` is safe
everywhere; only calling ``qwen_call`` requires the GPU stack.

Single global model handle — Kaggle runs one process per notebook, batching
across images is sequential. JSON-mode guardrails (xgrammar guided_json) are
gone; we rely on the model + permissive parsing in ``htr.pipeline.parse_json``.
"""
from __future__ import annotations

import base64
import re
import threading
from dataclasses import dataclass
from io import BytesIO
from typing import Any

# Image token budget. For Qwen3-VL: tokens ≈ pixels / (28*28).
# 2048*28*28 → ≤2048 image tokens. Mirrors the prior vLLM cap so prompts and
# resize logic in image_ops stay in sync.
MAX_PIXELS = 2048 * 28 * 28
MIN_PIXELS = 512 * 28 * 28

DEFAULT_LANGUAGE = "uk"
DEFAULT_MODEL = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"

_print_lock = threading.Lock()


def log(*args: Any, **kwargs: Any) -> None:
    with _print_lock:
        print(*args, **kwargs, flush=True)


@dataclass
class _ModelHandle:
    model: Any
    processor: Any
    name: str
    adapter_path: str | None = None


_HANDLE: _ModelHandle | None = None


def load_model(
    model_name: str = DEFAULT_MODEL,
    *,
    adapter_path: str | None = None,
    max_seq_length: int = 4096,
    load_in_4bit: bool = True,
) -> _ModelHandle:
    """Load (or return cached) Unsloth FastVisionModel. Idempotent."""
    global _HANDLE
    if (_HANDLE is not None
            and _HANDLE.name == model_name
            and _HANDLE.adapter_path == adapter_path):
        return _HANDLE

    from unsloth import FastVisionModel  # noqa: WPS433 — lazy, GPU-only

    log(f"[backend] loading {model_name} (4bit={load_in_4bit})"
        + (f" + adapter {adapter_path}" if adapter_path else ""))
    # When an adapter is given, load the adapter repo directly: Unsloth reads
    # base_model_name_or_path from its adapter_config.json and materialises
    # (base + LoRA) in one shot. This is the canonical Unsloth inference path
    # and deliberately avoids transformers' model.load_adapter(), which
    # KeyErrors on 'qwen2_vl' inside _convert_peft_config_moe on recent
    # transformers (it assumes every model is MoE).
    model, processor = FastVisionModel.from_pretrained(
        adapter_path or model_name,
        load_in_4bit=load_in_4bit,
        max_seq_length=max_seq_length,
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_inference(model)
    _HANDLE = _ModelHandle(model=model, processor=processor,
                           name=model_name, adapter_path=adapter_path)
    return _HANDLE


def _decode_b64_to_pil(image_b64: str):
    from PIL import Image  # noqa: WPS433

    raw = base64.b64decode(image_b64)
    return Image.open(BytesIO(raw)).convert("RGB")


_CLEAN_THINK = re.compile(r"<think>[\s\S]*?</think>")
_CLEAN_FENCE_OPEN = re.compile(r"```json\s*")
_CLEAN_FENCE_CLOSE = re.compile(r"```\s*$")


def _strip(raw: str) -> str:
    raw = _CLEAN_THINK.sub("", raw).strip()
    raw = _CLEAN_FENCE_OPEN.sub("", raw)
    raw = _CLEAN_FENCE_CLOSE.sub("", raw).strip()
    return raw


def qwen_call(image_b64: str, prompt: str, *,
              schema: dict | None = None,
              max_tokens: int = 4000) -> str:
    """Run Qwen3-VL on one image + prompt. Returns raw text or ``[ERROR:*]``.

    ``schema`` is accepted for signature compatibility with the old vLLM
    guided_json path — Unsloth does not enforce it, so it's ignored. Callers
    rely on permissive JSON parsing in ``htr.pipeline.parse_json``.
    """
    del schema  # unused; signature kept for callsite parity

    try:
        import torch  # noqa: WPS433

        handle = _HANDLE if _HANDLE is not None else load_model()
        model, processor = handle.model, handle.processor

        image = _decode_b64_to_pil(image_b64)
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(
            text=[text], images=[image], return_tensors="pt", padding=True,
        ).to(model.device)

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.0,
                use_cache=True,
            )
        gen = out[:, inputs["input_ids"].shape[1]:]
        decoded = processor.batch_decode(gen, skip_special_tokens=True)[0]
        return _strip(decoded)
    except Exception as e:  # noqa: BLE001 — never raise into pipeline
        log(f"    QWEN failed: {str(e)[:160]}")
        return "[ERROR:qwen_failed]"


__all__ = [
    "DEFAULT_LANGUAGE",
    "DEFAULT_MODEL",
    "MAX_PIXELS",
    "MIN_PIXELS",
    "load_model",
    "log",
    "qwen_call",
]
