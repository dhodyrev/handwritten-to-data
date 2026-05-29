"""Image loading, area-based resize for Qwen, JPEG b64 encode, deskew, crop helpers."""
from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps

from .backend import MAX_PIXELS


def resize_for_qwen(img: Image.Image, max_pixels: int = MAX_PIXELS) -> Image.Image:
    """Scale image so total pixel count ≤ max_pixels (matches server-side limit).

    Area-based, not max(w,h)-based: a 4284×5712 page clamped to max(w,h)=2048
    is still 3.1M pixels — 2× the server limit. Server then rescales again.
    """
    w, h = img.size
    pixels = w * h
    if pixels <= max_pixels:
        return img
    scale = (max_pixels / pixels) ** 0.5
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def encode_jpeg(img: Image.Image, *, quality: int = 90) -> bytes:
    buf = BytesIO()
    resize_for_qwen(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def encode_jpeg_b64(img: Image.Image) -> str:
    return base64.b64encode(encode_jpeg(img)).decode()


def load_image_exif(path: str) -> Image.Image:
    """Open + auto-rotate EXIF orientation + convert RGB."""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


# ── Phase 1: auto-deskew ───────────────────────────────────────────────

def deskew(img: Image.Image, *, max_angle_deg: float = 10.0) -> Image.Image:
    """Hough-based skew correction. Returns img unchanged when |angle| is tiny
    or OpenCV isn't installed. Bounded at ±max_angle_deg so we never spin a
    page upside down."""
    try:
        import cv2  # noqa: WPS433 — optional dep, kept lazy
    except ImportError:
        return img

    gray = np.array(img.convert("L"))
    # Edges: threshold to ink, invert so text=1.
    _, bin_ = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(bin_ > 0))
    if len(coords) < 100:
        return img  # too sparse — likely blank or mostly graphic

    # minAreaRect angle is in [-90, 0); normalize to a small skew estimate.
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.3 or abs(angle) > max_angle_deg:
        return img

    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        np.array(img), m, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )
    return Image.fromarray(rotated)


def crop_with_margin(img: Image.Image, bbox: list[int], margin: float) -> Image.Image:
    """Crop bbox expanded by `margin` fraction on each side, clamped to image."""
    img_w, img_h = img.size
    x1, y1, x2, y2 = bbox
    mx = int((x2 - x1) * margin)
    my = int((y2 - y1) * margin)
    return img.crop((
        max(0, x1 - mx), max(0, y1 - my),
        min(img_w, x2 + mx), min(img_h, y2 + my),
    ))
